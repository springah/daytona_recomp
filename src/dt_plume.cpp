#include "dt_plume.h"

#include "dt_shader_blobs.h"

#include <rex/logging.h>

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "plume_render_interface.h"

namespace plume {
// Defined in plume_d3d12.cpp / plume_vulkan.cpp; plume declares neither in a
// public header, so the example does the same thing.
extern std::unique_ptr<RenderInterface> CreateD3D12Interface();
extern std::unique_ptr<RenderInterface> CreateVulkanInterface();
}  // namespace plume

namespace dt_plume {
namespace {

std::mutex g_mutex;
bool g_init_attempted = false;
bool g_init_ok = false;

std::unique_ptr<plume::RenderInterface> g_interface;
std::unique_ptr<plume::RenderDevice> g_device;
plume::RenderShaderFormat g_shader_format = plume::RenderShaderFormat::UNKNOWN;
const char* g_backend = "none";

struct Created {
    const dt_shader_blobs::Entry* entry = nullptr;
    std::unique_ptr<plume::RenderShader> shader;
};
std::unordered_map<uint64_t, Created> g_shaders;

uint32_t g_resolved = 0;
uint32_t g_unresolved = 0;
uint32_t g_failed = 0;

// ── Draw-state shadow ────────────────────────────────────────────────────────
// The D3D9 calls all arrive on one render thread (every trace dump so far shows
// a single thread id), so the shadow is plain globals with no locking -- at ~6k
// calls/frame a mutex per call would be a real cost. That assumption is CHECKED
// below rather than trusted: if a second thread ever touches it, we shout.
constexpr uint32_t kMaxStreams = 16;
constexpr uint32_t kMaxCombos = 64;

struct StreamBinding {
    uint32_t buffer_ea = 0;
    uint32_t offset = 0;
    uint32_t stride = 0;
};

// Per-combo record of what a draw ACTUALLY had bound. Captured at draw time,
// not at dump time: SetStreamSource passes a null buffer on ~85% of calls, so a
// snapshot taken between frames shows everything unbound and tells you nothing.
constexpr uint32_t kMaxComboStreams = 6;

struct ComboStream {
    uint32_t stream = 0;
    uint32_t stride = 0;
};

struct Combo {
    uint64_t vs = 0;
    uint64_t ps = 0;
    uint32_t topology = 0;
    uint64_t draws = 0;
    uint64_t vertices = 0;
    uint32_t min_count = 0xFFFFFFFFu;
    uint32_t max_count = 0;
    ComboStream streams[kMaxComboStreams];
    uint32_t stream_count = 0;
    uint32_t stream_overflow = 0;
    uint32_t decl_ea = 0;
    uint64_t decl_mismatch = 0;
};

std::unordered_map<uint32_t, uint64_t> g_shader_objects;   // guest object ea -> container hash
StreamBinding g_streams[kMaxStreams];
uint32_t g_cur_vs_ea = 0;
uint32_t g_cur_ps_ea = 0;
uint32_t g_cur_decl_ea = 0;
Combo g_combos[kMaxCombos];
uint32_t g_combo_count = 0;
uint64_t g_draws = 0;
uint64_t g_draws_unresolved = 0;     // a draw whose VS or PS we have no blob for
uint32_t g_combo_overflow = 0;
uint32_t g_decl_changes = 0;

constexpr uint32_t kMaxDecls = 8;
struct DeclBind {
    uint32_t ea = 0;
    uint64_t binds = 0;
};
DeclBind g_decls[kMaxDecls];
uint32_t g_decl_count = 0;
uint32_t g_decl_overflow = 0;

unsigned long g_render_thread = 0;
bool g_thread_warned = false;

// Portable and cheap: a thread_local counter, so the hot path is one TLS read.
unsigned long CurrentThreadIdCheap() {
    static std::atomic<unsigned long> next{1};
    static thread_local unsigned long id = next.fetch_add(1, std::memory_order_relaxed);
    return id;
}

// Returns false the first time a second thread shows up, and shouts once.
bool OnRenderThread() {
    const unsigned long id = CurrentThreadIdCheap();
    if (g_render_thread == 0) {
        g_render_thread = id;
        return true;
    }
    if (g_render_thread != id) {
        if (!g_thread_warned) {
            g_thread_warned = true;
            REXLOG_WARN("plume: DRAW STATE TOUCHED FROM A SECOND THREAD ({} then {}) -- the "
                        "lock-free shadow is unsafe; the census below may be wrong",
                        g_render_thread, id);
        }
        return false;
    }
    return true;
}

uint64_t HashOf(uint32_t shader_ea) {
    if (shader_ea == 0) return 0;
    auto it = g_shader_objects.find(shader_ea);
    return it == g_shader_objects.end() ? ~0ull : it->second;
}

const char* TopologyName(uint32_t t) {
    switch (t) {
        case 1:  return "POINTLIST";
        case 2:  return "LINELIST";
        case 3:  return "LINESTRIP";
        case 4:  return "TRIANGLELIST";
        case 5:  return "TRIANGLESTRIP";
        case 6:  return "TRIANGLEFAN";
        case 13: return "QUADLIST";
        default: return "?";
    }
}

// XenosRecomp invokes DXC without -E, so the entry point is the default and the
// generated HLSL declares `void main(...)`.
constexpr const char* kEntryPoint = "main";

const char* UsageName(uint8_t usage) {
    switch (usage) {
        case 0:  return "POSITION";
        case 3:  return "NORMAL";
        case 5:  return "TEXCOORD";
        case 6:  return "TANGENT";
        case 10: return "COLOR";
        case 12: return "DEPTH";
        default: return "?";
    }
}

}  // namespace

bool Enabled() {
    static const bool on = std::getenv("DT_PLUME") != nullptr;
    return on;
}

bool Init() {
    if (!Enabled()) return false;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_init_attempted) return g_init_ok;
    g_init_attempted = true;

    REXLOG_WARN("DT_PLUME set -- bringing up the native renderer (shader resolve only, no drawing)");

#ifdef _WIN32
    g_interface = plume::CreateD3D12Interface();
    g_backend = "D3D12";
#endif
    if (g_interface == nullptr) {
        g_interface = plume::CreateVulkanInterface();
        g_backend = "Vulkan";
    }
    if (g_interface == nullptr) {
        REXLOG_WARN("plume: no render interface could be created");
        return false;
    }

    g_device = g_interface->createDevice();
    if (g_device == nullptr) {
        REXLOG_WARN("plume: {} interface created but createDevice() failed", g_backend);
        g_interface.reset();
        return false;
    }

    g_shader_format = g_interface->getCapabilities().shaderFormat;
    const auto& desc = g_device->getDescription();
    REXLOG_WARN("plume: {} device ready -- \"{}\" (vendor {}), shaderFormat={}",
                g_backend, desc.name, static_cast<uint32_t>(desc.vendor),
                static_cast<uint32_t>(g_shader_format));

    // The blobs are DXIL and SPIR-V only. Metal would need a third target from
    // XenosRecomp, so say so plainly rather than failing shader-by-shader later.
    if (g_shader_format != plume::RenderShaderFormat::DXIL &&
        g_shader_format != plume::RenderShaderFormat::SPIRV) {
        REXLOG_WARN("plume: backend wants shaderFormat={} but dt_shader_blobs only has DXIL+SPIRV",
                    static_cast<uint32_t>(g_shader_format));
        g_device.reset();
        g_interface.reset();
        return false;
    }

    g_init_ok = true;
    return true;
}

void OnShaderCreated(uint64_t container_hash, bool is_vertex) {
    if (!Enabled()) return;
    if (!Init()) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_shaders.count(container_hash)) return;   // already built

    const dt_shader_blobs::Entry* entry = dt_shader_blobs::Find(container_hash);
    if (entry == nullptr) {
        ++g_unresolved;
        REXLOG_WARN("plume: NO PRECOMPILED SHADER for {} container hash {:016X} -- "
                    "re-run gen_shader_blobs.py, the game created a shader we have never seen",
                    is_vertex ? "VS" : "PS", container_hash);
        return;
    }
    if (entry->is_vertex != is_vertex) {
        ++g_failed;
        REXLOG_WARN("plume: STAGE MISMATCH for {}: table says {}, hook says {}",
                    entry->name, entry->is_vertex ? "VS" : "PS", is_vertex ? "VS" : "PS");
        return;
    }

    const void* data = nullptr;
    uint64_t size = 0;
    if (g_shader_format == plume::RenderShaderFormat::DXIL) {
        data = dt_shader_blobs::kDxil + entry->dxil_offset;
        size = entry->dxil_size;
    } else {
        // SPIR-V in the table is smol-v encoded; decoding it needs smol-v, which
        // is not linked yet. D3D12 is the first target, so this is a stub.
        REXLOG_WARN("plume: {} is SPIR-V (smol-v) and the decoder is not linked yet -- skipped",
                    entry->name);
        ++g_failed;
        return;
    }

    Created created;
    created.entry = entry;
    created.shader = g_device->createShader(data, size, kEntryPoint, g_shader_format);
    if (created.shader == nullptr) {
        ++g_failed;
        REXLOG_WARN("plume: createShader FAILED for {} ({} bytes)", entry->name, size);
        return;
    }

    ++g_resolved;
    if (entry->element_count) {
        std::string elems;
        for (uint32_t i = 0; i < entry->element_count; ++i) {
            const auto& e = entry->elements[i];
            elems += UsageName(e.usage);
            elems += static_cast<char>('0' + (e.usage_index % 10));
            elems += ' ';
        }
        REXLOG_WARN("plume: created VS {} ({} B) inputs: {}", entry->name, size, elems);
    } else {
        REXLOG_WARN("plume: created PS {} ({} B)", entry->name, size);
    }

    g_shaders.emplace(container_hash, std::move(created));
}

void OnShaderObject(uint32_t shader_object_ea, uint64_t container_hash) {
    if (!Enabled() || shader_object_ea == 0) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_shader_objects[shader_object_ea] = container_hash;
}

void OnSetVertexShader(uint32_t shader_ea) {
    if (!Enabled()) return;
    if (!OnRenderThread()) return;
    g_cur_vs_ea = shader_ea;
}

void OnSetPixelShader(uint32_t shader_ea) {
    if (!Enabled()) return;
    if (!OnRenderThread()) return;
    g_cur_ps_ea = shader_ea;
}

void OnSetVertexDeclaration(uint32_t decl_ea) {
    if (!Enabled()) return;
    if (!OnRenderThread()) return;
    if (decl_ea != g_cur_decl_ea) {
        ++g_decl_changes;
        g_cur_decl_ea = decl_ea;
    }
    // Distinct declarations matter: the spec says ONE game-wide, but the change
    // count runs at 2x the draw count, which means it alternates between two
    // values (very likely real <-> null). Record what they are.
    for (uint32_t i = 0; i < g_decl_count; ++i) {
        if (g_decls[i].ea == decl_ea) {
            ++g_decls[i].binds;
            return;
        }
    }
    if (g_decl_count < kMaxDecls) {
        g_decls[g_decl_count].ea = decl_ea;
        g_decls[g_decl_count].binds = 1;
        ++g_decl_count;
    } else {
        ++g_decl_overflow;
    }
}

void OnSetStreamSource(uint32_t stream, uint32_t buffer_ea, uint32_t offset, uint32_t stride) {
    if (!Enabled()) return;
    if (!OnRenderThread()) return;
    if (stream >= kMaxStreams) return;
    g_streams[stream] = {buffer_ea, offset, stride};
}

void OnDrawVertices(uint32_t primitive_type, uint32_t start_vertex, uint32_t vertex_count) {
    if (!Enabled()) return;
    if (!OnRenderThread()) return;
    (void)start_vertex;
    ++g_draws;

    const uint64_t vs = HashOf(g_cur_vs_ea);
    const uint64_t ps = HashOf(g_cur_ps_ea);
    if (vs == ~0ull || ps == ~0ull) ++g_draws_unresolved;

    Combo* found = nullptr;
    for (uint32_t i = 0; i < g_combo_count; ++i) {
        Combo& c = g_combos[i];
        if (c.vs == vs && c.ps == ps && c.topology == primitive_type) {
            found = &c;
            break;
        }
    }
    if (found == nullptr) {
        if (g_combo_count >= kMaxCombos) {
            ++g_combo_overflow;   // never silently truncate -- reported in Dump()
            return;
        }
        found = &g_combos[g_combo_count++];
        found->vs = vs;
        found->ps = ps;
        found->topology = primitive_type;
        found->decl_ea = g_cur_decl_ea;
    }
    ++found->draws;
    found->vertices += vertex_count;
    // A combination is assumed to use one declaration; count it if that is ever
    // false rather than silently reporting whichever one happened to be first.
    if (g_cur_decl_ea != found->decl_ea) ++found->decl_mismatch;
    if (vertex_count < found->min_count) found->min_count = vertex_count;
    if (vertex_count > found->max_count) found->max_count = vertex_count;

    // Fold in whatever is actually bound right now.
    for (uint32_t s = 0; s < kMaxStreams; ++s) {
        if (g_streams[s].buffer_ea == 0) continue;
        bool seen = false;
        for (uint32_t k = 0; k < found->stream_count; ++k) {
            if (found->streams[k].stream == s && found->streams[k].stride == g_streams[s].stride) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        if (found->stream_count >= kMaxComboStreams) {
            ++found->stream_overflow;
            continue;
        }
        found->streams[found->stream_count++] = {s, g_streams[s].stride};
    }
}

void Dump(const char* reason) {
    if (!Enabled()) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    REXLOG_WARN("===== plume renderer ({}) =====", reason);
    if (!g_init_ok) {
        REXLOG_WARN("   device NOT initialised");
    } else {
        REXLOG_WARN("   backend={} shaderFormat={}", g_backend,
                    static_cast<uint32_t>(g_shader_format));
    }
    REXLOG_WARN("   shaders created {} / {} known    unresolved={} failed={}",
                g_resolved, dt_shader_blobs::kEntryCount, g_unresolved, g_failed);
    // The full set is known and closed (confirmed under race load), so anything
    // short of all of them is a regression worth shouting about.
    if (g_init_ok && g_resolved != dt_shader_blobs::kEntryCount) {
        REXLOG_WARN("   NOTE: {} of {} not yet created -- expected only if the game has not "
                    "reached the code that creates them",
                    dt_shader_blobs::kEntryCount - g_resolved, dt_shader_blobs::kEntryCount);
    }

    REXLOG_WARN("   draws={} unresolved={} decl changes={} combos={}",
                g_draws, g_draws_unresolved, g_decl_changes, g_combo_count);
    if (g_combo_overflow) {
        REXLOG_WARN("   WARNING: {} draws fell outside the {}-combo table -- the census below "
                    "is INCOMPLETE, raise kMaxCombos", g_combo_overflow, kMaxCombos);
    }
    if (g_draws_unresolved) {
        REXLOG_WARN("   WARNING: {} draws used a shader object we never saw created", g_draws_unresolved);
    }

    // This is the number the renderer actually needs: one plume pipeline per row.
    REXLOG_WARN("   --- pipeline combinations (one plume pipeline each) ---");
    for (uint32_t i = 0; i < g_combo_count; ++i) {
        const Combo& c = g_combos[i];
        const auto* ve = c.vs == ~0ull ? nullptr : dt_shader_blobs::Find(c.vs);
        const auto* pe = c.ps == ~0ull ? nullptr : dt_shader_blobs::Find(c.ps);
        std::string streams;
        for (uint32_t k = 0; k < c.stream_count; ++k) {
            streams += "s" + std::to_string(c.streams[k].stream) +
                       "/" + std::to_string(c.streams[k].stride) + " ";
        }
        if (c.stream_overflow) streams += "(+more) ";
        REXLOG_WARN("   [{:2}] {:>13} vs={:<20} ps={:<20} draws={:<9} verts {}..{} decl={:08X}{} streams: {}",
                    i, TopologyName(c.topology),
                    ve ? ve->name : (c.vs == 0 ? "(none)" : "(UNKNOWN)"),
                    pe ? pe->name : (c.ps == 0 ? "(none)" : "(UNKNOWN)"),
                    c.draws, c.min_count == 0xFFFFFFFFu ? 0 : c.min_count, c.max_count,
                    c.decl_ea, c.decl_mismatch ? "(VARIES!)" : "",
                    streams.empty() ? "(none bound)" : streams);
    }
    REXLOG_WARN("   --- vertex declarations bound ({} distinct) ---", g_decl_count);
    for (uint32_t i = 0; i < g_decl_count; ++i) {
        REXLOG_WARN("   decl {:08X}  binds={}", g_decls[i].ea, g_decls[i].binds);
    }
    if (g_decl_overflow) REXLOG_WARN("   (+{} more distinct declarations)", g_decl_overflow);
    REXLOG_WARN("===== end plume renderer =====");
}

void Shutdown() {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_shaders.clear();
    g_device.reset();
    g_interface.reset();
    g_init_ok = false;
}

}  // namespace dt_plume

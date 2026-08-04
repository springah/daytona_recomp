#include "dt_plume.h"

#include "dt_shader_blobs.h"

#include <rex/logging.h>

#include <cstdlib>
#include <cstring>
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
std::unique_ptr<plume::RenderPipelineLayout> g_pipeline_layout;

// Static expansion index buffers. Neither depends on vertex data, so one per
// topology serves every draw. Peak observed VertexCount is 1624 (census);
// 4096 leaves headroom and still fits 16-bit indices.
enum class IndexState { kNone, kReady, kFailed };
constexpr uint32_t kMaxExpandVerts = 4096;
constexpr uint32_t kSnapshotsPerShader = 2;
IndexState g_index_state = IndexState::kNone;
std::unique_ptr<plume::RenderBuffer> g_quad_indices;
std::unique_ptr<plume::RenderBuffer> g_fan_indices;
plume::RenderShaderFormat g_shader_format = plume::RenderShaderFormat::UNKNOWN;
const char* g_backend = "none";

// One decoded vfetch, i.e. one vertex input attribute.
struct FetchedAttribute {
    uint8_t usage = 0;
    uint8_t usage_index = 0;
    uint32_t instruction = 0;   // microcode instruction index (element `address`)
    uint32_t format = 0;        // GPUVERTEXFORMAT
    uint32_t stride = 0;        // bytes (the instruction stores dwords)
    uint32_t offset = 0;        // bytes
    bool num_format_all = false;
    bool signed_all = false;
    // Fetch constant = constIndex*3 + constIndexSelect. It identifies the
    // STREAM, which is what an input slot must key on. Keying on stride
    // instead collapses a de-interleaved layout: every attribute sits at
    // offset 0 with the same stride, so they would all alias slot 0.
    uint32_t fetch_const = 0;
};

struct Created {
    const dt_shader_blobs::Entry* entry = nullptr;
    std::unique_ptr<plume::RenderShader> shader;

    uint32_t object_ea = 0;
    std::vector<uint8_t> code;   // microcode as created (pre-patch)
    uint32_t code_ea = 0;        // where the LIVE copy lives, once located
    bool located = false;
    bool layout_done = false;
    bool diagnosed_stored = false;      // DeriveVertexLayouts' unpatched dump
    bool diagnosed_incremental = false; // OnIncrementalPatch's anchor dump
    bool logged_placeholder = false;    // ucode still holds placeholder operands
    std::vector<FetchedAttribute> attributes;
    uint32_t snapshots = 0;
    bool snapshot_overflow = false;
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

enum class PipelineState { kPending, kBuilt, kFailed };

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
    PipelineState pipeline_state = PipelineState::kPending;
    std::unique_ptr<plume::RenderPipeline> pipeline;
};

std::unordered_map<uint32_t, uint64_t> g_shader_objects;   // guest object ea -> container hash
StreamBinding g_streams[kMaxStreams];
uint32_t g_cur_vs_ea = 0;
uint32_t g_cur_ps_ea = 0;
uint32_t g_cur_decl_ea = 0;
uint32_t g_device_ea = 0;   // guest CDevice*, r3 on every D3D9 entry point
// Latched by OnDrawVertices BEFORE __imp__ runs. The post-draw snapshot
// cannot re-read g_cur_vs_ea: the draw flushes state through these same
// hooks, so by then it names whatever the flush bound last.
uint64_t g_draw_vs = 0;
uint32_t g_draw_start = 0;
uint32_t g_draw_count = 0;
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
uint64_t g_incremental_calls = 0;

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

uint32_t LoadBE32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}

// Image is 0x82000000-based; the guest heap this title uses runs 0xA0000000+.
bool PlausibleGuestPointer(uint32_t v) {
    return (v >= 0x82000000u && v < 0x83500000u) || (v >= 0xA0000000u && v < 0xD0000000u);
}

// GPUVERTEXFORMAT (d3d9gpu.h). Only the ones this title has been seen to use are
// named; anything else prints its number so it shows up as unhandled rather than
// being silently mapped to something plausible.
const char* VertexFormatName(uint32_t f) {
    switch (f) {
        case 6:  return "8_8_8_8";
        case 36: return "32_FLOAT";
        case 37: return "32_32_FLOAT";
        case 57: return "32_32_32_FLOAT";
        case 38: return "32_32_32_32_FLOAT";
        default: return "UNHANDLED";
    }
}

// GPUVERTEXFORMAT -> plume vertex input format.
//
// Returns UNKNOWN for anything unseen rather than guessing: a wrong vertex
// format is silent corruption, not a crash.
plume::RenderFormat ToRenderFormat(uint32_t gpu_format, bool num_format_all, bool signed_all) {
    switch (gpu_format) {
        case 57:  // 32_32_32_FLOAT
            return plume::RenderFormat::R32G32B32_FLOAT;
        case 36:  // 32_FLOAT -- Daytona's DEPTH0 attribute
            return plume::RenderFormat::R32_FLOAT;
        case 37:  // 32_32_FLOAT
            return plume::RenderFormat::R32G32_FLOAT;
        case 38:  // 32_32_32_32_FLOAT
            return plume::RenderFormat::R32G32B32A32_FLOAT;
        case 6:   // 8_8_8_8
            // numFormatAll selects FRACTION (normalised) over INTEGER.
            if (!num_format_all) {
                return signed_all ? plume::RenderFormat::R8G8B8A8_SINT
                                  : plume::RenderFormat::R8G8B8A8_UINT;
            }
            return signed_all ? plume::RenderFormat::R8G8B8A8_SNORM
                              : plume::RenderFormat::R8G8B8A8_UNORM;
        default:
            return plume::RenderFormat::UNKNOWN;
    }
}

// Bytes per component, which is what the endian swap needs.
//
// ⚠ Guest vertex data is BIG-ENDIAN. The swap is per COMPONENT, not a blanket
// dword swap over the buffer: dword-swapping an 8_8_8_8 colour reverses its
// channels (RGBA becomes ABGR) and there is no DXGI format to undo that. So an
// all-32-bit layout happens to be a uniform dword swap, but the moment an
// attribute has 8- or 16-bit components the pattern differs per attribute --
// which is precisely why the derived layout is needed for the UPLOAD path, not
// only for the input layout.
// Total bytes an attribute occupies, needed to bound the swap.
uint32_t AttributeBytes(uint32_t gpu_format) {
    switch (gpu_format) {
        case 6:  return 4;    // 8_8_8_8
        case 36: return 4;    // 32_FLOAT
        case 37: return 8;    // 32_32_FLOAT
        case 57: return 12;   // 32_32_32_FLOAT
        case 38: return 16;   // 32_32_32_32_FLOAT
        default: return 0;
    }
}

uint32_t ComponentBytes(uint32_t gpu_format) {
    switch (gpu_format) {
        case 57: case 37: case 38: return 4;
        case 6:                    return 1;
        default:                   return 0;   // unknown: refuse to swap
    }
}

// Semantic name and Vulkan location MUST match what XenosRecomp emitted into the
// shader, or D3D12 fails to match by semantic and Vulkan fails to match by
// location. Mirrored from its USAGE_SEMANTICS / USAGE_LOCATIONS tables --
// including the Depth entry added by xenosrecomp-main-daytona.patch, which
// upstream lacks because Sonic Unleashed never used Depth as a vertex element.
const char* SemanticName(uint8_t usage) {
    switch (usage) {
        case 0:  return "POSITION";
        case 1:  return "BLENDWEIGHT";
        case 2:  return "BLENDINDICES";
        case 3:  return "NORMAL";
        case 4:  return "PSIZE";
        case 5:  return "TEXCOORD";
        case 6:  return "TANGENT";
        case 7:  return "BINORMAL";
        case 10: return "COLOR";
        case 12: return "DEPTH";
        default: return "TEXCOORD";
    }
}

uint32_t VulkanLocation(uint8_t usage, uint8_t index) {
    switch (usage) {
        case 0:  return index == 0 ? 0u : 15u;              // Position0 / Position1
        case 3:  return 1;                                   // Normal0
        case 6:  return 2;                                   // Tangent0
        case 7:  return 3;                                   // Binormal0
        case 5:  return index < 4 ? 4u + index : 12u + (index - 4u);   // TexCoord0..7
        case 10: return index == 0 ? 8u : 11u;               // Color0 / Color1
        case 2:  return 9;                                   // BlendIndices0
        case 1:  return 10;                                  // BlendWeight0
        case 12: return 16;                                  // Depth0 -- the local patch
        default: return 0;
    }
}

// Provisional until the swapchain is wired: the guest presents B8G8R8A8 and the
// pixel shaders write SV_Depth, so a depth target must exist for them to compile
// against. Revisit when the render targets are real.
constexpr plume::RenderFormat kRenderTargetFormat = plume::RenderFormat::B8G8R8A8_UNORM;
constexpr plume::RenderFormat kDepthTargetFormat = plume::RenderFormat::D32_FLOAT;

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

void OnShaderObject(uint32_t shader_object_ea, uint64_t container_hash,
                    const uint8_t* code, uint32_t code_size) {
    if (!Enabled() || shader_object_ea == 0) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_shader_objects[shader_object_ea] = container_hash;
    auto it = g_shaders.find(container_hash);
    if (it == g_shaders.end()) return;
    it->second.object_ea = shader_object_ea;
    if (code != nullptr && code_size != 0 && it->second.code.empty()) {
        it->second.code.assign(code, code + code_size);
    }
}

void OnIncrementalPatch(uint32_t shader_ea, uint32_t decl_ea, uint32_t ucode_ea,
                        GuestResolve resolve, void* ctx) {
    if (!Enabled()) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_incremental_calls;
    // Log the first few unconditionally: "no output" must be distinguishable
    // from "never called". The menu takes DirectShaderPatch, so zero calls there
    // is the expected result, not a broken hook.
    {
        auto oit = g_shader_objects.find(shader_ea);
        const char* who = "(unknown object)";
        if (oit != g_shader_objects.end()) {
            auto e = dt_shader_blobs::Find(oit->second);
            if (e) who = e->name;
        }
        // Log the first call FOR EACH DISTINCT SHADER, not just the first few
        // overall -- the shader that matters may not be among the first.
        static std::unordered_map<uint32_t, bool> seen;
        if (!seen[shader_ea]) {
            seen[shader_ea] = true;
            REXLOG_WARN("plume: IncrementalShaderPatchAndLoad first call for {} "
                        "(shader={:08X} decl={:08X} ucode={:08X})",
                        who, shader_ea, decl_ea, ucode_ea);
        }
    }
    if (ucode_ea == 0 || resolve == nullptr) return;

    auto it = g_shader_objects.find(shader_ea);
    if (it == g_shader_objects.end()) return;
    auto sit = g_shaders.find(it->second);
    if (sit == g_shaders.end() || sit->second.entry == nullptr) return;
    Created& s = sit->second;
    if (!s.entry->is_vertex || s.layout_done) return;

    const uint8_t* ucode = resolve(ucode_ea, ctx);
    if (ucode == nullptr) return;

    // The ucode buffer is not the container, so the instruction stream does not
    // necessarily start at offset 0. Find it by locating the vfetch whose
    // operands are the ones we know: dstRegister and constIndex survive
    // patching, so the pre-patch word0 of the FIRST element is a reliable
    // anchor, and everything else is at a fixed 3-dword stride from it.
    const uint32_t anchor_d = uint32_t(s.entry->elements[0].address) * 3;
    if ((anchor_d + 3) * 4 > s.code.size()) return;
    const uint32_t anchor_w0 = LoadBE32(&s.code[anchor_d * 4]);

    int32_t base_dword = -1;
    for (uint32_t i = 0; i < 512 && base_dword < 0; ++i) {
        if (LoadBE32(ucode + i * 4) == anchor_w0) {
            base_dword = int32_t(i) - int32_t(anchor_d);
        }
    }
    if (base_dword < 0) {
        if (!s.diagnosed_incremental) {
            s.diagnosed_incremental = true;
            REXLOG_WARN("plume: {} incremental patch -- anchor {:08X} not found in ucode@{:08X}; "
                        "first 12 dwords: {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} "
                        "{:08X} {:08X} {:08X} {:08X} {:08X} {:08X}",
                        s.entry->name, anchor_w0, ucode_ea,
                        LoadBE32(ucode +  0), LoadBE32(ucode +  4), LoadBE32(ucode +  8),
                        LoadBE32(ucode + 12), LoadBE32(ucode + 16), LoadBE32(ucode + 20),
                        LoadBE32(ucode + 24), LoadBE32(ucode + 28), LoadBE32(ucode + 32),
                        LoadBE32(ucode + 36), LoadBE32(ucode + 40), LoadBE32(ucode + 44));
        }
        return;
    }

    std::vector<FetchedAttribute> attrs;
    for (uint32_t e = 0; e < s.entry->element_count; ++e) {
        const auto& el = s.entry->elements[e];
        const uint32_t d = uint32_t(base_dword) + uint32_t(el.address) * 3;
        const uint32_t w0 = LoadBE32(ucode + d * 4);
        const uint32_t w1 = LoadBE32(ucode + (d + 1) * 4);
        const uint32_t w2 = LoadBE32(ucode + (d + 2) * 4);
        FetchedAttribute a;
        a.usage = el.usage;
        a.usage_index = el.usage_index;
        a.instruction = el.address;
        a.format = (w1 >> 16) & 0x3F;
        a.num_format_all = ((w1 >> 13) & 1) != 0;
        a.signed_all = ((w1 >> 14) & 1) != 0;
        a.stride = (w2 & 0xFF) * 4;
        a.offset = ((w2 >> 8) & 0x7FFFFF) * 4;
        a.fetch_const = ((w0 >> 20) & 0x1F) * 3 + ((w0 >> 25) & 0x3);
        if (a.format == 0 && a.stride == 0) {
            if (!s.logged_placeholder) {
                s.logged_placeholder = true;
                REXLOG_WARN("plume: {} incremental ucode@{:08X}+{} STILL PLACEHOLDER at "
                            "insn {} (w1={:08X} w2={:08X}) -- r8 is likely the SOURCE "
                            "ucode, with the patched copy going into the PM4 stream",
                            s.entry->name, ucode_ea, base_dword, el.address,
                            LoadBE32(ucode + (d + 1) * 4), LoadBE32(ucode + (d + 2) * 4));
            }
            return;
        }
        attrs.push_back(a);
    }
    if (attrs.empty()) return;

    s.attributes = std::move(attrs);
    s.layout_done = true;
    REXLOG_WARN("plume: vertex layout for {} via INCREMENTAL patch (ucode @{:08X}+{} dwords, decl={:08X})",
                s.entry->name, ucode_ea, base_dword, decl_ea);
    for (const auto& a : s.attributes) {
        const auto rf = ToRenderFormat(a.format, a.num_format_all, a.signed_all);
        REXLOG_WARN("   {}{}  insn {:<3} format {:<3} ({})  stride {:<4} offset {:<4} -> plume {}",
                    UsageName(a.usage), a.usage_index, a.instruction,
                    a.format, VertexFormatName(a.format), a.stride, a.offset,
                    static_cast<uint32_t>(rf));
        if (rf == plume::RenderFormat::UNKNOWN) {
            REXLOG_WARN("      ^^ UNMAPPED GPUVERTEXFORMAT {}", a.format);
        }
    }
}

bool BuildIndexBuffers() {
    if (!Enabled() || !g_init_ok) return false;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_index_state != IndexState::kNone) return g_index_state == IndexState::kReady;

    // Peak observed VertexCount is 1624 (census). kMaxExpandVerts is the ceiling
    // this supports; a draw beyond it is reported, never silently clipped.
    std::vector<uint16_t> quad, fan;
    quad.reserve((kMaxExpandVerts / 4) * 6);
    for (uint32_t q = 0; q + 4 <= kMaxExpandVerts; q += 4) {
        // Split on the 0-2 diagonal, preserving perimeter winding.
        const uint16_t b = uint16_t(q);
        quad.insert(quad.end(), {b, uint16_t(b + 1), uint16_t(b + 2),
                                 b, uint16_t(b + 2), uint16_t(b + 3)});
    }
    fan.reserve((kMaxExpandVerts - 2) * 3);
    for (uint32_t i = 1; i + 1 < kMaxExpandVerts; ++i) {
        fan.insert(fan.end(), {0, uint16_t(i), uint16_t(i + 1)});
    }

    auto upload = [&](const std::vector<uint16_t>& src,
                      std::unique_ptr<plume::RenderBuffer>& dst) -> bool {
        const uint64_t bytes = src.size() * sizeof(uint16_t);
        dst = g_device->createBuffer(plume::RenderBufferDesc::IndexBuffer(
            bytes, plume::RenderHeapType::UPLOAD));
        if (dst == nullptr) return false;
        void* p = dst->map();
        if (p == nullptr) return false;
        std::memcpy(p, src.data(), size_t(bytes));
        dst->unmap();
        return true;
    };

    if (!upload(quad, g_quad_indices) || !upload(fan, g_fan_indices)) {
        g_index_state = IndexState::kFailed;
        REXLOG_WARN("plume: failed to create the expansion index buffers");
        return false;
    }
    g_index_state = IndexState::kReady;
    REXLOG_WARN("plume: index buffers ready -- quad {} indices ({} quads), fan {} indices, "
                "max {} vertices/draw",
                quad.size(), quad.size() / 6, fan.size(), kMaxExpandVerts);
    return true;
}

void SnapshotDrawVertices(const uint8_t* base) {
    const uint32_t start_vertex = g_draw_start;
    const uint32_t vertex_count = g_draw_count;
    if (!Enabled() || !g_init_ok || base == nullptr || vertex_count == 0) return;
    const auto resolve = [base](uint32_t ea) -> const uint8_t* {
        return ea ? base + ea + (ea >= 0xE0000000u ? 0x1000u : 0u) : nullptr;
    };
    std::lock_guard<std::mutex> lock(g_mutex);

    // Find the combination this draw belongs to; it carries the layout and the
    // bindings. Only the current one is needed, and only a few times.
    auto sit = g_shaders.find(g_draw_vs);
    if (sit == g_shaders.end() || !sit->second.layout_done) return;
    Created& s = sit->second;
    if (s.snapshots >= kSnapshotsPerShader) return;

    if (vertex_count > kMaxExpandVerts) {
        if (!s.snapshot_overflow) {
            s.snapshot_overflow = true;
            REXLOG_WARN("plume: {} draw of {} vertices exceeds the {}-vertex expansion ceiling",
                        s.entry->name, vertex_count, kMaxExpandVerts);
        }
        return;
    }

    ++s.snapshots;
    REXLOG_WARN("plume: vertex snapshot {} for {} (start {}, {} verts)",
                s.snapshots, s.entry->name, start_vertex, vertex_count);

    // ⚠ SetStreamSource's pStreamData is a D3DVertexBuffer OBJECT, not the data.
    // Reading it as vertices yields the object header (a refcount reads as
    // "80 00 00 00"). The real byte address is BaseAddress in the vertex fetch
    // constant, which the runtime programs from SetStreamSource -- exactly what
    // rexglue's translator does ("extract the byte address from the fetch
    // constant", masking off the low 2 bits).
    //
    // The fetch constant array is at device+1152: SetVertexShader computes it as
    // `addi r28,r30,1152`, and it is the same array the sampler work found at
    // device+(sampler+48)*24 -- a texture constant is 24 bytes, i.e. three
    // 8-byte vertex constants.
    if (g_device_ea == 0) {
        REXLOG_WARN("   (no device pointer latched -- snapshot cannot resolve anything)");
        return;
    }
    for (const auto& a : s.attributes) {
        const uint32_t stream = (a.fetch_const <= 95u) ? (95u - a.fetch_const) : UINT32_MAX;
        const uint8_t* fc = resolve(g_device_ea + 1152u + a.fetch_const * 8u);
        if (fc == nullptr) continue;
        const uint32_t base_addr = LoadBE32(fc) & ~uint32_t(3);
        if (base_addr == 0) {
            // Never silently skip: every early exit in a diagnostic must say why.
            REXLOG_WARN("   {}{} fetch const {} @{:08X} reads {:08X} {:08X} -- BaseAddress is 0, "
                        "so device+1152+fc*8 is not where this title keeps them",
                        UsageName(a.usage), a.usage_index, a.fetch_const,
                        g_device_ea + 1152u + a.fetch_const * 8u,
                        LoadBE32(fc), LoadBE32(fc + 4));
            continue;
        }
        const uint32_t stride = (stream < kMaxStreams && g_streams[stream].stride)
                                    ? g_streams[stream].stride : a.stride;
        const uint32_t comp = ComponentBytes(a.format);
        const uint32_t ea = base_addr + start_vertex * stride + a.offset;
        const uint8_t* p = resolve(ea);
        if (p == nullptr || comp == 0) continue;

        // Byte-swap the first vertex's worth of this attribute and print it, so
        // the swap width can be checked against plausible values rather than
        // assumed correct.
        uint8_t v[16] = {};
        const uint32_t width = AttributeBytes(a.format);
        if (width == 0 || width > sizeof(v)) continue;
        for (uint32_t b = 0; b < width; ++b) {
            const uint32_t grp = b / comp, idx = b % comp;
            v[b] = p[grp * comp + (comp - 1 - idx)];   // reverse WITHIN each component
        }
        if (a.format == 57 || a.format == 37 || a.format == 36 || a.format == 38) {
            float f[4] = {};
            std::memcpy(f, v, width);
            REXLOG_WARN("   {}{} s{} @{:08X} stride {} -> {:.4f} {:.4f} {:.4f} {:.4f}",
                        UsageName(a.usage), a.usage_index, stream, ea, stride,
                        f[0], width > 4 ? f[1] : 0.0f, width > 8 ? f[2] : 0.0f,
                        width > 12 ? f[3] : 0.0f);
        } else {
            REXLOG_WARN("   {}{} s{} @{:08X} stride {} -> {:02X} {:02X} {:02X} {:02X}",
                        UsageName(a.usage), a.usage_index, stream, ea, stride,
                        v[0], v[1], v[2], v[3]);
        }
    }
}

void BuildPipelines(const char* reason) {
    if (!Enabled() || !g_init_ok) return;
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_pipeline_layout == nullptr) {
        plume::RenderPipelineLayoutDesc desc;
        desc.allowInputLayout = true;
        g_pipeline_layout = g_device->createPipelineLayout(desc);
        if (g_pipeline_layout == nullptr) {
            REXLOG_WARN("plume: createPipelineLayout failed; no pipelines can be built");
            return;
        }
    }

    uint32_t built = 0, waiting = 0, failed = 0;
    for (uint32_t i = 0; i < g_combo_count; ++i) {
        Combo& c = g_combos[i];
        if (c.pipeline_state != PipelineState::kPending) {
            built += (c.pipeline_state == PipelineState::kBuilt);
            failed += (c.pipeline_state == PipelineState::kFailed);
            continue;
        }
        auto vs = g_shaders.find(c.vs);
        auto ps = g_shaders.find(c.ps);
        if (vs == g_shaders.end() || ps == g_shaders.end() ||
            vs->second.shader == nullptr || ps->second.shader == nullptr ||
            !vs->second.layout_done) {
            ++waiting;
            continue;
        }

        // One input slot per distinct stride the layout uses. The UI shaders are
        // de-interleaved (every attribute at offset 0 in its own stream) while
        // the 3D path is a single interleaved buffer, so slots are assigned from
        // what the vfetch operands actually say rather than assumed.
        std::vector<plume::RenderInputSlot> slots;
        std::vector<uint32_t> slot_consts;
        std::vector<plume::RenderInputElement> elements;
        bool usable = true;
        for (const auto& a : vs->second.attributes) {
            const auto fmt = ToRenderFormat(a.format, a.num_format_all, a.signed_all);
            if (fmt == plume::RenderFormat::UNKNOWN) {
                REXLOG_WARN("plume: combo {} unusable -- {}{} has unmapped GPUVERTEXFORMAT {}",
                            i, UsageName(a.usage), a.usage_index, a.format);
                usable = false;
                break;
            }
            // One slot per FETCH CONSTANT (i.e. per stream). The 3D path has all
            // five attributes on one constant with distinct offsets -> one slot;
            // the UI paths have one constant each at offset 0 -> a slot each.
            uint32_t slot = UINT32_MAX;
            for (uint32_t s = 0; s < slot_consts.size(); ++s) {
                if (slot_consts[s] == a.fetch_const) { slot = s; break; }
            }
            if (slot == UINT32_MAX) {
                // STRIDE IS PER-DRAW, NOT PER-SHADER. The same shader is bound
                // with different strides: combo 0 uses s0/12 s3/4 s8/8 while
                // combo 1 uses s0/24 s3/24 s8/24, and the ground-truth vfetch
                // dump shows fetch constant 95 with 3, 4, 6 and 9 dwords. So the
                // stride comes from THIS combination's SetStreamSource bindings;
                // only the format and offset are properties of the shader.
                //
                // Fetch constants are allocated downward from 95, so
                // stream = 95 - fetch_const (observed: s0->95, s3->92, s8->87,
                // and the 3D path's single stream 0 -> 95). Validated below
                // rather than assumed.
                const uint32_t stream = (a.fetch_const <= 95u) ? (95u - a.fetch_const) : UINT32_MAX;
                uint32_t stride = 0;
                for (uint32_t k = 0; k < c.stream_count; ++k) {
                    if (c.streams[k].stream == stream) { stride = c.streams[k].stride; break; }
                }
                if (stride == 0) {
                    REXLOG_WARN("plume: combo {} {}{} maps to fetch const {} -> stream {}, which "
                                "this combination never bound; falling back to the shader's "
                                "stride {}", i, UsageName(a.usage), a.usage_index,
                                a.fetch_const, stream, a.stride);
                    stride = a.stride;
                } else if (stride != a.stride) {
                    REXLOG_WARN("plume: combo {} stream {} stride {} (bound) vs {} (shader) -- "
                                "using the bound one", i, stream, stride, a.stride);
                }
                slot = uint32_t(slots.size());
                slot_consts.push_back(a.fetch_const);
                slots.emplace_back(slot, stride);
            }
            elements.emplace_back(SemanticName(a.usage), a.usage_index,
                                  VulkanLocation(a.usage, a.usage_index), fmt, slot, a.offset);
        }
        if (!usable) { c.pipeline_state = PipelineState::kFailed; ++failed; continue; }

        plume::RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = g_pipeline_layout.get();
        desc.vertexShader = vs->second.shader.get();
        desc.pixelShader = ps->second.shader.get();
        desc.inputSlots = slots.data();
        desc.inputSlotsCount = uint32_t(slots.size());
        desc.inputElements = elements.data();
        desc.inputElementsCount = uint32_t(elements.size());
        desc.renderTargetFormat[0] = kRenderTargetFormat;
        desc.renderTargetBlend[0] = plume::RenderBlendDesc::Copy();
        desc.renderTargetCount = 1;
        desc.depthTargetFormat = kDepthTargetFormat;
        // QUADLIST and TRIANGLEFAN both become an indexed TRIANGLE_LIST: neither
        // exists in D3D12/Vulkan, and both expand with a static index pattern
        // (0,1,2, 0,2,3 per quad; 0,i,i+1 for a fan). See RENDERER_INPUT_SPEC.md.
        desc.primitiveTopology = plume::RenderPrimitiveTopology::TRIANGLE_LIST;

        c.pipeline = g_device->createGraphicsPipeline(desc);
        if (c.pipeline == nullptr) {
            c.pipeline_state = PipelineState::kFailed;
            ++failed;
            REXLOG_WARN("plume: pipeline {} FAILED ({} + {}, {})", i,
                        vs->second.entry->name, ps->second.entry->name, TopologyName(c.topology));
            continue;
        }
        c.pipeline_state = PipelineState::kBuilt;
        ++built;
        std::string desc_str;
        for (const auto& e : elements) {
            desc_str += std::string(e.semanticName) + std::to_string(e.semanticIndex) +
                        "@" + std::to_string(e.alignedByteOffset) +
                        "/s" + std::to_string(e.slotIndex) +
                        "(fc" + std::to_string(slot_consts[e.slotIndex]) + ") ";
        }
        REXLOG_WARN("plume: pipeline {} BUILT  {} + {}  {} -> TRIANGLE_LIST  slots={} : {}",
                    i, vs->second.entry->name, ps->second.entry->name,
                    TopologyName(c.topology), slots.size(), desc_str);
    }
    if (built || waiting || failed) {
        REXLOG_WARN("plume: pipelines ({}) -- built {} / waiting on layout {} / failed {} "
                    "(of {} observed combinations)",
                    reason, built, waiting, failed, g_combo_count);
    }
}

void DeriveVertexLayouts(const char* reason, GuestResolve resolve, void* ctx) {
    if (!Enabled() || resolve == nullptr) return;
    std::lock_guard<std::mutex> lock(g_mutex);

    uint32_t derived = 0, pending = 0;
    for (auto& [hash, s] : g_shaders) {
        if (s.entry == nullptr || !s.entry->is_vertex) continue;   // only vfetch matters
        if (s.layout_done) { ++derived; continue; }
        if (s.code.empty() || s.object_ea == 0) { ++pending; continue; }

        // Locate the live microcode: scan the object's leading dwords for a
        // pointer whose target matches the code captured at creation. Located,
        // not hardcoded -- object+32 held for every vertex shader observed, but
        // that is an observation, not a documented layout.
        if (!s.located) {
            const uint8_t* obj = resolve(s.object_ea, ctx);
            if (obj == nullptr) { ++pending; continue; }
            const size_t probe = s.code.size() < 32 ? s.code.size() : 32;
            for (uint32_t i = 0; i < 32 && !s.located; ++i) {
                const uint32_t cand = LoadBE32(obj + i * 4);
                if (!PlausibleGuestPointer(cand)) continue;
                const uint8_t* p = resolve(cand, ctx);
                if (p != nullptr && std::memcmp(p, s.code.data(), probe) == 0) {
                    s.code_ea = cand;
                    s.located = true;
                }
            }
            if (!s.located) { ++pending; continue; }
        }

        const uint8_t* code = resolve(s.code_ea, ctx);
        if (code == nullptr) { ++pending; continue; }

        // Decode one vfetch per declared vertex element. dt_shader_blobs carries
        // each element's microcode instruction index in `address`, and each
        // instruction is 3 dwords.
        std::vector<FetchedAttribute> attrs;
        bool all_patched = true;
        for (uint32_t e = 0; e < s.entry->element_count; ++e) {
            const auto& el = s.entry->elements[e];
            const uint32_t d = uint32_t(el.address) * 3;
            if ((d + 3) * 4 > s.code.size()) { all_patched = false; break; }
            const uint32_t w0 = LoadBE32(code + d * 4);
            const uint32_t w1 = LoadBE32(code + (d + 1) * 4);
            const uint32_t w2 = LoadBE32(code + (d + 2) * 4);
            FetchedAttribute a;
            a.usage = el.usage;
            a.usage_index = el.usage_index;
            a.instruction = el.address;
            a.format = (w1 >> 16) & 0x3F;
            a.num_format_all = ((w1 >> 13) & 1) != 0;
            a.signed_all = ((w1 >> 14) & 1) != 0;
            a.stride = (w2 & 0xFF) * 4;            // instruction stores dwords
            a.offset = ((w2 >> 8) & 0x7FFFFF) * 4;
            a.fetch_const = ((w0 >> 20) & 0x1F) * 3 + ((w0 >> 25) & 0x3);
            // format 0 and stride 0 is the unpatched placeholder the compiler
            // emits; that shader simply has not been bound yet.
            if (a.format == 0 && a.stride == 0) { all_patched = false; break; }
            attrs.push_back(a);
        }
        if (!all_patched || attrs.empty()) {
            ++pending;
            // vs_0f1a is bound for 94.6% of draws yet never patches, and the
            // three shaders that DO patch are exactly the three binding a real
            // vertex declaration -- so patching looks declaration-driven and
            // this path has decl=NULL. Dump what is actually there, and every
            // other plausible pointer in the object, in case the runtime keeps
            // a second (patched) microcode copy per declaration.
            if (!s.diagnosed_stored) {
                s.diagnosed_stored = true;
                REXLOG_WARN("plume: {} NOT PATCHED -- diagnostic (object@{:08X} code@{:08X})",
                            s.entry->name, s.object_ea, s.code_ea);
                for (uint32_t e = 0; e < s.entry->element_count; ++e) {
                    const uint32_t d = uint32_t(s.entry->elements[e].address) * 3;
                    if ((d + 3) * 4 > s.code.size()) continue;
                    REXLOG_WARN("   insn {:<3} live {:08X} {:08X} {:08X}   created {:08X} {:08X} {:08X}",
                                s.entry->elements[e].address,
                                LoadBE32(code + d * 4), LoadBE32(code + (d + 1) * 4),
                                LoadBE32(code + (d + 2) * 4),
                                LoadBE32(&s.code[d * 4]), LoadBE32(&s.code[(d + 1) * 4]),
                                LoadBE32(&s.code[(d + 2) * 4]));
                }
                const uint8_t* obj = resolve(s.object_ea, ctx);
                if (obj != nullptr) {
                    const size_t probe = s.code.size() < 32 ? s.code.size() : 32;
                    for (uint32_t i = 0; i < 48; ++i) {
                        const uint32_t cand = LoadBE32(obj + i * 4);
                        if (!PlausibleGuestPointer(cand)) continue;
                        const uint8_t* p = resolve(cand, ctx);
                        const bool match = p != nullptr && std::memcmp(p, s.code.data(), probe) == 0;
                        REXLOG_WARN("   object+{:<3} -> {:08X}{}", i * 4, cand,
                                    match ? "   *** MICROCODE COPY ***" : "");
                    }
                }
            }
            continue;
        }

        s.attributes = std::move(attrs);
        s.layout_done = true;
        ++derived;

        REXLOG_WARN("plume: vertex layout for {} (microcode @{:08X})", s.entry->name, s.code_ea);
        for (const auto& a : s.attributes) {
            const auto rf = ToRenderFormat(a.format, a.num_format_all, a.signed_all);
            const uint32_t comp = ComponentBytes(a.format);
            REXLOG_WARN("   {}{}  insn {:<3} format {:<3} ({})  stride {:<4} offset {:<4} "
                        "-> plume {} , endian swap {}",
                        UsageName(a.usage), a.usage_index, a.instruction,
                        a.format, VertexFormatName(a.format), a.stride, a.offset,
                        static_cast<uint32_t>(rf), comp ? comp : 0);
            if (rf == plume::RenderFormat::UNKNOWN) {
                REXLOG_WARN("      ^^ UNMAPPED GPUVERTEXFORMAT {} -- a wrong vertex format is "
                            "silent corruption, so this attribute is not usable yet", a.format);
            }
        }
    }
    if (derived || pending) {
        REXLOG_WARN("plume: vertex layouts ({}) -- derived {} / pending {} "
                    "(pending = shader not bound yet, so not patched yet)",
                    reason, derived, pending);
    }
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

void OnDrawVertices(uint32_t device_ea, uint32_t primitive_type, uint32_t start_vertex,
                    uint32_t vertex_count, const uint8_t* base) {
    if (!Enabled()) return;
    g_device_ea = device_ea;
    if (!OnRenderThread()) return;
    ++g_draws;

    const uint64_t vs = HashOf(g_cur_vs_ea);
    g_draw_vs = vs;
    g_draw_start = start_vertex;
    g_draw_count = vertex_count;
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

    // Bounded: a couple of draws per shader, purely to validate the endian swap
    // against plausible values before it becomes load-bearing.
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

    REXLOG_WARN("   draws={} unresolved={} decl changes={} combos={} incremental-patches={}",
                g_draws, g_draws_unresolved, g_decl_changes, g_combo_count, g_incremental_calls);
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
    for (uint32_t i = 0; i < g_combo_count; ++i) g_combos[i].pipeline.reset();
    g_pipeline_layout.reset();
    g_quad_indices.reset();
    g_fan_indices.reset();
    g_index_state = IndexState::kNone;
    g_device.reset();
    g_interface.reset();
    g_init_ok = false;
}

}  // namespace dt_plume

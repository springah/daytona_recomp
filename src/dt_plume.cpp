#include "dt_plume.h"

#include "dt_guest_mem.h"
#include "dt_shader_blobs.h"

#include <rex/logging.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "plume_render_interface.h"
#include "plume_render_interface_builders.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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
constexpr uint32_t kMaxLayoutVersionsLogged = 8;
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
    // ⚠ Both of these are decoded from the vfetch word, and both were wrong.
    // Ground truth is rexglue's own rex/graphics/format/ucode.h:716-718:
    //     is_signed()     { return fomat_comp_all == 1; }   // bit 12
    //     is_normalized() { return num_format_all == 0; }   // bit 13
    // and the XDK agrees on the polarity: GPUNUMFORMAT_FRACTION = 0,
    // GPUNUMFORMAT_INTEGER = 1 (d3d9gpu.h:609-610).
    //
    // Stored as the QUESTIONS ASKED rather than as raw bit names, because
    // "num_format_all" tells you nothing about which way round it goes and
    // that is precisely how it came to be inverted.
    bool normalized = false;   // num_format_all == 0
    bool is_signed = false;    // fomat_comp_all == 1, bit 12 -- NOT bit 14,
                               // which is signed_rf_mode_all (a rounding mode)
    // Fetch constant = constIndex*3 + constIndexSelect. It identifies the
    // STREAM, which is what an input slot must key on. Keying on stride
    // instead collapses a de-interleaved layout: every attribute sits at
    // offset 0 with the same stride, so they would all alias slot 0.
    uint32_t fetch_const = 0;

    // "Is this the same layout the runtime patched last time?" -- asked once
    // per draw, so it must not be a string compare or a re-decode.
    bool operator==(const FetchedAttribute& o) const {
        return usage == o.usage && usage_index == o.usage_index &&
               instruction == o.instruction && format == o.format && stride == o.stride &&
               offset == o.offset && normalized == o.normalized && is_signed == o.is_signed &&
               fetch_const == o.fetch_const;
    }
    bool operator!=(const FetchedAttribute& o) const { return !(*this == o); }
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
    // How many DISTINCT layouts the runtime has patched into this shader. More
    // than one is normal and is the whole reason a combination latches its own
    // copy; it is logged (bounded) because a shader that flips layouts is worth
    // knowing about when a combination renders wrong.
    uint32_t layout_versions = 0;
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
    // Packed RB_BLENDCONTROL, read from the device shadow (+10552, written by
    // D3DDevice_SetBlendState sub_8221F868 and the individual blend render
    // states) at draw time, and the colour write mask (+10460 bits 0..3).
    // Part of the combination KEY: one pipeline per distinct blend state.
    // Default = src ONE, op ADD, dst ZERO both halves = blending off.
    uint32_t blend = 0x00010001u;
    uint8_t write_mask = 0xFu;
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
    // The input layout this combination's pipeline was built with. Kept because
    // a draw has to bind vertex buffers with exactly the slots and strides the
    // pipeline expects, and the stride is a property of the COMBINATION (the
    // same shader is bound at 12/4/8 in one and 24/24/24 in another), so it
    // cannot be recomputed from the shader alone.
    std::vector<plume::RenderInputSlot> slots;
    std::vector<uint32_t> slot_consts;   // slot -> fetch constant (i.e. stream)
    // THE LAYOUT IS PER COMBINATION, NOT PER SHADER. D3D9 re-patches a shader's
    // vfetch every time the declaration or the stream strides change, so a
    // shader has as many layouts as it has uses: vs_bbb7 draws both from one
    // interleaved 24-byte buffer (offsets 0/12/16 on fetch constant 95) and
    // from three de-interleaved streams (offset 0 on constants 95/92/87).
    // Keeping one layout per SHADER means whichever use patched first wins and
    // the others read garbage. Latched on this combination's first recorded
    // draw -- the patch that ran for that draw is by definition its own.
    std::vector<FetchedAttribute> attributes;
    bool layout_latched = false;
    // Draws this combination has actually recorded into a frame -- distinct
    // from `draws` above, which counts every draw the GAME made whether or not
    // the renderer could reproduce it. Used only to bound logging now that
    // every draw is recorded rather than sampled.
    uint32_t recorded_draws = 0;
    bool warned_unreadable = false;
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
uint32_t g_draw_combo = 0xFFFFFFFFu;   // index into g_combos, latched the same way
Combo g_combos[kMaxCombos];
uint32_t g_combo_count = 0;
uint64_t g_draws = 0;
uint64_t g_draws_unresolved = 0;     // a draw whose VS or PS we have no blob for
// A draw thrown away because nothing can render it: its vertex shader's layout
// is not derived yet, or its combination's pipeline is missing. These were
// INVISIBLE -- both drop sites returned early without touching a counter, so a
// frame that dropped every piece of 3D geometry still reported "0 skipped",
// which reads as "nothing is missing" and is the opposite of the truth.
// Cumulative, not per-frame: the earlier of the two drop sites runs before
// BeginFrame, so there is no open frame to attribute it to.
uint64_t g_draws_nopipe = 0;
uint32_t g_combo_overflow = 0;
bool g_bisected = false;   // the pipeline-failure bisect runs once, not per combo
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
uint64_t g_vfetch_patches = 0;

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
        // ⚠ XENOS encoding (xenos.h:49-51), NOT the PC D3D9 one -- fan and
        // strip are REVERSED between them. Decoding 6 as FAN drew the title's
        // fullscreen strip-ordered quad with fan indices, which covers all of
        // the screen except a bottom-centre triangle: the black wedge.
        case 4:  return "TRIANGLELIST";
        case 5:  return "TRIANGLEFAN";
        case 6:  return "TRIANGLESTRIP";
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
plume::RenderFormat ToRenderFormat(uint32_t gpu_format, bool normalized, bool is_signed) {
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
            // ⚠ This was inverted, and it is why every draw came out black.
            //
            // D3DCOLOR / UBYTE4N are FRACTION, i.e. num_format_all == 0, i.e.
            // NORMALISED -- so the correct format is UNORM. Binding UINT means
            // D3D12's input assembler hands the shader raw 0..255 as integers
            // in a float register: those bit patterns are denormals (~1e-43)
            // and flush to zero. Every menu pixel shader multiplies by COLOR0
            // (verified in the generated HLSL), so COLOR0 == 0 makes every
            // output exactly zero in all four channels -- no API error, every
            // pipeline builds, and the readback reports full coverage with no
            // colour. Textures, samplers and constants are all irrelevant to
            // it, which is exactly why the controls kept coming back clean.
            if (normalized) {
                return is_signed ? plume::RenderFormat::R8G8B8A8_SNORM
                                 : plume::RenderFormat::R8G8B8A8_UNORM;
            }
            return is_signed ? plume::RenderFormat::R8G8B8A8_SINT
                             : plume::RenderFormat::R8G8B8A8_UINT;
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
// Sizes of the unbounded descriptor arrays the translated shaders declare. The
// game binds 8 samplers and, per the trace, has ~536 distinct textures; these
// are ceilings for the heap, not a measured requirement.
// Which shader specialisation to use. 0 = every spec constant off; the only one
// Daytona's shaders declare is ALPHA_TEST. See the note at the pick site.
constexpr uint32_t kSpecConstants = 0;

constexpr uint32_t kTextureDescriptorSize = 1024;
constexpr uint32_t kSamplerDescriptorSize = 64;

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

// Decode one vfetch instruction (3 big-endian dwords) into an attribute. The
// field polarities are the ones ucode.h:716-718 documents; see FetchedAttribute.
FetchedAttribute DecodeVfetch(uint32_t w0, uint32_t w1, uint32_t w2) {
    FetchedAttribute a;
    a.format = (w1 >> 16) & 0x3F;
    a.normalized = ((w1 >> 13) & 1) == 0;
    a.is_signed = ((w1 >> 12) & 1) != 0;
    a.stride = (w2 & 0xFF) * 4;              // the instruction stores dwords
    a.offset = ((w2 >> 8) & 0x7FFFFF) * 4;
    a.fetch_const = ((w0 >> 20) & 0x1F) * 3 + ((w0 >> 25) & 0x3);
    return a;
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
        // Pick the specialisation. Every pixel shader here has an ALPHA_TEST
        // spec constant, and D3D12 will not take the lib_6_3 that XenosRecomp
        // emits for those, so the table carries a real ps_6_0 per value.
        //
        // ⚠ PROVISIONAL: alpha test is a guest render state we do not shadow
        // yet, so this always takes the alpha-test-OFF build. Correct for the
        // geometry, wrong for anything that clips -- wire it to the shadow
        // state when the draw path reads render states.
        const auto* variant = dt_shader_blobs::FindVariant(*entry, kSpecConstants);
        if (variant == nullptr) {
            ++g_failed;
            REXLOG_WARN("plume: {} has no specialisation for spec constants {:#x} "
                        "(mask {:#x}, {} variants)", entry->name, kSpecConstants,
                        entry->spec_constants_mask, entry->variant_count);
            return;
        }
        data = dt_shader_blobs::kDxil + variant->dxil_offset;
        size = variant->dxil_size;
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

    // The anchor search below reads up to 512 dwords, so ask for all of it: a
    // 4-byte check in front of a 2 KB scan would not have caught anything.
    constexpr uint32_t kAnchorScanDwords = 512;
    const uint8_t* ucode = resolve(ucode_ea, kAnchorScanDwords * 4, ctx);
    if (ucode == nullptr) {
        if (!s.diagnosed_incremental) {
            s.diagnosed_incremental = true;
            REXLOG_WARN("plume: {} incremental patch -- ucode@{:08X} is not {} bytes of "
                        "readable guest memory, so the layout cannot be read from it",
                        s.entry->name, ucode_ea, kAnchorScanDwords * 4);
        }
        return;
    }

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
        a.normalized = ((w1 >> 13) & 1) == 0;   // num_format_all == 0
        a.is_signed = ((w1 >> 12) & 1) != 0;    // fomat_comp_all
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
        const auto rf = ToRenderFormat(a.format, a.normalized, a.is_signed);
        REXLOG_WARN("   {}{}  insn {:<3} format {:<3} ({})  stride {:<4} offset {:<4} -> plume {}",
                    UsageName(a.usage), a.usage_index, a.instruction,
                    a.format, VertexFormatName(a.format), a.stride, a.offset,
                    static_cast<uint32_t>(rf));
        if (rf == plume::RenderFormat::UNKNOWN) {
            REXLOG_WARN("      ^^ UNMAPPED GPUVERTEXFORMAT {}", a.format);
        }
    }
}

// sub_82231BF8 -- the leaf that actually rewrites the vfetch operands. See the
// header for the contract; the short version is that `ucode_ea` is a WRITABLE
// SCRATCH COPY of the microcode, and by the time __imp__ returns the patched
// instructions are sitting at ucode_ea + address*12.
//
// This is the only place vs_0f1a's layout ever exists. Its scratch copy is the
// PM4 command buffer that IncrementalShaderPatchAndLoad is about to hand to the
// GPU, so nothing durable is left behind to read afterwards: the shader object,
// its stored microcode and the second copy at *(shader+32)+*(record+0) all keep
// the compiler's placeholder operands forever.
void OnVfetchPatch(uint32_t shader_ea, uint32_t ucode_ea, uint32_t decl_ea,
                   uint32_t variant, GuestResolve resolve, void* ctx) {
    if (!Enabled() || ucode_ea == 0 || resolve == nullptr) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_vfetch_patches;

    auto oit = g_shader_objects.find(shader_ea);
    if (oit == g_shader_objects.end()) return;
    auto sit = g_shaders.find(oit->second);
    if (sit == g_shaders.end() || sit->second.entry == nullptr) return;
    Created& s = sit->second;
    if (!s.entry->is_vertex || s.entry->element_count == 0) return;
    // Deliberately NOT gated on layout_done: this runs on every re-patch, and a
    // re-patch is how a shader's SECOND use announces its different layout.
    // Measured ~300 calls a second against ~10k draws, so tracking all of them
    // costs a mutex the draw path was taking anyway.

    // Instructions are addressed from ucode_ea by instruction NUMBER, so the
    // read span is fixed by the highest element the container declares.
    uint32_t max_addr = 0;
    for (uint32_t e = 0; e < s.entry->element_count; ++e) {
        max_addr = std::max<uint32_t>(max_addr, uint32_t(s.entry->elements[e].address));
    }
    const uint32_t span = (max_addr + 1) * 12;
    const uint8_t* ucode = resolve(ucode_ea, span, ctx);
    if (ucode == nullptr) {
        if (!s.diagnosed_incremental) {
            s.diagnosed_incremental = true;
            REXLOG_WARN("plume: {} vfetch patch -- destination {:08X} is not {} bytes of "
                        "readable guest memory (variant {}, decl {:08X})",
                        s.entry->name, ucode_ea, span, variant, decl_ea);
        }
        return;
    }

    // Only the fetch-constant field (bits 20-29) of word 0 is rewritten; the
    // opcode, source and destination registers and the must-be-one bit are the
    // compiler's and survive. Checking them is what makes it safe to trust an
    // address computed from a reverse-engineered call contract: if this buffer
    // were not this shader's microcode, five 20-bit checks would not all pass.
    std::vector<FetchedAttribute> attrs;
    for (uint32_t e = 0; e < s.entry->element_count; ++e) {
        const auto& el = s.entry->elements[e];
        const uint32_t d = uint32_t(el.address) * 3;
        if ((d + 3) * 4 > s.code.size()) return;
        const uint32_t w0 = LoadBE32(ucode + d * 4);
        const uint32_t w1 = LoadBE32(ucode + (d + 1) * 4);
        const uint32_t w2 = LoadBE32(ucode + (d + 2) * 4);
        const uint32_t made_w0 = LoadBE32(&s.code[d * 4]);
        if ((w0 & 0x000FFFFFu) != (made_w0 & 0x000FFFFFu)) {
            if (!s.diagnosed_incremental) {
                s.diagnosed_incremental = true;
                REXLOG_WARN("plume: {} vfetch patch at {:08X}+{} is NOT this shader -- insn {} "
                            "reads {:08X}, the container says {:08X} (variant {}, decl {:08X})",
                            s.entry->name, ucode_ea, d * 4, el.address, w0, made_w0,
                            variant, decl_ea);
            }
            return;
        }
        FetchedAttribute a = DecodeVfetch(w0, w1, w2);
        a.usage = el.usage;
        a.usage_index = el.usage_index;
        a.instruction = el.address;
        if (a.format == 0 && a.stride == 0) {
            if (!s.logged_placeholder) {
                s.logged_placeholder = true;
                REXLOG_WARN("plume: {} vfetch patch at {:08X} left insn {} on the compiler's "
                            "placeholder (w1={:08X} w2={:08X}) -- variant {}, decl {:08X}",
                            s.entry->name, ucode_ea, el.address, w1, w2, variant, decl_ea);
            }
            return;
        }
        attrs.push_back(a);
    }
    if (attrs.empty() || attrs == s.attributes) return;   // unchanged: nothing to say

    s.attributes = std::move(attrs);
    s.layout_done = true;
    // Bounded: a shader with two uses re-patches thousands of times as the game
    // alternates between them, and logging every flip would bury the log.
    if (++s.layout_versions > kMaxLayoutVersionsLogged) return;
    REXLOG_WARN("plume: vertex layout {} for {} via the VFETCH PATCHER sub_82231BF8 "
                "(ucode {:08X}, variant {}, decl {:08X} -- decl=0 is the 83%-of-draws case)",
                s.layout_versions, s.entry->name, ucode_ea, variant, decl_ea);
    for (const auto& a : s.attributes) {
        const auto rf = ToRenderFormat(a.format, a.normalized, a.is_signed);
        REXLOG_WARN("   {}{}  insn {:<3} format {:<3} ({})  stride {:<4} offset {:<4} "
                    "fetch-const {} -> plume {}",
                    UsageName(a.usage), a.usage_index, a.instruction,
                    a.format, VertexFormatName(a.format), a.stride, a.offset,
                    a.fetch_const, static_cast<uint32_t>(rf));
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

// Defined below, next to the frame target they render into.
void DrawIntoFrame(Combo& c, const Created& s, uint32_t start_vertex, uint32_t vertex_count,
                   const uint8_t* base);
void EndFrame();
void FlushConstants(const uint8_t* base);

void SnapshotDrawVertices(const uint8_t* base) {
    const uint32_t start_vertex = g_draw_start;
    const uint32_t vertex_count = g_draw_count;
    if (!Enabled() || !g_init_ok || base == nullptr || vertex_count == 0) return;
    // Every address below is derived from an offset that is still being
    // reverse-engineered, so none of them may be dereferenced unguarded. This
    // used to be a bare `base + ea` and it segfaulted the moment the fetch
    // constant read garbage instead of zero.
    const auto resolve = [base](uint32_t ea, uint32_t size) -> const uint8_t* {
        return dt_guest_mem::Resolve(base, ea, size);
    };
    std::lock_guard<std::mutex> lock(g_mutex);

    // Find the combination this draw belongs to; it carries the layout and the
    // bindings. Only the current one is needed, and only a few times.
    auto sit = g_shaders.find(g_draw_vs);
    if (sit == g_shaders.end() || !sit->second.layout_done) {
        // THE drop site that matters: it is ahead of DrawIntoFrame, so a draw
        // whose shader never patched its vfetch never even reaches the pipeline
        // check below.
        ++g_draws_nopipe;
        return;
    }
    Created& s = sit->second;

    // Constants arrive ~9 times a frame and the pending list is 64 deep, so it
    // must be drained on EVERY draw. Draining only at the (once per
    // combination) offscreen draw overflowed it within the first few frames and
    // dropped every region, which looked exactly like "the constants are wrong".
    FlushConstants(base);

    // The real work: upload this draw's vertex data and record it into the
    // frame. Kept ahead of the logging below because it is now the point.
    if (g_draw_combo < g_combo_count) {
        Combo& c = g_combos[g_draw_combo];
        // Latch this combination's layout. THIS is the moment it is correct:
        // the draw's own state flush (inside __imp__, which has already run)
        // is what re-patched the vfetch, so the shader's current attributes
        // describe THIS draw and no other. Re-latched while the pipeline is
        // still pending -- a combination seen before its shader was ever
        // patched would otherwise be stuck with nothing forever -- and frozen
        // once built, since the pipeline's input layout cannot change.
        if (c.pipeline_state == PipelineState::kPending && !s.attributes.empty() &&
            (!c.layout_latched || c.attributes != s.attributes)) {
            c.attributes = s.attributes;
            c.layout_latched = true;
        }
        DrawIntoFrame(c, s, start_vertex, vertex_count, base);
    }

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
        const uint32_t fc_ea = g_device_ea + 1152u + a.fetch_const * 8u;
        const uint8_t* fc = resolve(fc_ea, 8u);   // a fetch constant is 2 dwords
        if (fc == nullptr) {
            REXLOG_WARN("   {}{} fetch const {} @{:08X} is not readable guest memory -- "
                        "device+1152+fc*8 is not where this title keeps them",
                        UsageName(a.usage), a.usage_index, a.fetch_const, fc_ea);
            continue;
        }
        const uint32_t base_addr = LoadBE32(fc) & ~uint32_t(3);
        if (base_addr == 0) {
            // Never silently skip: every early exit in a diagnostic must say why.
            REXLOG_WARN("   {}{} fetch const {} @{:08X} reads {:08X} {:08X} -- BaseAddress is 0, "
                        "so device+1152+fc*8 is not where this title keeps them",
                        UsageName(a.usage), a.usage_index, a.fetch_const, fc_ea,
                        LoadBE32(fc), LoadBE32(fc + 4));
            continue;
        }
        const uint32_t stride = (stream < kMaxStreams && g_streams[stream].stride)
                                    ? g_streams[stream].stride : a.stride;
        const uint32_t comp = ComponentBytes(a.format);

        // Byte-swap the first vertex's worth of this attribute and print it, so
        // the swap width can be checked against plausible values rather than
        // assumed correct.
        uint8_t v[16] = {};
        const uint32_t width = AttributeBytes(a.format);
        if (width == 0 || width > sizeof(v) || comp == 0) continue;

        // BaseAddress is PHYSICAL. Reading it as a virtual address is what used
        // to fault here -- and what made device+1152+fc*8 look like the wrong
        // offset when it was right all along.
        const uint32_t span = start_vertex * stride + a.offset + width;
        const uint32_t va = dt_guest_mem::PhysicalToVirtual(base_addr, span);
        if (va == 0) {
            REXLOG_WARN("   {}{} s{} physical BaseAddress {:08X} (+{} bytes) is not readable "
                        "through any physical alias -- fetch constant @{:08X}",
                        UsageName(a.usage), a.usage_index, stream, base_addr, span, fc_ea);
            continue;
        }
        const uint32_t ea = va + start_vertex * stride + a.offset;
        const uint8_t* p = resolve(ea, width);
        if (p == nullptr) continue;
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

// ── Shader constants ─────────────────────────────────────────────────────────
// The game sets every float constant through D3DDevice_GpuBeginShaderConstantF4,
// which returns a pointer for the game to write into. Nothing is shadowed in
// D3DDevice -- searching 128 KB around it for a known matrix found nothing, and
// SetPending_AluConstants' pointer argument moves per call and does not hold the
// values either. So the only way to read them is to remember where each block
// went and read it once the game has written it.
//
// Ground truth for cross-checking comes from REX_DUMP_ALUCONST in rexglue's D3D12
// command processor, which prints what the translated shader actually receives:
// vertex c72..c75 = {2,0,0,0 / 0,2,0,0 / 0,0,-1,0 / -1,-1,0,1}, a 2D ortho
// matrix. That is what these constants must reproduce.

constexpr uint32_t kConstRegisters = 256;   // float4s per stage, Xenos
constexpr uint32_t kMaxPendingConst = 64;

struct PendingConstants {
    bool pixel = false;
    uint32_t start = 0;
    uint32_t count = 0;
    uint32_t ea = 0;
};
PendingConstants g_pending_const[kMaxPendingConst];
uint32_t g_pending_const_count = 0;
uint32_t g_pending_const_dropped = 0;

float g_vs_constants[kConstRegisters][4] = {};
float g_ps_constants[kConstRegisters][4] = {};
bool g_constants_seen = false;

void OnGpuBeginShaderConstantF4(bool pixel_shader, uint32_t start_register, uint32_t count,
                                uint32_t dest_ea, const uint8_t* base) {
    if (!Enabled() || dest_ea == 0 || count == 0) return;
    // ⚠ The pointer aims into the COMMAND BUFFER RING, which the game keeps
    // advancing. Reading a block at the next DRAW was far too late: the bytes
    // there had already become later PM4 packets (type-3 headers, physical
    // addresses). Drain whatever is outstanding now instead -- one
    // constant-set interval old, not a whole frame.
    FlushConstants(base);
    if (g_pending_const_count >= kMaxPendingConst) {
        ++g_pending_const_dropped;   // reported, never silently lost
        return;
    }
    g_pending_const[g_pending_const_count++] = {pixel_shader, start_register, count, dest_ea};
    static bool logged_vs = false;
    if (!pixel_shader && start_register == 72 && !logged_vs) {
        logged_vs = true;
        REXLOG_WARN("plume: first vertex constant block -- start={} count={} dest={:08X}",
                    start_register, count, dest_ea);
    }
}

// Read everything written since the last draw into the shadow. Done at draw
// time on purpose: at the moment of the call the game has not written the
// values yet, so reading then yields whatever the buffer held before.
void FlushConstants(const uint8_t* base) {
    for (uint32_t i = 0; i < g_pending_const_count; ++i) {
        const PendingConstants& p = g_pending_const[i];
        if (p.start >= kConstRegisters) continue;
        const uint32_t count = std::min(p.count, kConstRegisters - p.start);
        const uint8_t* src = dt_guest_mem::Resolve(base, p.ea, count * 16);
        if (src == nullptr) continue;
        if (!p.pixel && p.start == 72) {
            static bool shown = false;
            if (!shown) {
                shown = true;
                REXLOG_WARN("plume: raw dwords at {:08X} (vertex c72, {} regs):", p.ea, count);
                for (uint32_t v = 0; v < count; ++v) {
                    REXLOG_WARN("   {:08X} {:08X} {:08X} {:08X}",
                                LoadBE32(src + v * 16 + 0), LoadBE32(src + v * 16 + 4),
                                LoadBE32(src + v * 16 + 8), LoadBE32(src + v * 16 + 12));
                }
            }
        }
        auto* dst = p.pixel ? g_ps_constants : g_vs_constants;
        for (uint32_t v = 0; v < count; ++v) {
            for (uint32_t c = 0; c < 4; ++c) {
                const uint32_t bits = LoadBE32(src + (v * 4 + c) * 4);
                std::memcpy(&dst[p.start + v][c], &bits, sizeof(float));
            }
        }
        g_constants_seen = true;
    }
    g_pending_const_count = 0;
}

// ── Textures ─────────────────────────────────────────────────────────────────
// The shaders sample bindlessly: `g_Texture2DDescriptorHeap[index]` with the
// index read out of SharedConstants (the pixel shader declares
// `SrcImage_Texture2DDescriptorIndex : packoffset(c0.x)` and its sampler index
// at c12.x). So a texture reaches a draw by being placed in the space0
// descriptor set and having its slot number written into b2.
//
// The source of truth for what to upload is the guest's Xenos texture fetch
// constant. The fetch constant file sits at device+1152 in 24-byte slots -- the
// same array the vertex fetch constants come from, since a slot is either one
// texture constant (6 dwords) or three vertex constants (2 dwords each). That
// is why vertex constant N is at +N*8 and sampler S is at +S*24.
struct DecodedTexture {
    uint32_t base_address = 0;   // physical
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t format = 0;
    uint32_t pitch = 0;
    bool tiled = false;
    // Endian (xenos.h): 0 none, 1 8in16, 2 8in32, 3 16in32. Read rather than
    // assumed -- the swap width is a property of the binding, and picking it by
    // convention is how a texture comes out plausible but wrong.
    uint32_t endianness = 0;
    bool valid = false;
};

// Decode a texture fetch constant at an arbitrary guest address. Split out of
// the sampler-slot reader because the RESOLVE path hands us a D3DBaseTexture
// OBJECT, whose fetch constant is embedded in the header rather than bound to
// a sampler slot.
DecodedTexture DecodeFetchConstantAt(uint32_t ea, const uint8_t* base) {
    DecodedTexture out;
    const uint8_t* p = dt_guest_mem::Resolve(base, ea, 24);
    if (p == nullptr) return out;

    uint32_t d[6];
    for (uint32_t i = 0; i < 6; ++i) d[i] = LoadBE32(p + i * 4);

    const uint32_t type = d[0] & 0x3;
    out.pitch = (d[0] >> 22) & 0x1FF;
    out.tiled = ((d[0] >> 31) & 0x1) != 0;
    out.format = d[1] & 0x3F;
    out.endianness = (d[1] >> 6) & 0x3;
    out.base_address = ((d[1] >> 12) & 0xFFFFF) << 12;
    out.width = ((d[2] >> 0) & 0x1FFF) + 1;
    out.height = ((d[2] >> 13) & 0x1FFF) + 1;
    out.valid = (type == 2) && out.base_address != 0 && out.width > 0 && out.height > 0;
    return out;
}

DecodedTexture ReadTextureFetchConstant(uint32_t sampler, const uint8_t* base) {
    DecodedTexture out;
    if (g_device_ea == 0 || sampler >= 8) return out;
    const uint32_t ea = g_device_ea + 1152u + sampler * 24u;
    const uint8_t* p = dt_guest_mem::Resolve(base, ea, 24);
    if (p == nullptr) return out;

    uint32_t d[6];
    for (uint32_t i = 0; i < 6; ++i) d[i] = LoadBE32(p + i * 4);

    // Layout per rex/graphics/xenos.h xe_gpu_texture_fetch_t. Decoded by hand
    // rather than by including that header so the app does not depend on a
    // plugin-internal type; the field positions are quoted alongside.
    const uint32_t type = d[0] & 0x3;                   // FetchConstantType
    out.pitch = (d[0] >> 22) & 0x1FF;                   // dword_0 +22, rows>>5
    out.tiled = ((d[0] >> 31) & 0x1) != 0;              // dword_0 +31
    out.format = d[1] & 0x3F;                           // dword_1 +0
    out.endianness = (d[1] >> 6) & 0x3;                 // dword_1 +6
    out.base_address = ((d[1] >> 12) & 0xFFFFF) << 12;  // dword_1 +12, >>12
    out.width = ((d[2] >> 0) & 0x1FFF) + 1;             // size_2d, stored -1
    out.height = ((d[2] >> 13) & 0x1FFF) + 1;
    // type 2 == kTexture; anything else is not a texture binding.
    out.valid = (type == 2) && out.base_address != 0 && out.width > 0 && out.height > 0;
    return out;
}

// Xenos 2D tiled address. Copied from rexglue's
// graphics/pipeline/texture/util.cpp GetTiledOffset2D -- it is declared in a
// public SDK header but DEFINED inside the rexgpu-xenos plugin, whose symbols
// are hidden, so the app cannot link it.
int32_t TiledOffset2D(int32_t x, int32_t y, uint32_t pitch, uint32_t bytes_per_block_log2) {
    pitch = (pitch + 31u) & ~31u;   // align to the 32-block tile
    int32_t macro = ((x >> 5) + (y >> 5) * int32_t(pitch >> 5)) << (bytes_per_block_log2 + 7);
    int32_t micro = ((x & 7) + ((y & 0xE) << 2)) << bytes_per_block_log2;
    int32_t offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 1) << 4);
    return ((offset & ~0x1FF) << 3) + ((y & 16) << 7) + ((offset & 0x1C0) << 2) +
           (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (offset & 0x3F);
}

// One texture ready to upload: rows packed tightly, in a format plume takes.
//
// Rows are rows of BLOCKS, not of texels, so the compressed and uncompressed
// paths share the staging and copy code below. For an uncompressed format the
// block is 1x1 and the two are the same thing.
struct TextureUpload {
    std::vector<uint8_t> data;
    plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
    uint32_t row_bytes = 0;   // bytes in one row of blocks
    uint32_t rows = 0;        // rows of blocks
    uint32_t block = 1;       // texels per block edge
};

// Where a texture's bytes actually live.
//
// PhysicalToVirtual returns the FIRST readable alias, which was right for
// vertex data but proves nothing for a surface: the 360 maps the same physical
// memory into several windows, and a mapped-but-stale one reads as zeros. Pick
// the alias that actually HAS content, and say which.
const uint8_t* ResolveTextureMemory(const DecodedTexture& t, const uint8_t* base,
                                    uint32_t needed) {
    static constexpr uint32_t kAliases[] = {0xA0000000u, 0xC0000000u, 0xE0000000u, 0x80000000u};
    const uint8_t* src = nullptr;
    for (uint32_t alias : kAliases) {
        const uint8_t* p = dt_guest_mem::Resolve(base, alias | t.base_address, needed);
        if (p == nullptr) continue;
        size_t nonzero = 0;
        for (uint32_t i = 0; i < needed && i < 0x40000u; ++i) nonzero += (p[i] != 0);
        static std::set<uint32_t> reported;
        if (reported.insert(alias | t.base_address).second) {
            REXLOG_WARN("plume: texture {:08X} via {:08X} alias -- {} non-zero bytes in the first "
                        "256 KB", t.base_address, alias, nonzero);
        }
        if (src == nullptr || nonzero != 0) {
            src = p;
            if (nonzero != 0) break;
        }
    }
    return src;
}

// Undo the GPU-side endian swap the fetch constant asks for. The width is a
// property of the BINDING, so it is read from the constant rather than assumed:
// a 16-bit swap applied to 32-bit data produces a texture that looks plausible
// and is wrong, which is the expensive kind of wrong.
void ApplyEndianSwap(uint8_t* p, size_t bytes, uint32_t endianness) {
    switch (endianness) {
        case 1:   // k8in16
            for (size_t i = 0; i + 2 <= bytes; i += 2) std::swap(p[i], p[i + 1]);
            break;
        case 2:   // k8in32
            for (size_t i = 0; i + 4 <= bytes; i += 4) {
                std::swap(p[i], p[i + 3]);
                std::swap(p[i + 1], p[i + 2]);
            }
            break;
        case 3:   // k16in32
            for (size_t i = 0; i + 4 <= bytes; i += 4) {
                std::swap(p[i], p[i + 2]);
                std::swap(p[i + 1], p[i + 3]);
            }
            break;
        default:
            break;
    }
}

// Read one texture out of guest memory into something uploadable.
//
// The DXT formats are passed through COMPRESSED rather than decoded: D3D12 and
// Vulkan both take BC1/2/3 natively, the block layout is identical to the
// guest's, and decoding would cost a decompressor plus 8x the memory to arrive
// at the same picture. Only the endian swap is undone.
//
// Anything else is reported and skipped rather than uploaded as garbage.
bool ReadTexture(const DecodedTexture& t, const uint8_t* base, TextureUpload& out) {
    constexpr uint32_t kFormat_8_8_8_8 = 6;
    constexpr uint32_t kFormat_DXT1 = 18, kFormat_DXT2_3 = 19, kFormat_DXT4_5 = 20;

    if (t.format == kFormat_DXT1 || t.format == kFormat_DXT2_3 || t.format == kFormat_DXT4_5) {
        const uint32_t block_bytes = (t.format == kFormat_DXT1) ? 8u : 16u;
        out.format = (t.format == kFormat_DXT1)   ? plume::RenderFormat::BC1_UNORM
                     : (t.format == kFormat_DXT2_3) ? plume::RenderFormat::BC2_UNORM
                                                    : plume::RenderFormat::BC3_UNORM;
        out.block = 4;

        // `pitch` is in texels (stored >>5) even for a block format, and it can
        // exceed the width -- linear textures are padded to 256 bytes a row.
        const uint32_t src_texels = std::max(t.pitch * 32u, t.width);
        const uint32_t src_blocks = (src_texels + 3u) / 4u;
        out.rows = (t.height + 3u) / 4u;
        const uint32_t dst_blocks = (t.width + 3u) / 4u;
        out.row_bytes = dst_blocks * block_bytes;

        // Tiling for a compressed format is the SAME swizzle, applied over
        // blocks instead of texels: the pitch counts blocks and the element
        // size is the block size. Getting that wrong produces a picture that is
        // recognisably the right texture, shuffled -- so it is worth stating
        // that this is the block interpretation and not a second algorithm.
        const uint32_t block_log2 = (block_bytes == 8u) ? 3u : 4u;
        const uint32_t needed =
            t.tiled ? (((src_blocks + 31u) & ~31u) * ((out.rows + 31u) & ~31u) * block_bytes)
                    : (src_blocks * block_bytes * out.rows);
        const uint8_t* src = ResolveTextureMemory(t, base, needed);
        if (src == nullptr) return false;

        out.data.resize(size_t(out.row_bytes) * out.rows);
        for (uint32_t by = 0; by < out.rows; ++by) {
            uint8_t* dst_row = out.data.data() + size_t(by) * out.row_bytes;
            if (!t.tiled) {
                std::memcpy(dst_row, src + size_t(by) * src_blocks * block_bytes, out.row_bytes);
                continue;
            }
            for (uint32_t bx = 0; bx < dst_blocks; ++bx) {
                const uint32_t off = uint32_t(
                    TiledOffset2D(int32_t(bx), int32_t(by), src_blocks, block_log2));
                if (off + block_bytes > needed) continue;
                std::memcpy(dst_row + size_t(bx) * block_bytes, src + off, block_bytes);
            }
        }
        ApplyEndianSwap(out.data.data(), out.data.size(), t.endianness);
        return true;
    }

    if (t.format != kFormat_8_8_8_8) return false;

    out.format = plume::RenderFormat::R8G8B8A8_UNORM;
    out.block = 1;
    out.row_bytes = t.width * 4u;
    out.rows = t.height;

    const uint32_t pitch_texels = t.pitch * 32u;
    const uint32_t src_pitch = std::max(pitch_texels, t.width);
    const uint32_t needed = t.tiled ? (((src_pitch + 31u) & ~31u) * ((t.height + 31u) & ~31u) * 4u)
                                    : (src_pitch * t.height * 4u);
    const uint8_t* src = ResolveTextureMemory(t, base, needed);
    if (src == nullptr) return false;

    out.data.resize(size_t(t.width) * t.height * 4);
    for (uint32_t y = 0; y < t.height; ++y) {
        for (uint32_t x = 0; x < t.width; ++x) {
            const uint32_t src_off =
                t.tiled ? uint32_t(TiledOffset2D(int32_t(x), int32_t(y), src_pitch, 2))
                        : ((y * src_pitch + x) * 4u);
            if (src_off + 4 > needed) continue;
            // Guest is big-endian ARGB; DXGI R8G8B8A8_UNORM wants R,G,B,A.
            const uint32_t px = LoadBE32(src + src_off);
            uint8_t* d = out.data.data() + (size_t(y) * t.width + x) * 4;
            d[0] = uint8_t((px >> 16) & 0xFF);   // R
            d[1] = uint8_t((px >> 8) & 0xFF);    // G
            d[2] = uint8_t((px >> 0) & 0xFF);    // B
            d[3] = uint8_t((px >> 24) & 0xFF);   // A
        }
    }
    return true;
}

// ── Frame composition and presentation ───────────────────────────────────────
// The previous stage rendered each draw into its own 256x256 target, cleared it
// first, and read it straight back. That proved a single draw works, but it can
// never produce a picture: every draw destroyed the one before it.
//
// This stage keeps ONE colour + depth target for the whole frame. The frame is
// cleared once, every draw the game makes is recorded into the same open
// command list, and the result is copied into a swapchain backbuffer at
// D3DDevice_Swap. That is the point at which "it drew" stops being a pixel
// count and becomes something a person can look at.
//
// The window is our own, deliberately. rexglue owns the main window and its
// message loop, and taking that over is a much larger change than this spike
// should make -- a second window costs nothing and keeps a normal run intact,
// so the two renderers can be compared side by side rather than swapped.
//
// The readback survives as a measurement, now per FRAME instead of per draw:
// a headless run (DT_PLUME_WINDOW=0) still has to be able to say what
// happened, and a window nobody is looking at proves nothing on its own.

constexpr uint32_t kFrameW = 1280;
constexpr uint32_t kFrameH = 720;
// A safety ceiling, not a sampling limit: the census peak is ~885 draws/frame,
// so this only ever trips if something loops.
constexpr uint32_t kMaxDrawsPerFrame = 8192;
// Which frames get the full readback + coverage report. The first few, and
// then periodically forever: the interesting content arrives long after boot
// (the menu's real textures only load once the attract sequence advances), and
// a measurement that stops at frame 8 goes blind exactly when it starts to
// matter.
constexpr uint32_t kReportedFrames = 8;
constexpr uint32_t kReportInterval = 300;

// The frame clear. Dark blue-grey: not black (a shader that outputs black is
// still drawing, and "count the non-black pixels" once reported NOTHING DREW
// for a draw that covered the whole target), and not magenta either -- that
// was right when one draw owned the target, but it makes a composed frame
// unreadable. The channels are exact 8-bit values so the readback below can
// test equality instead of guessing how a float rounded.
constexpr uint8_t kClearR8 = 16, kClearG8 = 16, kClearB8 = 24;
constexpr float kClearR = kClearR8 / 255.0f;
constexpr float kClearG = kClearG8 / 255.0f;
constexpr float kClearB = kClearB8 / 255.0f;

std::unique_ptr<plume::RenderCommandQueue> g_queue;
std::unique_ptr<plume::RenderCommandList> g_cmd;
// Texture uploads cannot share g_cmd: it is mid-recording for most of a frame,
// and begin() on an already-open list is invalid.
std::unique_ptr<plume::RenderCommandList> g_upload_cmd;
std::unique_ptr<plume::RenderCommandFence> g_fence;
// Uploads are submitted and waited on in the middle of a frame's recording, so
// they get their own fence rather than racing the frame's.
std::unique_ptr<plume::RenderCommandFence> g_upload_fence;
std::unique_ptr<plume::RenderTexture> g_color_target;
std::unique_ptr<plume::RenderTexture> g_depth_target;
std::unique_ptr<plume::RenderFramebuffer> g_framebuffer;
std::unique_ptr<plume::RenderBuffer> g_readback;
enum class OffscreenState { kNone, kReady, kFailed };
OffscreenState g_offscreen = OffscreenState::kNone;

// The presentation window and its swapchain. Both are optional: if either
// fails, the frame still composes and reads back, and only the picture is lost.
std::unique_ptr<plume::RenderSwapChain> g_swapchain;
#ifdef _WIN32
HWND g_window = nullptr;
#endif
bool g_window_tried = false;

// Frame state. `g_frame_buffers` holds every per-draw upload alive until the
// frame's fence signals -- a vertex buffer freed while the GPU is still reading
// it is a use-after-free that usually does not crash, which is worse.
bool g_frame_open = false;
// Set by OnResolve around its EndFrame call: forces the readback and captures
// the frame's pixels into g_resolve_pixels for the write-back to guest memory.
bool g_capture_resolve = false;
std::vector<uint8_t> g_resolve_pixels;
uint32_t g_frame_index = 0;
uint32_t g_frame_draws = 0;
uint32_t g_frame_skipped = 0;
std::vector<std::unique_ptr<plume::RenderBuffer>> g_frame_buffers;

std::unique_ptr<plume::RenderDescriptorSet> g_texture_set;
std::unique_ptr<plume::RenderDescriptorSet> g_sampler_set;
std::unique_ptr<plume::RenderSampler> g_sampler;
std::vector<std::unique_ptr<plume::RenderTexture>> g_textures;
std::unordered_map<uint32_t, uint32_t> g_texture_index;   // guest base address -> heap slot
// DT_PLUME_TEXCONTROL: the whole heap is the control checker, so guest textures
// must not be uploaded over it.
bool g_control_texture_only = false;

// Upload the texture bound to `sampler` and return its slot in the bindless
// heap, or UINT32_MAX. Cached by guest base address: the menu rebinds the same
// 1920x1080 surface every draw, and untiling it each time would be absurd.
uint32_t EnsureTexture(uint32_t sampler, const uint8_t* base) {
    if (g_control_texture_only) return 0;   // every slot is the checker
    const DecodedTexture t = ReadTextureFetchConstant(sampler, base);
    if (!t.valid) return UINT32_MAX;
    auto it = g_texture_index.find(t.base_address);
    if (it != g_texture_index.end()) return it->second;

    TextureUpload up;
    if (!ReadTexture(t, base, up)) {
        static std::set<uint32_t> reported;
        if (reported.insert(t.format).second) {
            REXLOG_WARN("plume: texture format {} ({}x{}, {}) not handled yet -- not uploaded",
                        t.format, t.width, t.height, t.tiled ? "tiled" : "linear");
        }
        return UINT32_MAX;
    }

    // Is the source actually black, or did the read produce zeros? Those need
    // completely different fixes, and the readback cannot tell them apart.
    // Counted over raw bytes so it means something for compressed data too,
    // where there are no texels to inspect without a decompressor.
    {
        size_t nonzero = 0;
        for (uint8_t byte : up.data) nonzero += (byte != 0);
        REXLOG_WARN("plume: texture {:08X} {}x{} read as {} rows x {} B: {} of {} bytes non-zero "
                    "(endian {})",
                    t.base_address, t.width, t.height, up.rows, up.row_bytes, nonzero,
                    up.data.size(), t.endianness);
    }

    auto texture = g_device->createTexture(
        plume::RenderTextureDesc::Texture2D(t.width, t.height, 1, up.format));
    if (texture == nullptr) return UINT32_MAX;

    // D3D12 wants each row of a buffer->texture copy 256-byte aligned, so the
    // staging copy is repacked to that pitch rather than the texture's own.
    const uint32_t padded = (up.row_bytes + 255u) & ~255u;
    auto staging = g_device->createBuffer(
        plume::RenderBufferDesc::UploadBuffer(uint64_t(padded) * up.rows));
    if (staging == nullptr) return UINT32_MAX;
    auto* dst = static_cast<uint8_t*>(staging->map());
    if (dst == nullptr) return UINT32_MAX;
    for (uint32_t y = 0; y < up.rows; ++y) {
        std::memcpy(dst + size_t(y) * padded, up.data.data() + size_t(y) * up.row_bytes,
                    up.row_bytes);
    }
    staging->unmap();

    // plume takes the footprint pitch in TEXELS and converts: it divides by the
    // block edge and multiplies by the format's block size. So the padded byte
    // pitch has to be expressed back in texels, which for a block format means
    // padded/blockBytes blocks, each `block` texels wide.
    const uint32_t block_bytes = (up.block == 1) ? 4u : (up.row_bytes / ((t.width + 3u) / 4u));
    const uint32_t footprint_texels = (padded / block_bytes) * up.block;

    g_upload_cmd->begin();
    const plume::RenderTextureBarrier to_dst(texture.get(), plume::RenderTextureLayout::COPY_DEST);
    g_upload_cmd->barriers(plume::RenderBarrierStage::COPY, nullptr, 0, &to_dst, 1);
    g_upload_cmd->copyTextureRegion(
        plume::RenderTextureCopyLocation::Subresource(texture.get()),
        plume::RenderTextureCopyLocation::PlacedFootprint(
            staging.get(), up.format, t.width, t.height, 1, footprint_texels));
    const plume::RenderTextureBarrier to_read(texture.get(),
                                              plume::RenderTextureLayout::SHADER_READ);
    g_upload_cmd->barriers(plume::RenderBarrierStage::GRAPHICS, nullptr, 0, &to_read, 1);
    g_upload_cmd->end();
    const plume::RenderCommandList* lists[] = {g_upload_cmd.get()};
    g_queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0, g_upload_fence.get());
    g_queue->waitForCommandFence(g_upload_fence.get());

    const uint32_t slot = uint32_t(g_textures.size());
    g_texture_set->setTexture(slot, texture.get(), plume::RenderTextureLayout::SHADER_READ);
    g_textures.push_back(std::move(texture));
    g_texture_index[t.base_address] = slot;
    REXLOG_WARN("plume: uploaded texture {}x{} guest format {} as plume format {} {} base {:08X} "
                "-> heap slot {}",
                t.width, t.height, t.format, static_cast<uint32_t>(up.format),
                t.tiled ? "tiled" : "linear", t.base_address, slot);
    return slot;
}

// Create the presentation window, on whichever thread first draws. Windows
// dispatches messages to the thread that created the window, so the pump in
// EndFrame has to live on that same thread -- which it does, because both run
// from the game's render thread.
//
// Opt out with DT_PLUME_WINDOW=0 for a headless run; the frame still composes
// and still reads back, so the measurements are unaffected.
bool EnsureWindow() {
#ifdef _WIN32
    if (g_window != nullptr) return true;
    if (g_window_tried) return false;
    g_window_tried = true;

    const char* opt = std::getenv("DT_PLUME_WINDOW");
    if (opt != nullptr && opt[0] == '0') {
        REXLOG_WARN("plume: DT_PLUME_WINDOW=0 -- composing headless, no window");
        return false;
    }

    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = [](HWND h, UINT msg, WPARAM w, LPARAM l) -> LRESULT {
        // Closing the plume window must not tear down a renderer the game is
        // still driving, so it only hides.
        if (msg == WM_CLOSE) { ShowWindow(h, SW_HIDE); return 0; }
        return DefWindowProcW(h, msg, w, l);
    };
    wc.hInstance = GetModuleHandleW(nullptr);
    // IDC_ARROW is MAKEINTRESOURCEA here (this TU is not built with UNICODE),
    // so the W-suffixed call needs the W-suffixed resource id.
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    wc.lpszClassName = L"dt_plume_window";
    RegisterClassExW(&wc);

    // Fixed size: the frame is copied 1:1 into the backbuffer, and the
    // swapchain takes its size from the client rect, so the two must agree.
    const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    RECT r = {0, 0, LONG(kFrameW), LONG(kFrameH)};
    AdjustWindowRect(&r, style, FALSE);
    g_window = CreateWindowExW(0, wc.lpszClassName, L"Daytona USA -- plume native renderer",
                               style, CW_USEDEFAULT, CW_USEDEFAULT,
                               r.right - r.left, r.bottom - r.top,
                               nullptr, nullptr, wc.hInstance, nullptr);
    if (g_window == nullptr) {
        REXLOG_WARN("plume: CreateWindowExW failed ({}), composing headless", GetLastError());
        return false;
    }
    // NOACTIVATE: stealing focus from the game window would break input.
    ShowWindow(g_window, SW_SHOWNOACTIVATE);
    return true;
#else
    return false;
#endif
}

bool EnsureOffscreen() {
    if (g_offscreen != OffscreenState::kNone) return g_offscreen == OffscreenState::kReady;
    g_offscreen = OffscreenState::kFailed;   // set to ready only on the way out

    g_queue = g_device->createCommandQueue(plume::RenderCommandListType::DIRECT);
    // Command lists come from the queue they will run on, not the device.
    g_cmd = g_queue != nullptr ? g_queue->createCommandList() : nullptr;
    g_upload_cmd = g_queue != nullptr ? g_queue->createCommandList() : nullptr;
    g_fence = g_device->createCommandFence();
    g_upload_fence = g_device->createCommandFence();
    if (g_queue == nullptr || g_cmd == nullptr || g_upload_cmd == nullptr ||
        g_fence == nullptr || g_upload_fence == nullptr) {
        REXLOG_WARN("plume: frame setup failed -- queue={} list={} upload={} fences={}/{}",
                    g_queue != nullptr, g_cmd != nullptr, g_upload_cmd != nullptr,
                    g_fence != nullptr, g_upload_fence != nullptr);
        return false;
    }

    g_color_target = g_device->createTexture(
        plume::RenderTextureDesc::ColorTarget(kFrameW, kFrameH, kRenderTargetFormat));
    // The pixel shaders write SV_Depth, so the pipelines were compiled against a
    // depth target; the framebuffer must supply one or the draw is invalid.
    g_depth_target = g_device->createTexture(
        plume::RenderTextureDesc::DepthTarget(kFrameW, kFrameH, kDepthTargetFormat));
    if (g_color_target == nullptr || g_depth_target == nullptr) {
        REXLOG_WARN("plume: offscreen targets failed -- color={} depth={}",
                    g_color_target != nullptr, g_depth_target != nullptr);
        return false;
    }

    const plume::RenderTexture* color = g_color_target.get();
    plume::RenderFramebufferDesc fb_desc(&color, 1, g_depth_target.get());
    g_framebuffer = g_device->createFramebuffer(fb_desc);
    if (g_framebuffer == nullptr) {
        REXLOG_WARN("plume: createFramebuffer failed");
        return false;
    }

    // 1280 * 4 = 5120 bytes/row, already a multiple of the 256 bytes D3D12
    // requires of a texture->buffer copy footprint. A width that is not would
    // need a padded pitch here and an unpadding read below.
    static_assert((kFrameW * 4) % 256 == 0, "frame width needs a padded readback pitch");
    g_readback = g_device->createBuffer(
        plume::RenderBufferDesc::ReadbackBuffer(uint64_t(kFrameW) * kFrameH * 4));
    if (g_readback == nullptr) {
        REXLOG_WARN("plume: readback buffer failed");
        return false;
    }

    // The bindless heaps the shaders declare. Textures land in set 0 (space0);
    // sets 1 and 2 (Texture3D/TextureCube) exist so the root signature matches
    // the shaders, but Daytona binds neither.
    plume::RenderDescriptorSetBuilder tex;
    tex.begin();
    tex.addTexture(0, kTextureDescriptorSize);
    tex.end(true, kTextureDescriptorSize);
    g_texture_set = tex.create(g_device.get());

    plume::RenderDescriptorSetBuilder smp;
    smp.begin();
    smp.addSampler(0, kSamplerDescriptorSize);
    smp.end(true, kSamplerDescriptorSize);
    g_sampler_set = smp.create(g_device.get());

    plume::RenderSamplerDesc sampler_desc;
    g_sampler = g_device->createSampler(sampler_desc);
    if (g_texture_set == nullptr || g_sampler_set == nullptr || g_sampler == nullptr) {
        REXLOG_WARN("plume: descriptor sets/sampler failed -- textures={} samplers={} sampler={}",
                    g_texture_set != nullptr, g_sampler_set != nullptr, g_sampler != nullptr);
        return false;
    }
    g_sampler_set->setSampler(0, g_sampler.get());

    // ── Positive control for the bindless path ───────────────────────────────
    // Heap slot 0 is a 2x2 magenta/green checker that no guest texture can
    // produce, uploaded before any of them so real textures start at slot 1.
    //
    // It answers a question the frame readback cannot. Every draw currently
    // comes out black, and there are two very different reasons that could be
    // true: the shaders are sampling the texture we point them at and it is
    // black, or the descriptor index never reaches them and they are all
    // sampling slot 0. If the frame turns into the checker, it is the second.
    {
        static const uint8_t kCheck[16] = {
            255, 0, 255, 255,   0, 255, 0, 255,
            0, 255, 0, 255,     255, 0, 255, 255,
        };
        auto control = g_device->createTexture(
            plume::RenderTextureDesc::Texture2D(2, 2, 1, plume::RenderFormat::R8G8B8A8_UNORM));
        auto staging = g_device->createBuffer(plume::RenderBufferDesc::UploadBuffer(256 * 2));
        void* p = (control != nullptr && staging != nullptr) ? staging->map() : nullptr;
        if (p != nullptr) {
            std::memcpy(static_cast<uint8_t*>(p) + 0, kCheck + 0, 8);
            std::memcpy(static_cast<uint8_t*>(p) + 256, kCheck + 8, 8);
            staging->unmap();

            g_upload_cmd->begin();
            const plume::RenderTextureBarrier to_dst(control.get(),
                                                     plume::RenderTextureLayout::COPY_DEST);
            g_upload_cmd->barriers(plume::RenderBarrierStage::COPY, nullptr, 0, &to_dst, 1);
            g_upload_cmd->copyTextureRegion(
                plume::RenderTextureCopyLocation::Subresource(control.get()),
                plume::RenderTextureCopyLocation::PlacedFootprint(
                    staging.get(), plume::RenderFormat::R8G8B8A8_UNORM, 2, 2, 1, 64));
            const plume::RenderTextureBarrier to_read(control.get(),
                                                      plume::RenderTextureLayout::SHADER_READ);
            g_upload_cmd->barriers(plume::RenderBarrierStage::GRAPHICS, nullptr, 0, &to_read, 1);
            g_upload_cmd->end();
            const plume::RenderCommandList* lists[] = {g_upload_cmd.get()};
            g_queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0, g_upload_fence.get());
            g_queue->waitForCommandFence(g_upload_fence.get());

            // DT_PLUME_TEXCONTROL=1 puts the checker in EVERY heap slot, so no
            // descriptor index can be wrong. Putting it only in slot 0 answered
            // a narrower question than it looked like it did: the frame stayed
            // black, which rules out "everything samples slot 0" but not "the
            // index points somewhere unbound". Filling the heap removes the
            // index from the experiment entirely -- if the frame is STILL
            // black, nothing about texture binding is the cause and the zero
            // is coming from after the sample.
            const char* all = std::getenv("DT_PLUME_TEXCONTROL");
            const uint32_t fill = (all != nullptr && all[0] == '1') ? kTextureDescriptorSize : 1;
            for (uint32_t i = 0; i < fill; ++i) {
                g_texture_set->setTexture(i, control.get(), plume::RenderTextureLayout::SHADER_READ);
            }
            g_control_texture_only = (fill > 1);
            g_textures.push_back(std::move(control));
            REXLOG_WARN("plume: control checker in {} heap slot(s){}", fill,
                        g_control_texture_only
                            ? " -- DT_PLUME_TEXCONTROL: guest textures are NOT uploaded, so a "
                              "black frame cannot be blamed on texture binding"
                            : " (slot 0)");
        }
    }

#ifdef _WIN32
    if (EnsureWindow()) {
        // maxFrameLatency must be at least 1: the swapchain is created with the
        // waitable-object flag and SetMaximumFrameLatency(0) is invalid.
        plume::RenderSwapChainDesc sc(g_window, kRenderTargetFormat, 2, false, 2);
        g_swapchain = g_queue->createSwapChain(sc);
        if (g_swapchain == nullptr || g_swapchain->isEmpty()) {
            REXLOG_WARN("plume: swapchain creation failed -- composing headless");
            g_swapchain.reset();
        } else if (g_swapchain->getWidth() != kFrameW || g_swapchain->getHeight() != kFrameH) {
            // The copy into the backbuffer is 1:1, so a mismatch would be a
            // silently wrong picture rather than an error. Refuse it instead.
            REXLOG_WARN("plume: swapchain is {}x{} but the frame is {}x{} -- not presenting",
                        g_swapchain->getWidth(), g_swapchain->getHeight(), kFrameW, kFrameH);
            g_swapchain.reset();
        } else {
            g_swapchain->setVsyncEnabled(false);
            REXLOG_WARN("plume: swapchain ready -- {}x{}, 2 buffers, vsync off",
                        g_swapchain->getWidth(), g_swapchain->getHeight());
        }
    }
#endif

    g_offscreen = OffscreenState::kReady;
    REXLOG_WARN("plume: frame target ready -- {}x{} format {} + depth, readback, {}",
                kFrameW, kFrameH, static_cast<uint32_t>(kRenderTargetFormat),
                g_swapchain != nullptr ? "presenting to a window" : "headless");
    return true;
}

// Copy one slot's vertex range out of guest memory, byte-swapping per COMPONENT
// for each attribute that lives in that slot.
//
// The swap width is per component, not per vertex and not a blanket dword swap:
// dword-swapping an 8_8_8_8 colour reverses its channels and no DXGI format
// undoes that. Bytes not covered by any attribute are copied through untouched
// rather than zeroed -- padding we have not identified is not the same thing as
// padding we know to be unused.
bool BuildSlotBytes(const std::vector<FetchedAttribute>& attributes,
                    uint32_t fetch_const, uint32_t stride,
                    uint32_t start_vertex, uint32_t vertex_count, const uint8_t* base,
                    std::vector<uint8_t>& out, uint32_t& out_src_ea) {
    out_src_ea = 0;
    if (stride == 0 || vertex_count == 0) return false;

    const uint32_t fc_ea = g_device_ea + 1152u + fetch_const * 8u;
    const uint8_t* fc = dt_guest_mem::Resolve(base, fc_ea, 8u);
    if (fc == nullptr) return false;
    const uint32_t phys = LoadBE32(fc) & ~uint32_t(3);
    if (phys == 0) return false;

    const uint32_t span = (start_vertex + vertex_count) * stride;
    const uint32_t va = dt_guest_mem::PhysicalToVirtual(phys, span);
    if (va == 0) return false;
    const uint32_t src_ea = va + start_vertex * stride;
    const uint32_t bytes = vertex_count * stride;
    const uint8_t* src = dt_guest_mem::Resolve(base, src_ea, bytes);
    if (src == nullptr) return false;

    out.assign(src, src + bytes);
    out_src_ea = src_ea;

    for (const auto& a : attributes) {
        if (a.fetch_const != fetch_const) continue;
        const uint32_t comp = ComponentBytes(a.format);
        const uint32_t width = AttributeBytes(a.format);
        if (comp <= 1 || width == 0) continue;   // single-byte components need no swap
        for (uint32_t v = 0; v < vertex_count; ++v) {
            const uint32_t at = v * stride + a.offset;
            if (at + width > bytes) break;       // attribute runs past the range
            for (uint32_t g = 0; g + comp <= width; g += comp) {
                std::reverse(out.begin() + at + g, out.begin() + at + g + comp);
            }
        }
    }
    return true;
}

// Render one captured draw offscreen and report how many pixels it changed.
// The first offscreen draw runs a long sequence of plume calls, any of which
// can fault on a wrong assumption. A crash is not an early return, so it cannot
// log a reason on the way out -- instead each step announces itself BEFORE it
// runs, and the last line in the log names the one that died.
bool g_offscreen_verbose = true;
void Step(const char* what) {
    if (!g_offscreen_verbose) return;
    REXLOG_WARN("plume: offscreen step -- {}", what);
    // Unbuffered too: a crash can eat the last buffered log line, which would
    // point at the step BEFORE the one that actually died. plume writes its own
    // errors to stderr as well, so they interleave in the right order.
    std::fprintf(stderr, "[plume step] %s\n", what);
    std::fflush(stderr);
}

// Open the frame's command list and clear it. Called lazily by the first draw
// of a frame rather than at the previous Swap, so a frame in which the game
// draws nothing costs nothing and does not present a stale picture.
bool BeginFrame() {
    if (g_frame_open) return true;
    if (!EnsureOffscreen()) return false;

    g_cmd->begin();
    const plume::RenderTextureBarrier to_write[] = {
        plume::RenderTextureBarrier(g_color_target.get(), plume::RenderTextureLayout::COLOR_WRITE),
        plume::RenderTextureBarrier(g_depth_target.get(), plume::RenderTextureLayout::DEPTH_WRITE),
    };
    g_cmd->barriers(plume::RenderBarrierStage::GRAPHICS, nullptr, 0, to_write, 2);
    g_cmd->setFramebuffer(g_framebuffer.get());
    g_cmd->clearColor(0, plume::RenderColor(kClearR, kClearG, kClearB, 1.0f));
    g_cmd->clearDepthStencil();
    g_frame_open = true;
    g_frame_draws = 0;
    g_frame_skipped = 0;
    return true;
}

// Where does the frame time go? Accumulated per bucket, reported every 300
// frames by EndFrame. Wall-clock between EndFrames is the frame time; the
// buckets name our share of it, and the remainder is the guest's.
double g_ms_endframe = 0.0;
double g_ms_scatter = 0.0;
double g_ms_record = 0.0;
uint32_t g_scatter_count = 0;
std::chrono::steady_clock::time_point g_last_endframe{};

struct ScopedMs {
    double& acc;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit ScopedMs(double& a) : acc(a) {}
    ~ScopedMs() {
        acc += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }
};

void DrawIntoFrame(Combo& c, const Created& s, uint32_t start_vertex, uint32_t vertex_count,
                   const uint8_t* base) {
    ScopedMs timer(g_ms_record);
    if (c.pipeline_state != PipelineState::kBuilt || c.pipeline == nullptr) {
        ++g_draws_nopipe;
        return;
    }
    if (g_index_state != IndexState::kReady) return;
    if (!BeginFrame()) return;
    if (g_frame_draws >= kMaxDrawsPerFrame) { ++g_frame_skipped; return; }

    // Topology decides the draw shape. QUADLIST expands 4 verts -> 6 indices
    // and TRIANGLEFAN n -> (n-2)*3, both through static index buffers, because
    // neither exists in D3D12/Vulkan. TRIANGLELIST and TRIANGLESTRIP need no
    // index expansion at all -- they draw non-indexed, with the strip's
    // pipeline built with STRIP topology.
    //
    // ⚠ The first version fanned EVERYTHING non-quad. A TRIANGLELIST expanded
    // with fan indices re-anchors every triangle at vertex 0 -- measured as a
    // giant black wedge across the title screen that the xenos referee does
    // not have.
    const bool is_quad = (c.topology == 0x0D);
    const bool is_fan = (c.topology == 5);                         // XENOS: 5 = fan
    const bool is_direct = (c.topology == 4 || c.topology == 6);   // list, strip
    if (!is_quad && !is_fan && !is_direct) {
        if (!c.warned_unreadable) {
            c.warned_unreadable = true;
            REXLOG_WARN("plume: topology {} ({}) not drawable yet -- draws skipped",
                        c.topology, TopologyName(c.topology));
        }
        ++g_frame_skipped;
        return;
    }
    uint32_t index_count = 0;
    const plume::RenderBuffer* ib = nullptr;
    if (is_quad) {
        if (vertex_count < 4) return;
        index_count = (vertex_count / 4) * 6;
        ib = g_quad_indices.get();
    } else if (is_fan) {
        if (vertex_count < 3) return;
        index_count = (vertex_count - 2) * 3;
        ib = g_fan_indices.get();
    } else {
        if (vertex_count < 3) return;
    }
    if (!is_direct && (ib == nullptr || index_count == 0)) return;

    // One vertex buffer per input slot, each with that slot's own stride.
    std::vector<std::unique_ptr<plume::RenderBuffer>> vbs;
    std::vector<plume::RenderVertexBufferView> views;
    uint32_t first_src_ea = 0;
    for (uint32_t sl = 0; sl < c.slots.size(); ++sl) {
        std::vector<uint8_t> bytes;
        uint32_t src_ea = 0;
        if (!BuildSlotBytes(c.attributes, c.slot_consts[sl], c.slots[sl].stride, start_vertex,
                            vertex_count, base, bytes, src_ea)) {
            // Once per combination: this runs on every draw now, and a stream
            // that is unreadable once is unreadable every frame.
            if (!c.warned_unreadable) {
                c.warned_unreadable = true;
                REXLOG_WARN("plume: draws skipped -- slot {} (fetch const {}, stride {}) "
                            "has no readable vertex data", sl, c.slot_consts[sl],
                            c.slots[sl].stride);
            }
            ++g_frame_skipped;
            return;
        }
        if (sl == 0) {
            first_src_ea = src_ea;
            // What geometry is this actually? A degenerate quad draws nothing
            // and looks identical to a broken pipeline from the readback alone.
            if (c.recorded_draws <= 2) {
                std::string pos;
                const uint32_t stride = c.slots[sl].stride;
                for (uint32_t v = 0; v < vertex_count && v < 8; ++v) {
                    float f[3] = {};
                    for (uint32_t k = 0; k < 3 && (v * stride + k * 4 + 4) <= bytes.size(); ++k) {
                        std::memcpy(&f[k], bytes.data() + v * stride + k * 4, sizeof(float));
                    }
                    pos += fmt::format("({:.3f},{:.3f},{:.3f}) ", f[0], f[1], f[2]);
                }
                REXLOG_WARN("plume: draw {} slot0 stride {} positions: {}",
                            c.recorded_draws, stride, pos);
            }
        }
        Step("createBuffer(vertex)");
        auto vb = g_device->createBuffer(
            plume::RenderBufferDesc::VertexBuffer(bytes.size(), plume::RenderHeapType::UPLOAD));
        if (vb == nullptr) return;
        void* p = vb->map();
        if (p == nullptr) return;
        std::memcpy(p, bytes.data(), bytes.size());
        vb->unmap();
        views.emplace_back(vb->at(0), uint32_t(bytes.size()));
        // The GPU reads this buffer when the frame is submitted, not now, so it
        // has to outlive this function -- ownership moves to the frame.
        g_frame_buffers.push_back(std::move(vb));
    }
    if (views.empty()) return;

    ++c.recorded_draws;

    // The constant buffers the translated shaders declare: b0 = vertex, b1 =
    // pixel, b2 = shared, all in space4. The shaders index them by REGISTER
    // (vs_bbb7 reads `float4 mvp[4] : packoffset(c72)`), so each buffer is the
    // whole 256-register file rather than a packed subset.
    if (!g_constants_seen) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            REXLOG_WARN("plume: no shader constants captured yet -- draws would transform by "
                        "whatever is in the buffer, so they are skipped");
        }
        ++g_frame_skipped;
        return;
    }
    if (g_offscreen_verbose) {
        // What is bound to each sampler? The shaders index a bindless heap, so
        // this is the list of textures the draw would need uploaded.
        for (uint32_t sampler = 0; sampler < 8; ++sampler) {
            const DecodedTexture t = ReadTextureFetchConstant(sampler, base);
            if (!t.valid) continue;
            REXLOG_WARN("plume: sampler {} -> texture {}x{} format {} {} base {:08X} pitch {}",
                        sampler, t.width, t.height, t.format,
                        t.tiled ? "TILED" : "linear", t.base_address, t.pitch * 32);
        }
        // Positive control against REX_DUMP_ALUCONST's ground truth for c72..c75:
        // {2,0,0,0 / 0,2,0,0 / 0,0,-1,0 / -1,-1,0,1}.
        for (uint32_t r = 72; r <= 75; ++r) {
            REXLOG_WARN("plume: captured vertex c{} = {:.4f} {:.4f} {:.4f} {:.4f}", r,
                        g_vs_constants[r][0], g_vs_constants[r][1],
                        g_vs_constants[r][2], g_vs_constants[r][3]);
        }
    }
    auto upload_constants = [&](const void* src, uint64_t bytes,
                                std::unique_ptr<plume::RenderBuffer>& out) -> bool {
        out = g_device->createBuffer(
            plume::RenderBufferDesc::UploadBuffer(bytes));
        if (out == nullptr) return false;
        void* p = out->map();
        if (p == nullptr) return false;
        std::memcpy(p, src, size_t(bytes));
        out->unmap();
        return true;
    };
    std::unique_ptr<plume::RenderBuffer> vs_cb, ps_cb, shared_cb;
    // SharedConstants holds the alpha threshold and sampler descriptor indices.
    // Nothing this stage draws samples a texture and the shaders are built with
    // alpha test off, so zeros are correct here -- but the buffer must exist,
    // because an unbound root CBV reads undefined data.
    // SharedConstants carries the bindless indices: the pixel shader declares
    // SrcImage_Texture2DDescriptorIndex at c0.x and its sampler index at c12.x.
    // Without these the shader samples heap slot 0 and the picture is whatever
    // happens to be there.
    // Layout confirmed against the emitter (XenosRecomp shader_recompiler.cpp,
    // the SharedConstants block) rather than inferred from one shader:
    //   sN_Texture2DDescriptorIndex     c(N/4)[N%4]        -> s0 = c0.x
    //   sN_Texture3DDescriptorIndex     c(4 + N/4)[N%4]
    //   sN_TextureCubeDescriptorIndex   c(8 + N/4)[N%4]
    //   sN_SamplerDescriptorIndex       c(12 + N/4)[N%4]   -> s0 = c12.x
    //   g_Booleans c16.x, g_SwappedTexcoords c16.y,
    //   g_HalfPixelOffset c16.zw, g_AlphaThreshold c17.x
    //
    // The tail may stay zero for now: these shaders are built with the
    // ALPHA_TEST-off variant, and g_HalfPixelOffset only shifts the picture.
    //
    // ⚠ Every sampler, not just 0. ps_e785a247caf76ef9 declares SrcImage1 at
    // c0.y/c12.y; filling only s0 left it reading heap slot 0.
    uint32_t shared[64][4] = {};
    for (uint32_t s_i = 0; s_i < 8; ++s_i) {
        const uint32_t slot = EnsureTexture(s_i, base);
        if (slot == UINT32_MAX) continue;
        shared[s_i / 4][s_i % 4] = slot;
        shared[12 + s_i / 4][s_i % 4] = 0;   // the one sampler serves them all
    }
    if (!upload_constants(g_vs_constants, sizeof(g_vs_constants), vs_cb) ||
        !upload_constants(g_ps_constants, sizeof(g_ps_constants), ps_cb) ||
        !upload_constants(shared, sizeof(shared), shared_cb)) {
        REXLOG_WARN("plume: could not upload the shader constant buffers");
        return;
    }

    const plume::RenderViewport viewport(0.0f, 0.0f, float(kFrameW), float(kFrameH));
    const plume::RenderRect scissor(0, 0, int32_t(kFrameW), int32_t(kFrameH));

    // From here it is pure recording into the frame's already-open list. No
    // clear: the frame was cleared once in BeginFrame, and clearing per draw is
    // exactly what stopped the previous stage from ever producing a picture.
    ++g_frame_draws;
    Step("setPipeline");
    g_cmd->setPipeline(c.pipeline.get());
    Step("setGraphicsPipelineLayout");
    g_cmd->setGraphicsPipelineLayout(g_pipeline_layout.get());
    // Root descriptor order matches how the layout was built: b0, b1, b2.
    Step("setGraphicsRootDescriptor x3");
    g_cmd->setGraphicsRootDescriptor(vs_cb->at(0), 0);
    g_cmd->setGraphicsRootDescriptor(ps_cb->at(0), 1);
    g_cmd->setGraphicsRootDescriptor(shared_cb->at(0), 2);
    Step("setGraphicsDescriptorSet x2");
    g_cmd->setGraphicsDescriptorSet(g_texture_set.get(), 0);   // space0 Texture2D
    g_cmd->setGraphicsDescriptorSet(g_sampler_set.get(), 3);   // space3 SamplerState
    Step("setViewports/setScissors");
    g_cmd->setViewports(&viewport, 1);
    g_cmd->setScissors(&scissor, 1);
    Step("setVertexBuffers");
    g_cmd->setVertexBuffers(0, views.data(), uint32_t(views.size()), c.slots.data());
    if (is_direct) {
        Step("drawInstanced");
        g_cmd->drawInstanced(vertex_count, 1, 0, 0);
    } else {
        const plume::RenderIndexBufferView ibv(
            ib->at(0), index_count * uint32_t(sizeof(uint16_t)), plume::RenderFormat::R16_UINT);
        Step("setIndexBuffer");
        g_cmd->setIndexBuffer(&ibv);
        Step("drawIndexedInstanced");
        g_cmd->drawIndexedInstanced(index_count, 1, 0, 0, 0);
    }

    // The constant buffers are read at submit time like the vertex data, so
    // they belong to the frame too.
    g_frame_buffers.push_back(std::move(vs_cb));
    g_frame_buffers.push_back(std::move(ps_cb));
    g_frame_buffers.push_back(std::move(shared_cb));

    // The sequence survived once; do not narrate it for every later draw.
    g_offscreen_verbose = false;

    if (c.recorded_draws == 1) {
        REXLOG_WARN("plume: first draw of {} + {} recorded -- {} verts -> {} indices, {} slots "
                    "(vertex data @{:08X})",
                    s.entry->name, TopologyName(c.topology), vertex_count, index_count,
                    views.size(), first_src_ea);
    }
}

#ifdef _WIN32
// Our window's messages are delivered to the thread that created it, which is
// this one. Without a pump it is marked unresponsive and never repaints.
void PumpWindow() {
    if (g_window == nullptr) return;
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}
#endif

// Close the frame: copy into the backbuffer, submit, present, and (for the
// first few frames) measure what actually landed.
//
// The measurement is not redundant with the window. A window shows a person
// what happened; the readback is what an unattended run, or a run on a machine
// with no display, can still report -- and it is the thing that caught a
// renderer reporting success while drawing nothing.
void EndFrame() {
    if (!g_frame_open) return;
    ScopedMs timer(g_ms_endframe);
    g_frame_open = false;
    ++g_frame_index;
    if ((g_frame_index % 300) == 0) {
        const auto now = std::chrono::steady_clock::now();
        const double wall = g_last_endframe.time_since_epoch().count() != 0
            ? std::chrono::duration<double, std::milli>(now - g_last_endframe).count() : 0.0;
        REXLOG_WARN("plume: timing over 300 frames -- wall {:.0f}ms ({:.1f}fps), endframe "
                    "{:.0f}ms, scatter {:.0f}ms ({} scatters), record {:.0f}ms, guest+rest "
                    "{:.0f}ms",
                    wall, wall > 0 ? 300000.0 / wall : 0.0, g_ms_endframe, g_ms_scatter,
                    g_scatter_count, g_ms_record,
                    wall - g_ms_endframe - g_ms_scatter - g_ms_record);
        g_last_endframe = now;
        g_ms_endframe = g_ms_scatter = g_ms_record = 0.0;
        g_scatter_count = 0;
    }

    const bool report = g_capture_resolve ||
        g_frame_index <= kReportedFrames || (g_frame_index % kReportInterval) == 0;

    uint32_t swap_index = 0;
    plume::RenderTexture* backbuffer = nullptr;
    if (g_swapchain != nullptr && !g_swapchain->needsResize() &&
        g_swapchain->acquireTexture(nullptr, &swap_index)) {
        backbuffer = g_swapchain->getTexture(swap_index);
    }

    // One barrier batch: the frame target stops being a render target, and the
    // backbuffer becomes a copy destination.
    std::vector<plume::RenderTextureBarrier> to_copy;
    to_copy.emplace_back(g_color_target.get(), plume::RenderTextureLayout::COPY_SOURCE);
    if (backbuffer != nullptr) {
        to_copy.emplace_back(backbuffer, plume::RenderTextureLayout::COPY_DEST);
    }
    g_cmd->barriers(plume::RenderBarrierStage::COPY, nullptr, 0, to_copy.data(),
                    uint32_t(to_copy.size()));

    if (backbuffer != nullptr) {
        // 1:1 -- the window was created at exactly the frame size and the
        // swapchain was rejected at startup if it did not agree.
        g_cmd->copyTexture(backbuffer, g_color_target.get());
        const plume::RenderTextureBarrier to_present(backbuffer,
                                                     plume::RenderTextureLayout::PRESENT);
        g_cmd->barriers(plume::RenderBarrierStage::GRAPHICS, nullptr, 0, &to_present, 1);
    }
    if (report) {
        g_cmd->copyTextureRegion(
            plume::RenderTextureCopyLocation::PlacedFootprint(g_readback.get(), kRenderTargetFormat,
                                                              kFrameW, kFrameH, 1, kFrameW),
            plume::RenderTextureCopyLocation::Subresource(g_color_target.get()));
    }
    g_cmd->end();

    const plume::RenderCommandList* lists[] = {g_cmd.get()};
    g_queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0, g_fence.get());
    // Waiting on every frame gives up all overlap, but it is what makes freeing
    // the frame's buffers below safe without a per-frame ring. Correct first.
    g_queue->waitForCommandFence(g_fence.get());

    if (backbuffer != nullptr) {
        g_swapchain->present(swap_index, nullptr, 0);
    }
#ifdef _WIN32
    PumpWindow();
#endif

    const size_t buffers = g_frame_buffers.size();
    g_frame_buffers.clear();

    if (!report) return;

    const uint8_t* pixels = static_cast<const uint8_t*>(g_readback->map());
    if (pixels == nullptr) {
        REXLOG_WARN("plume: frame readback map failed");
        return;
    }

    // The resolve path wants this frame's pixels CPU-side, to hand back to the
    // guest. Grabbed here because the readback is already mapped.
    if (g_capture_resolve) {
        g_resolve_pixels.assign(pixels, pixels + size_t(kFrameW) * kFrameH * 4);
    }

    // DT_PLUME_DUMP=<frame> writes that reported frame out as a BMP. A pixel
    // count says a picture exists; only a picture says it is the right one,
    // and this spike has twice been misled by a number that was true.
    // BMP because it needs no library: bottom-up rows of the BGRA we already
    // have, which is exactly what the 32-bit format stores.
    {
        static const char* want = std::getenv("DT_PLUME_DUMP");
        if (want != nullptr && g_frame_index == uint32_t(std::atoi(want))) {
            const char* path = "plume_frame.bmp";
            if (FILE* f = std::fopen(path, "wb")) {
                const uint32_t image = kFrameW * kFrameH * 4;
                const uint32_t offset = 14 + 40;
                uint8_t header[54] = {};
                header[0] = 'B'; header[1] = 'M';
                auto put32 = [&](uint32_t at, uint32_t v) {
                    header[at] = uint8_t(v); header[at + 1] = uint8_t(v >> 8);
                    header[at + 2] = uint8_t(v >> 16); header[at + 3] = uint8_t(v >> 24);
                };
                put32(2, offset + image);
                put32(10, offset);
                put32(14, 40);
                put32(18, kFrameW);
                put32(22, uint32_t(-int32_t(kFrameH)));   // negative = top-down
                header[26] = 1; header[28] = 32;          // 1 plane, 32 bpp
                put32(34, image);
                std::fwrite(header, 1, sizeof(header), f);
                std::fwrite(pixels, 1, image, f);
                std::fclose(f);
                REXLOG_WARN("plume: wrote frame {} to {}", g_frame_index, path);
            }
        }
    }
    // The pixel census and the FRAME log below are diagnostics: skip them on
    // frames that only reached this path because the resolve wanted the pixels
    // (g_capture_resolve is true on essentially EVERY frame now), or a 921600-
    // pixel scan plus a log line runs 60 times a second for nobody.
    const bool periodic = g_frame_index <= kReportedFrames ||
        (g_frame_index % kReportInterval) == 0;
    if (!periodic) {
        g_readback->unmap();
        return;
    }
    // "Differs from the clear", not "is not black": a shader that outputs black
    // is drawing, and counting non-black pixels once reported NOTHING DREW for
    // a draw that covered the entire target.
    //
    // The clear channels are exact 8-bit values on purpose (kClearR/G/B), so
    // this is an equality test and not a float-rounding guess. The stored order
    // is B, G, R, A -- kRenderTargetFormat is B8G8R8A8_UNORM.
    uint32_t touched = 0, coloured = 0, sample = 0;
    for (uint32_t i = 0; i < kFrameW * kFrameH; ++i) {
        const uint8_t b = pixels[i * 4 + 0], g = pixels[i * 4 + 1], r = pixels[i * 4 + 2];
        if (b == kClearB8 && g == kClearG8 && r == kClearR8) continue;
        ++touched;
        if ((b | g | r) != 0) {
            if (coloured == 0) {
                sample = (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
            }
            ++coloured;
        }
    }
    g_readback->unmap();

    REXLOG_WARN("plume: FRAME {} -- {} draws recorded ({} skipped, {} dropped for want of a "
                "pipeline so far), {} buffers; "
                "{} of {} pixels touched, {} with colour, first {:08X}{}",
                g_frame_index, g_frame_draws, g_frame_skipped, g_draws_nopipe, buffers,
                touched, kFrameW * kFrameH, coloured, sample,
                g_swapchain != nullptr ? ", presented" : "");

    // What is actually IN the constant buffers we bind. A register file that is
    // mostly zero is not a neutral starting state: the translated shaders
    // multiply by their constants, so an uncaptured register is a black pixel,
    // and it looks exactly like a broken sampler from the readback alone.
    {
        auto census = [](const float (*regs)[4], const char* what) {
            uint32_t nonzero = 0;
            std::string first;
            for (uint32_t r = 0; r < kConstRegisters; ++r) {
                if (regs[r][0] == 0.0f && regs[r][1] == 0.0f &&
                    regs[r][2] == 0.0f && regs[r][3] == 0.0f) {
                    continue;
                }
                ++nonzero;
                if (nonzero <= 6) {
                    first += fmt::format("c{}=({:.3f},{:.3f},{:.3f},{:.3f}) ", r,
                                         regs[r][0], regs[r][1], regs[r][2], regs[r][3]);
                }
            }
            REXLOG_WARN("plume:   {} constants -- {} of {} registers non-zero; {}", what, nonzero,
                        kConstRegisters, nonzero != 0 ? first : std::string("ALL ZERO"));
        };
        census(g_vs_constants, "vertex");
        census(g_ps_constants, "pixel");
        if (g_pending_const_dropped != 0) {
            REXLOG_WARN("plume:   {} constant regions were dropped -- the pending list overflowed",
                        g_pending_const_dropped);
        }
    }
}

// Xenos RB_BLENDCONTROL -> plume. Field layout is rexglue registers.h:779
// (color src +0/5, op +5/3, dst +8/5; alpha src +16/5, op +21/3, dst +24/5),
// factor and op encodings from xenos.h:700/719 -- the 360 D3DBLEND values ARE
// the hardware encodings, which is why SetBlendState can store the packed
// word untranslated.
plume::RenderBlend ToRenderBlend(uint32_t f) {
    using B = plume::RenderBlend;
    switch (f) {
        case 0:  return B::ZERO;
        case 1:  return B::ONE;
        case 4:  return B::SRC_COLOR;
        case 5:  return B::INV_SRC_COLOR;
        case 6:  return B::SRC_ALPHA;
        case 7:  return B::INV_SRC_ALPHA;
        case 8:  return B::DEST_COLOR;
        case 9:  return B::INV_DEST_COLOR;
        case 10: return B::DEST_ALPHA;
        case 11: return B::INV_DEST_ALPHA;
        // Constant colour and constant alpha both map to the blend factor;
        // plume has no separate constant-alpha, so the alpha channel of the
        // factor stands in. Daytona sets BlendFactor via sub_8221D620.
        case 12: return B::BLEND_FACTOR;
        case 13: return B::INV_BLEND_FACTOR;
        case 14: return B::BLEND_FACTOR;
        case 15: return B::INV_BLEND_FACTOR;
        case 16: return B::SRC_ALPHA_SAT;
        default: return B::ONE;
    }
}

plume::RenderBlendOperation ToRenderBlendOp(uint32_t op) {
    using O = plume::RenderBlendOperation;
    switch (op) {
        case 0:  return O::ADD;
        case 1:  return O::SUBTRACT;
        case 2:  return O::MIN;
        case 3:  return O::MAX;
        case 4:  return O::REV_SUBTRACT;
        default: return O::ADD;
    }
}

plume::RenderBlendDesc BlendDescFromControl(uint32_t bc, uint8_t write_mask) {
    const uint32_t csrc = bc & 0x1F, cop = (bc >> 5) & 0x7, cdst = (bc >> 8) & 0x1F;
    const uint32_t asrc = (bc >> 16) & 0x1F, aop = (bc >> 21) & 0x7, adst = (bc >> 24) & 0x1F;
    plume::RenderBlendDesc d;
    d.srcBlend = ToRenderBlend(csrc);
    d.blendOp = ToRenderBlendOp(cop);
    d.dstBlend = ToRenderBlend(cdst);
    d.srcBlendAlpha = ToRenderBlend(asrc);
    d.blendOpAlpha = ToRenderBlendOp(aop);
    d.dstBlendAlpha = ToRenderBlend(adst);
    // Xenos has no master blend-enable bit; ONE + ADD + ZERO IS "off". The
    // same convention rexglue's pipeline caches use.
    d.blendEnabled = !(csrc == 1 && cop == 0 && cdst == 0 && asrc == 1 && aop == 0 && adst == 0);
    d.renderTargetWriteMask = write_mask;
    return d;
}

void BuildPipelines(const char* reason) {
    if (!Enabled() || !g_init_ok) return;
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_pipeline_layout == nullptr) {
        // ⚠ An "allowInputLayout only" layout produces a root signature with
        // ZERO parameters, and D3D12 validates root-signature/shader binding
        // compatibility inside CreateGraphicsPipelineState. Every pipeline
        // failed E_INVALIDARG against it -- silently, because plume's
        // D3D12GraphicsPipeline constructor discarded the HRESULT (patched).
        //
        // The bindings are not a guess. XenosRecomp emits them:
        //   shader_recompiler.cpp:1237  cbuffer VertexShaderConstants : b0, space4
        //                               cbuffer PixelShaderConstants  : b1, space4
        //   shader_recompiler.cpp:1266  cbuffer SharedConstants       : b2, space4
        //   shader_common.h:50-53       Texture2D[]   t0, space0
        //                               Texture3D[]   t0, space1
        //                               TextureCube[] t0, space2
        //                               SamplerState[] s0, space3
        // Daytona's blobs come from that same translator, so this mirrors
        // UnleashedRecomp's layout (gpu/video.cpp:1888-1976), which is the
        // reference implementation for these shaders.
        //
        // The heaps are declared here so the shaders link; they are not
        // populated yet. Nothing this stage draws samples a texture.
        plume::RenderPipelineLayoutBuilder builder;
        builder.begin(false, true);   // allowInputLayout

        plume::RenderDescriptorSetBuilder set;
        set.begin();
        set.addTexture(0, kTextureDescriptorSize);
        set.end(true, kTextureDescriptorSize);
        builder.addDescriptorSet(set);   // space0 Texture2D
        builder.addDescriptorSet(set);   // space1 Texture3D
        builder.addDescriptorSet(set);   // space2 TextureCube

        plume::RenderDescriptorSetBuilder samplers;
        samplers.begin();
        samplers.addSampler(0, kSamplerDescriptorSize);
        samplers.end(true, kSamplerDescriptorSize);
        builder.addDescriptorSet(samplers);   // space3 SamplerState

        builder.addRootDescriptor(0, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
        builder.addRootDescriptor(1, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
        builder.addRootDescriptor(2, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
        builder.end();

        g_pipeline_layout = builder.create(g_device.get());
        if (g_pipeline_layout == nullptr) {
            REXLOG_WARN("plume: createPipelineLayout failed; no pipelines can be built");
            return;
        }
        REXLOG_WARN("plume: pipeline layout built -- 4 descriptor sets (t/space0-2, s/space3) "
                    "+ 3 root CBVs (b0/b1/b2 space4), matching XenosRecomp's binding model");
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
            !c.layout_latched || c.attributes.empty()) {
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
        for (const auto& a : c.attributes) {
            const auto fmt = ToRenderFormat(a.format, a.normalized, a.is_signed);
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
        c.slots = slots;
        c.slot_consts = slot_consts;

        plume::RenderGraphicsPipelineDesc desc;
        desc.pipelineLayout = g_pipeline_layout.get();
        desc.vertexShader = vs->second.shader.get();
        desc.pixelShader = ps->second.shader.get();
        desc.inputSlots = slots.data();
        desc.inputSlotsCount = uint32_t(slots.size());
        desc.inputElements = elements.data();
        desc.inputElementsCount = uint32_t(elements.size());
        desc.renderTargetFormat[0] = kRenderTargetFormat;
        desc.renderTargetBlend[0] = BlendDescFromControl(c.blend, c.write_mask);
        desc.renderTargetCount = 1;
        desc.depthTargetFormat = kDepthTargetFormat;
        // QUADLIST and TRIANGLEFAN both become an indexed TRIANGLE_LIST: neither
        // exists in D3D12/Vulkan, and both expand with a static index pattern
        // (0,1,2, 0,2,3 per quad; 0,i,i+1 for a fan). TRIANGLELIST draws as
        // itself, non-indexed; TRIANGLESTRIP keeps STRIP topology (it exists
        // natively). See RENDERER_INPUT_SPEC.md.
        desc.primitiveTopology = c.topology == 6   // XENOS: 6 = strip
            ? plume::RenderPrimitiveTopology::TRIANGLE_STRIP
            : plume::RenderPrimitiveTopology::TRIANGLE_LIST;

        c.pipeline = g_device->createGraphicsPipeline(desc);
        if (c.pipeline == nullptr) {
            c.pipeline_state = PipelineState::kFailed;
            ++failed;
            REXLOG_WARN("plume: pipeline {} FAILED ({} + {}, {})", i,
                        vs->second.entry->name, ps->second.entry->name, TopologyName(c.topology));
            // CreateGraphicsPipelineState returns a bare E_INVALIDARG, and the
            // D3D12 debug layer is an optional Windows feature that is not
            // installed here -- so there is no message to read. Instead, rebuild
            // the pipeline with one piece removed at a time: the first variant
            // that succeeds names the piece that is wrong. Runs once per failed
            // combination.
            if (!g_bisected) {
                g_bisected = true;
                auto try_variant = [&](const char* what, auto&& mutate) {
                    plume::RenderGraphicsPipelineDesc v = desc;
                    mutate(v);
                    auto p = g_device->createGraphicsPipeline(v);
                    REXLOG_WARN("plume:   bisect -- without {}: {}", what,
                                p != nullptr ? "BUILDS" : "still fails");
                    return p != nullptr;
                };
                try_variant("the input layout", [](plume::RenderGraphicsPipelineDesc& v) {
                    v.inputElements = nullptr; v.inputElementsCount = 0;
                    v.inputSlots = nullptr;    v.inputSlotsCount = 0;
                });
                try_variant("the pixel shader", [](plume::RenderGraphicsPipelineDesc& v) {
                    v.pixelShader = nullptr;
                });
                try_variant("the depth target", [](plume::RenderGraphicsPipelineDesc& v) {
                    v.depthTargetFormat = plume::RenderFormat::UNKNOWN;
                });
                try_variant("the colour target", [](plume::RenderGraphicsPipelineDesc& v) {
                    v.renderTargetCount = 0;
                    v.renderTargetFormat[0] = plume::RenderFormat::UNKNOWN;
                });
                // The pixel shader is the piece that fails. Is the root
                // signature missing something it needs, or does it conflict
                // with something it does not? An EMPTY layout answers that:
                // if the pipeline builds against nothing, the bindings we
                // declared are wrong rather than absent.
                plume::RenderPipelineLayoutDesc empty_desc;
                empty_desc.allowInputLayout = true;
                auto empty_layout = g_device->createPipelineLayout(empty_desc);
                if (empty_layout != nullptr) {
                    try_variant("our root signature (empty layout instead)",
                                [&](plume::RenderGraphicsPipelineDesc& v) {
                                    v.pipelineLayout = empty_layout.get();
                                });
                }
            }
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
            constexpr uint32_t kObjectProbeDwords = 32;
            const uint8_t* obj = resolve(s.object_ea, kObjectProbeDwords * 4, ctx);
            if (obj == nullptr) { ++pending; continue; }
            const size_t probe = s.code.size() < 32 ? s.code.size() : 32;
            for (uint32_t i = 0; i < kObjectProbeDwords && !s.located; ++i) {
                const uint32_t cand = LoadBE32(obj + i * 4);
                if (!PlausibleGuestPointer(cand)) continue;
                // Candidates come from scanning an object's raw dwords, so most
                // of them are NOT pointers at all -- this is the single most
                // likely place to dereference an arbitrary address.
                const uint8_t* p = resolve(cand, uint32_t(probe), ctx);
                if (p != nullptr && std::memcmp(p, s.code.data(), probe) == 0) {
                    s.code_ea = cand;
                    s.located = true;
                }
            }
            if (!s.located) { ++pending; continue; }
        }

        const uint8_t* code = resolve(s.code_ea, uint32_t(s.code.size()), ctx);
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
            a.normalized = ((w1 >> 13) & 1) == 0;   // num_format_all == 0
            a.is_signed = ((w1 >> 12) & 1) != 0;    // fomat_comp_all
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
                constexpr uint32_t kObjectDumpDwords = 48;
                const uint8_t* obj = resolve(s.object_ea, kObjectDumpDwords * 4, ctx);
                if (obj != nullptr) {
                    const size_t probe = s.code.size() < 32 ? s.code.size() : 32;
                    for (uint32_t i = 0; i < kObjectDumpDwords; ++i) {
                        const uint32_t cand = LoadBE32(obj + i * 4);
                        if (!PlausibleGuestPointer(cand)) continue;
                        const uint8_t* p = resolve(cand, uint32_t(probe), ctx);
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
            const auto rf = ToRenderFormat(a.format, a.normalized, a.is_signed);
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

    // The blend state at THIS draw, from the device's render-state shadow.
    // Reading the shadow word rather than hooking every setter means the
    // merged result is seen no matter which API wrote it (SetBlendState packs
    // the whole word; SetRenderState_SrcBlend and friends splice bit fields).
    // The host pointers are cached: the device object never moves, and this
    // path runs per draw.
    static uint32_t shadow_dev = 0;
    static const uint8_t* blend_ptr = nullptr;
    static const uint8_t* cwmask_ptr = nullptr;
    if (shadow_dev != device_ea) {
        shadow_dev = device_ea;
        blend_ptr = dt_guest_mem::Resolve(base, device_ea + 10552, 4);
        cwmask_ptr = dt_guest_mem::Resolve(base, device_ea + 10460, 4);
    }
    const uint32_t blend = blend_ptr != nullptr ? LoadBE32(blend_ptr) : 0x00010001u;
    const uint8_t write_mask = cwmask_ptr != nullptr ? uint8_t(LoadBE32(cwmask_ptr) & 0xF) : 0xFu;

    Combo* found = nullptr;
    for (uint32_t i = 0; i < g_combo_count; ++i) {
        Combo& c = g_combos[i];
        if (c.vs == vs && c.ps == ps && c.topology == primitive_type && c.blend == blend &&
            c.write_mask == write_mask) {
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
        found->blend = blend;
        found->write_mask = write_mask;
        found->decl_ea = g_cur_decl_ea;
        REXLOG_WARN("plume: new combo {} -- blend {:08X} write-mask {:X}", g_combo_count - 1,
                    blend, write_mask);
    }
    ++found->draws;
    found->vertices += vertex_count;
    // A combination is assumed to use one declaration; count it if that is ever
    // false rather than silently reporting whichever one happened to be first.
    if (g_cur_decl_ea != found->decl_ea) ++found->decl_mismatch;
    if (vertex_count < found->min_count) found->min_count = vertex_count;
    if (vertex_count > found->max_count) found->max_count = vertex_count;
    g_draw_combo = uint32_t(found - g_combos);

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

void OnResolve(uint32_t dest_texture_ea, const uint8_t* base) {
    // Every early-out here logs its first few hits: 362 resolves ran in the
    // first post-predication boot and produced zero log lines, which made
    // "handshake not wired" and "handshake wired but starved of pixels"
    // indistinguishable. Silence is the one thing this function must not do.
    static uint32_t entries = 0;
    const uint32_t entry = ++entries;
    if (!Enabled() || !g_init_ok || dest_texture_ea == 0 || base == nullptr) {
        if (entry <= 4) {
            REXLOG_WARN("plume: RESOLVE entry #{} early-out (enabled={} init={} dest={:08X})",
                        entry, Enabled(), g_init_ok, dest_texture_ea);
        }
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);

    // Submit whatever this frame has recorded, capturing the pixels. A resolve
    // with nothing recorded still writes the LAST captured frame: the guest is
    // asking for output, and stale output unblocks it where none never will.
    if (g_frame_open) {
        g_capture_resolve = true;
        EndFrame();
        g_capture_resolve = false;
    }
    if (g_resolve_pixels.empty()) {
        if (entry <= 4 || (entry % 512) == 0) {
            REXLOG_WARN("plume: RESOLVE entry #{} -- no pixels ever captured (frame_open={}, "
                        "dest={:08X}); the recorder has not drawn anything yet",
                        entry, g_frame_open, dest_texture_ea);
        }
        return;
    }

    // The destination's fetch constant is embedded in the D3DBaseTexture
    // header. XDK 20500's layout puts it after seven bookkeeping dwords
    // (Common..MipFlush) at +0x1C, but that is a header-derived belief, not a
    // measured one -- so probe a few nearby offsets and take the first that
    // decodes as a valid k_8_8_8_8 texture. A wrong pick fails validation and
    // logs; it cannot write anywhere, because the write range is guarded.
    DecodedTexture t;
    uint32_t used_off = 0;
    for (uint32_t off : {0x1Cu, 0x10u, 0x18u, 0x20u, 0x24u}) {
        DecodedTexture c = DecodeFetchConstantAt(dest_texture_ea + off, base);
        if (c.valid && c.format == 6 && c.width >= 64 && c.height >= 64 &&
            c.width <= 4096 && c.height <= 4096) {
            t = c;
            used_off = off;
            break;
        }
    }
    static uint32_t resolves = 0;
    ++resolves;
    if (!t.valid) {
        if (resolves <= 4) {
            REXLOG_WARN("plume: RESOLVE #{} -- no fetch constant found in texture {:08X}",
                        resolves, dest_texture_ea);
        }
        return;
    }

    const uint32_t pitch_texels = std::max(t.pitch * 32u, t.width);
    const uint32_t aligned_pitch = (pitch_texels + 31u) & ~31u;
    const uint32_t needed = t.tiled
        ? (aligned_pitch * ((t.height + 31u) & ~31u) * 4u)
        : (pitch_texels * t.height * 4u);
    // The fetch constant's base is a GPU address, and those carry bits above
    // the 512 MB physical range (measured: B0898000 for the 1080p resolve
    // target). Same mask as the INDIRECT_BUFFER pointer fix: the physical
    // address is the low 29 bits (rexglue round-trips via CpuToGpu).
    const uint32_t phys = t.base_address & 0x1FFFFFFFu;
    const uint32_t va = dt_guest_mem::PhysicalToVirtual(phys, needed);
    uint8_t* dst = va != 0
        ? const_cast<uint8_t*>(dt_guest_mem::Resolve(base, va, needed))
        : nullptr;
    if (dst == nullptr) {
        if (resolves <= 4) {
            REXLOG_WARN("plume: RESOLVE #{} -- dest {:08X} ({}x{}) not writable",
                        resolves, t.base_address, t.width, t.height);
        }
        return;
    }

    // One scatter per (destination, captured frame). The guest resolves ~120
    // times a second but the pixels only change when a frame is captured, and
    // a full 1080p tiled write is ~2M TiledOffset2D computations -- repeating
    // it for identical pixels was a large slice of the 18fps-instead-of-60.
    static std::unordered_map<uint32_t, uint32_t> written;   // dest phys -> frame
    auto& stamp = written[phys];
    if (stamp == g_frame_index + 1u) return;   // +1: 0 must not match a fresh entry
    stamp = g_frame_index + 1u;

    // Nearest-scale our frame into the destination, packing guest-order ARGB
    // big-endian -- the exact inverse of ReadTexture's k_8_8_8_8 unpack, which
    // round-tripped correctly (the logos' colours were verified by eye).
    ScopedMs scatter_timer(g_ms_scatter);
    ++g_scatter_count;
    std::vector<uint32_t> sx_map(t.width);
    for (uint32_t x = 0; x < t.width; ++x) sx_map[x] = x * kFrameW / t.width;
    for (uint32_t y = 0; y < t.height; ++y) {
        const uint32_t sy = y * kFrameH / t.height;
        const uint8_t* srow = g_resolve_pixels.data() + size_t(sy) * kFrameW * 4;
        for (uint32_t x = 0; x < t.width; ++x) {
            const uint8_t* sp = srow + sx_map[x] * 4;   // B8G8R8A8 as stored
            const uint32_t off = t.tiled
                ? uint32_t(TiledOffset2D(int32_t(x), int32_t(y), pitch_texels, 2))
                : ((y * pitch_texels + x) * 4u);
            if (off + 4 > needed) continue;
            dst[off + 0] = sp[3];   // A
            dst[off + 1] = sp[2];   // R
            dst[off + 2] = sp[1];   // G
            dst[off + 3] = sp[0];   // B
        }
    }
    if (resolves <= 8 || (resolves % 600) == 0) {
        REXLOG_WARN("plume: RESOLVE #{} -- frame -> guest {:08X} {}x{} {} (fetch at +{:#x})",
                    resolves, t.base_address, t.width, t.height,
                    t.tiled ? "tiled" : "linear", used_off);
    }
}

void OnFrameEnd() {
    if (!Enabled() || !g_init_ok) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    EndFrame();
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

    REXLOG_WARN("   draws={} unresolved={} decl changes={} combos={} incremental-patches={} "
                "vfetch-patches-seen={} dropped-for-want-of-a-pipeline={}",
                g_draws, g_draws_unresolved, g_decl_changes, g_combo_count, g_incremental_calls,
                g_vfetch_patches, g_draws_nopipe);
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
    // A frame left open holds a recording command list and the GPU may still be
    // reading last frame's uploads. Drain before anything is destroyed.
    if (g_queue != nullptr && g_fence != nullptr) {
        if (g_frame_open) {
            g_cmd->end();
            g_frame_open = false;
        }
        g_queue->waitForCommandFence(g_fence.get());
    }
    g_frame_buffers.clear();
    g_shaders.clear();
    for (uint32_t i = 0; i < g_combo_count; ++i) g_combos[i].pipeline.reset();
    g_pipeline_layout.reset();
    g_quad_indices.reset();
    g_fan_indices.reset();
    g_index_state = IndexState::kNone;
    g_textures.clear();
    g_texture_index.clear();
    g_texture_set.reset();
    g_sampler_set.reset();
    g_sampler.reset();
    g_readback.reset();
    g_framebuffer.reset();
    g_color_target.reset();
    g_depth_target.reset();
    // The swapchain must go before the queue that created it, and the window
    // before neither -- but after the swapchain that holds its HWND.
    g_swapchain.reset();
    g_cmd.reset();
    g_upload_cmd.reset();
    g_fence.reset();
    g_upload_fence.reset();
    g_queue.reset();
#ifdef _WIN32
    if (g_window != nullptr) {
        DestroyWindow(g_window);
        g_window = nullptr;
    }
#endif
    g_offscreen = OffscreenState::kNone;
    g_device.reset();
    g_interface.reset();
    g_init_ok = false;
}

}  // namespace dt_plume

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
#include <map>
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
// Declared here rather than beside the rest of the frame state below because
// Init() reads DT_PLUME_PERSIST, and Init() comes first in the file. See the
// frame-target block for what it means.
bool g_persist = true;
// Likewise: the texture cache's content check, declared early for Init().
bool g_texprobe = true;
// Likewise: the GPU-side close of the resolve loop. See the resolve mirror
// beside the frame target for what it does and why it is exact.
bool g_gpuloop = true;
// Likewise: whether D3DRESOLVE_CLEARRENDERTARGET / CLEARDEPTHSTENCIL in a
// resolve's Flags actually clear our frame target. MEASURED first: Daytona
// sets 0x0300 on three resolves in four. See OnResolveFlags.
bool g_resolveclear = true;
// Likewise: whether the per-draw uploads are suballocated out of one per-frame
// buffer or each get a buffer of their own. See the upload arena.
bool g_arena = true;
// Likewise: whether the swapchain is presented ONLY at the guest's Swap, or at
// every EndFrame the way it used to be. See EndFrame for what that cost.
bool g_swap_present = true;

// ── Sub-buckets inside `record` ──────────────────────────────────────────────
// The 300-frame timing line names three buckets; `record` is the biggest of
// them in a race and it is one number covering five different jobs. This splits
// it, and it is OFF by default (DT_PLUME_PROF=1) because it is two clock reads
// per bucket per draw at ~885 draws a frame -- affordable to measure with, not
// something to leave in the hot path when nobody is reading it.
bool g_prof = false;
double g_ms_p_vtx = 0.0;      // guest vertex copy + attribute permutation
double g_ms_p_vbuf = 0.0;     // vertex buffer createBuffer/map/memcpy
double g_ms_p_tex = 0.0;      // the 8-sampler EnsureTexture/EnsureSampler loop
double g_ms_p_probe = 0.0;    // ... of which, ProbeSource (nested in p_tex)
double g_ms_p_cb = 0.0;       // the three constant-buffer uploads
double g_ms_p_cmd = 0.0;      // the plume recording calls themselves
uint32_t g_probe_calls = 0;   // ProbeSource entries per 300 frames

// Reads no clock at all when the profile is off, so the buckets can be spelled
// into the hot path and cost nothing in a normal run.
struct ProfMs {
    double* acc = nullptr;
    std::chrono::steady_clock::time_point t0;
    explicit ProfMs(double& a) {
        if (!g_prof) return;
        acc = &a;
        t0 = std::chrono::steady_clock::now();
    }
    ~ProfMs() {
        if (acc == nullptr) return;
        *acc += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }
};

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

    // ⚠ The vfetch destination swizzle, bits 0-11 of word 1 -- FOUR 3-bit lanes,
    // lane i saying where register component i comes from (0-3 = xyzw, 4 = 0.0,
    // 5 = 1.0, 7 = untouched).
    //
    // Two of them matter, and the whole magenta wash lived in the gap between:
    //
    //   dst_swiz    what the RUNTIME patched, i.e. what the GPU would do.
    //   shader_swiz what the container held when XenosRecomp compiled it, i.e.
    //               what the generated HLSL hard-codes -- it emits
    //               `r0.<lanes> = iColor0.<swizzled>;` from the COMPILE-TIME
    //               word, and it can never learn about a later patch.
    //
    // When they agree the shader already does the reorder and the upload must
    // hand over memory order. When they disagree -- and for D3DCOLOR they do,
    // because that is exactly how the D3D9 runtime turns ARGB into RGBA -- the
    // difference has to be applied CPU-side or the channels land wrong.
    uint32_t dst_swiz = 0;
    uint32_t shader_swiz = 0;

    // "Is this the same layout the runtime patched last time?" -- asked once
    // per draw, so it must not be a string compare or a re-decode.
    bool operator==(const FetchedAttribute& o) const {
        return usage == o.usage && usage_index == o.usage_index &&
               instruction == o.instruction && format == o.format && stride == o.stride &&
               offset == o.offset && normalized == o.normalized && is_signed == o.is_signed &&
               fetch_const == o.fetch_const && dst_swiz == o.dst_swiz &&
               shader_swiz == o.shader_swiz;
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
    bool logged_samplers = false;
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
        // ⚠ 36 (32_FLOAT) was in AttributeBytes and in ToRenderFormat but not
        // here, so the one attribute that uses it -- DEPTH0, which only
        // vs_0f1a63ff9e54ebc8 declares, which is only the 3D scene -- fell
        // through to 0 and was skipped by every swap. It reached the shader
        // still big-endian, and a float whose bytes are reversed is not a
        // slightly wrong number: 145.0014f reads as 1.1e-12.
        case 57: case 37: case 38: case 36: return 4;
        case 6:                             return 1;
        default:                            return 0;   // unknown: refuse to swap
    }
}

uint32_t ComponentCount(uint32_t gpu_format) {
    switch (gpu_format) {
        case 36: return 1;   // 32_FLOAT
        case 37: return 2;   // 32_32_FLOAT
        case 57: return 3;   // 32_32_32_FLOAT
        case 6:  return 4;   // 8_8_8_8
        case 38: return 4;   // 32_32_32_32_FLOAT
        default: return 0;
    }
}

// One lane of a vfetch destination swizzle. 0-3 name a fetched component, 4 and
// 5 are the constants 0.0 and 1.0, 7 means "leave the register alone".
uint32_t SwizzleLane(uint32_t swiz, uint32_t lane) { return (swiz >> (lane * 3)) & 0x7; }

const char* SwizzleText(uint32_t swiz, char (&buf)[8]) {
    static const char kNames[8] = {'x', 'y', 'z', 'w', '0', '1', '?', '_'};
    for (uint32_t i = 0; i < 4; ++i) buf[i] = kNames[SwizzleLane(swiz, i)];
    buf[4] = '\0';
    return buf;
}

// Build the byte permutation that turns one attribute's GUEST bytes into the
// bytes the host input assembler must see. Returns the attribute width, or 0 if
// the format is not understood (in which case the caller must copy through).
//
// Two corrections compose here, and keeping them in one table is what makes the
// order impossible to get wrong:
//
//  1. ENDIAN. The vertex fetch constant asks for 8in32, so the GPU reverses
//     every 4-byte GROUP before it looks at components. For an all-32-bit
//     attribute that is the same thing as reversing each component, which is
//     why the old per-component swap looked right for years of float data. For
//     8_8_8_8 it is NOT: the four bytes are one dword, the reverse applies to
//     all four of them, and component 0 is then the LOW byte of the result
//     (rexglue dxbc_translator_fetch.cpp:323 -- offsets 0/8/16/24). So a
//     D3DCOLOR stored big-endian as AA RR GG BB fetches as (B, G, R, A).
//
//  2. SWIZZLE DELTA. The GPU then applies the PATCHED dst_swiz; the generated
//     shader applies the COMPILE-TIME one. Only the difference is ours to fix:
//     lane i of the register must end up holding fetched component
//     dst_swiz[i], and the shader will read it out of input component
//     shader_swiz[i] -- so upload component shader_swiz[i] must carry fetched
//     component dst_swiz[i]. With D3DCOLOR's patched zyxw against the
//     container's xyzw that turns (B, G, R, A) back into (R, G, B, A).
//
// When nothing is patched the two swizzles agree, the delta is the identity,
// and all that survives is the endian reverse -- i.e. the previous behaviour
// for every attribute except the packed one.
uint32_t BuildAttributePermutation(const FetchedAttribute& a, uint8_t perm[16]) {
    const uint32_t width = AttributeBytes(a.format);
    const uint32_t comp_bytes = ComponentBytes(a.format);
    const uint32_t comps = ComponentCount(a.format);
    if (width == 0 || width > 16 || comp_bytes == 0 || comps == 0) return 0;

    // Which fetched component each UPLOAD component must carry. Identity for
    // anything the two swizzles do not both name -- an upload component the
    // shader never reads is a don't-care, and guessing at it would be noise.
    uint8_t map[4] = {0, 1, 2, 3};
    for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t src = SwizzleLane(a.dst_swiz, lane);       // what the GPU fetches
        const uint32_t dst = SwizzleLane(a.shader_swiz, lane);    // where the shader reads
        if (src > 3 || dst > 3) continue;                          // 0.0/1.0/untouched
        if (src >= comps || dst >= comps) continue;                // beyond this format
        map[dst] = uint8_t(src);
    }

    for (uint32_t p = 0; p < comps; ++p) {
        for (uint32_t t = 0; t < comp_bytes; ++t) {
            const uint32_t k = map[p] * comp_bytes + t;   // byte k of the FETCHED value
            perm[p * comp_bytes + t] = uint8_t((k & ~3u) + (3u - (k & 3u)));   // 8in32
        }
    }
    return width;
}

bool PermutationIsIdentity(const uint8_t perm[16], uint32_t width) {
    for (uint32_t i = 0; i < width; ++i) {
        if (perm[i] != i) return false;
    }
    return true;
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
    a.dst_swiz = w1 & 0xFFF;
    a.shader_swiz = a.dst_swiz;   // overwritten by the caller that has the container word
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

    // The console's backbuffer persists across a present, so ours does too, and
    // only the guest's clear clears it. DT_PLUME_PERSIST=0 goes back to
    // clearing at the top of every frame, which is the A/B for this.
    if (const char* p = std::getenv("DT_PLUME_PERSIST")) g_persist = std::atoi(p) != 0;
    if (const char* p = std::getenv("DT_PLUME_TEXPROBE")) g_texprobe = std::atoi(p) != 0;
    if (const char* p = std::getenv("DT_PLUME_GPULOOP")) g_gpuloop = std::atoi(p) != 0;
    if (const char* p = std::getenv("DT_PLUME_RESOLVECLEAR")) g_resolveclear = std::atoi(p) != 0;
    if (const char* p = std::getenv("DT_PLUME_PROF")) g_prof = std::atoi(p) != 0;
    if (const char* p = std::getenv("DT_PLUME_ARENA")) g_arena = std::atoi(p) != 0;
    if (const char* p = std::getenv("DT_PLUME_SWAPPRESENT")) g_swap_present = std::atoi(p) != 0;
    REXLOG_WARN("plume: the swapchain is presented {} (DT_PLUME_SWAPPRESENT={})",
                g_swap_present ? "at the guest's Swap ONLY -- a mid-frame resolve closes and "
                                 "captures the frame but does not put it on the screen"
                               : "at every EndFrame, mid-frame resolves included",
                g_swap_present ? 1 : 0);
    REXLOG_WARN("plume: per-draw uploads come from {} (DT_PLUME_ARENA={})",
                g_arena ? "ONE per-frame arena, bump-allocated"
                        : "a buffer of their own, one createBuffer each",
                g_arena ? 1 : 0);
    REXLOG_WARN("plume: frame target {} across Swaps (DT_PLUME_PERSIST={})",
                g_persist ? "PERSISTS -- only the guest clear clears it" : "is cleared every frame",
                g_persist ? 1 : 0);
    REXLOG_WARN("plume: the resolve loop closes {} (DT_PLUME_GPULOOP={})",
                g_gpuloop ? "ON THE GPU -- a bind of a resolve destination samples the mirror"
                          : "through guest memory -- every bind re-uploads 1080p from the CPU",
                g_gpuloop ? 1 : 0);

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
        a.dst_swiz = w1 & 0xFFF;
        // The compiled swizzle is what the generated HLSL froze in; it comes
        // from the container, never from the patched copy.
        a.shader_swiz = (uint32_t(el.address) * 3 + 2) * 4 <= s.code.size()
                            ? LoadBE32(&s.code[(uint32_t(el.address) * 3 + 1) * 4]) & 0xFFF
                            : a.dst_swiz;
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
        a.shader_swiz = LoadBE32(&s.code[(d + 1) * 4]) & 0xFFF;
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
        char patched[8], compiled[8];
        uint8_t perm[16] = {};
        const uint32_t width = BuildAttributePermutation(a, perm);
        REXLOG_WARN("   {}{}  insn {:<3} format {:<3} ({})  stride {:<4} offset {:<4} "
                    "fetch-const {} -> plume {} , swizzle patched {} vs compiled {}{}",
                    UsageName(a.usage), a.usage_index, a.instruction,
                    a.format, VertexFormatName(a.format), a.stride, a.offset,
                    a.fetch_const, static_cast<uint32_t>(rf),
                    SwizzleText(a.dst_swiz, patched), SwizzleText(a.shader_swiz, compiled),
                    a.dst_swiz == a.shader_swiz ? "" : "  <-- DIFFERS, compensated CPU-side");
        if (width != 0) {
            REXLOG_WARN("      upload byte order {} {} {} {}{}", perm[0], perm[1],
                        perm[2], perm[3], width > 4 ? " ..." : "");
        }
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
void EndFrame(bool present);
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

        // Reorder the first vertex's worth of this attribute with the SAME
        // permutation the upload uses and print both ends of it, so the guest
        // bytes and the value the shader will see can be read off against each
        // other instead of assumed to agree.
        uint8_t v[16] = {};
        uint8_t perm[16] = {};
        const uint32_t width = BuildAttributePermutation(a, perm);
        if (width == 0 || width > sizeof(v)) continue;

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
        for (uint32_t b = 0; b < width; ++b) v[b] = p[perm[b]];
        if (a.format == 57 || a.format == 37 || a.format == 36 || a.format == 38) {
            float f[4] = {};
            std::memcpy(f, v, width);
            REXLOG_WARN("   {}{} s{} @{:08X} stride {} -> {:.4f} {:.4f} {:.4f} {:.4f}",
                        UsageName(a.usage), a.usage_index, stream, ea, stride,
                        f[0], width > 4 ? f[1] : 0.0f, width > 8 ? f[2] : 0.0f,
                        width > 12 ? f[3] : 0.0f);
        } else {
            REXLOG_WARN("   {}{} s{} @{:08X} stride {} -> guest {:02X} {:02X} {:02X} {:02X} "
                        "= shader RGBA {:02X} {:02X} {:02X} {:02X}",
                        UsageName(a.usage), a.usage_index, stream, ea, stride,
                        p[0], p[1], p[2], p[3], v[0], v[1], v[2], v[3]);
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
// Bumped whenever the two files above change. A draw uploads them only when
// this has moved since the last upload of the frame -- see the upload arena.
uint32_t g_const_gen = 0;

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
        // The constant files moved, so the copy the arena is handing to draws is
        // no longer the current one. See g_const_gen.
        ++g_const_gen;
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
// XE_GPU_MAKE_TEXTURE_SWIZZLE(X,Y,Z,W) -- the identity.
constexpr uint32_t XE_GPU_TEXTURE_SWIZZLE_XYZW = 0u | (1u << 3) | (2u << 6) | (3u << 9);

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
    // dword_3 +1, 12 bits, four 3-bit lanes x,y,z,w (XE_GPU_TEXTURE_SWIZZLE).
    // Per lane: 0-3 name a component of the DECODED format, 4 = constant 0,
    // 5 = constant 1. This is the destination selection -- it decides which of
    // a k_8_8 texture's two memory channels reaches shader .x, and there is no
    // convention to fall back on, so it is read rather than assumed. Default is
    // the identity so an undecoded constant behaves as XYZW.
    uint32_t swizzle = XE_GPU_TEXTURE_SWIZZLE_XYZW;
    // Sampler state also lives in the fetch constant, and for a colour-INDEX
    // texture the filter is not a quality knob: interpolating between two
    // palette INDICES produces an index that means nothing, and the colour that
    // comes back is unrelated to either neighbour. So it is decoded and honoured
    // per binding rather than left at plume's LINEAR default.
    uint32_t mag_filter = 1;   // dword_3 +19, TextureFilter: 0 point, 1 linear
    uint32_t min_filter = 1;   // dword_3 +21
    uint32_t clamp_x = 0;      // dword_0 +10, ClampMode
    uint32_t clamp_y = 0;      // dword_0 +13
    // TEX_FORMAT_COMP / GPUSIGN, dword_0 +2/+4/+6/+8, one TextureSign per DATA
    // component (before the swizzle). 0 unsigned, 1 signed, 2 unsigned-biased,
    // 3 GAMMA -- and kGamma means the hardware LINEARISES the byte on fetch. A
    // reader that skips that hands the shader gamma-encoded numbers as if they
    // were linear, which is a contrast distortion and nothing else, so it is
    // decoded and reported rather than assumed to be zero.
    uint32_t sign[4] = {0, 0, 0, 0};
    // dword_3 +0, TextureNumFormat: 0 fraction (byte/255), 1 integer (byte).
    uint32_t num_format = 0;
    bool valid = false;
};

// The texture fetch swizzle uses the SAME 3-bit lane encoding as the vfetch
// destination swizzle above, so SwizzleLane/SwizzleText serve both. This is
// just the std::string wrapper the log calls want.
std::string SwizzleName(uint32_t swizzle) {
    char buf[8];
    return std::string(SwizzleText(swizzle, buf));
}

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
    out.swizzle = (d[3] >> 1) & 0xFFF;                  // dword_3 +1, after num_format
    out.mag_filter = (d[3] >> 19) & 0x3;
    out.min_filter = (d[3] >> 21) & 0x3;
    out.clamp_x = (d[0] >> 10) & 0x7;
    out.clamp_y = (d[0] >> 13) & 0x7;
    for (uint32_t i = 0; i < 4; ++i) out.sign[i] = (d[0] >> (2 + i * 2)) & 0x3;
    out.num_format = d[3] & 0x1;
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
    out.swizzle = (d[3] >> 1) & 0xFFF;                  // dword_3 +1, num_format is +0
    out.mag_filter = (d[3] >> 19) & 0x3;                // dword_3 +19
    out.min_filter = (d[3] >> 21) & 0x3;                // dword_3 +21
    out.clamp_x = (d[0] >> 10) & 0x7;                   // dword_0 +10
    out.clamp_y = (d[0] >> 13) & 0x7;                   // dword_0 +13
    for (uint32_t i = 0; i < 4; ++i)                    // dword_0 +2/+4/+6/+8
        out.sign[i] = (d[0] >> (2 + i * 2)) & 0x3;      // TextureSign, 3 == kGamma
    out.num_format = d[3] & 0x1;                        // dword_3 +0
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
    // Where the bytes came from, so a later bind can ask whether they changed
    // without re-deriving the per-format size. `alias` is the 360 memory window
    // ResolveTextureMemory picked; the same address reads differently through
    // each of them, so a probe has to use the one the upload used.
    uint32_t source_bytes = 0;
    uint32_t source_alias = 0;
};

// Where a texture's bytes actually live.
//
// PhysicalToVirtual returns the FIRST readable alias, which was right for
// vertex data but proves nothing for a surface: the 360 maps the same physical
// memory into several windows, and a mapped-but-stale one reads as zeros. Pick
// the alias that actually HAS content, and say which.
const uint8_t* ResolveTextureMemory(const DecodedTexture& t, const uint8_t* base,
                                    uint32_t needed, uint32_t* out_alias = nullptr) {
    static constexpr uint32_t kAliases[] = {0xA0000000u, 0xC0000000u, 0xE0000000u, 0x80000000u};
    const uint8_t* src = nullptr;
    bool all_zero = false;
    for (uint32_t alias : kAliases) {
        const uint8_t* p = dt_guest_mem::Resolve(base, alias | t.base_address, needed);
        if (p == nullptr) continue;
        size_t nonzero = 0;
        for (uint32_t i = 0; i < needed && i < 0x40000u; ++i) nonzero += (p[i] != 0);
        // The per-alias line this used to print was one of three the upload path
        // emitted per texture, and at ~200 textures that is 600 lines of a log
        // whose signal lives elsewhere. The alias it CHOSE is now folded into
        // the single upload line; only the case where nothing has content is
        // still worth its own line, because that is the failure this function
        // exists to catch.
        if (src == nullptr || nonzero != 0) {
            src = p;
            if (out_alias != nullptr) *out_alias = alias;
            if (nonzero != 0) break;
            all_zero = true;
        }
    }
    if (src != nullptr && all_zero) {
        static std::set<uint32_t> reported;
        if (reported.insert(t.base_address).second) {
            REXLOG_WARN("plume: texture {:08X} reads as ALL ZERO through every alias -- either the "
                        "art is not there yet or the address is wrong", t.base_address);
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
    constexpr uint32_t kFormat_8_8 = 10;
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
        out.source_bytes = needed;
        const uint8_t* src = ResolveTextureMemory(t, base, needed, &out.source_alias);
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

    if (t.format == kFormat_8_8) {
        // k_8_8 is Daytona's whole 3D scene: two 8-bit channels, sampled as a
        // COLOUR INDEX -- the pixel shader takes .x as a coordinate into
        // texture_color_ram (sampler 1) and looks the real colour up there. So
        // getting the channel assignment wrong does not tint the picture, it
        // indexes the wrong palette entry, and the road comes out any colour at
        // all. Which of the two memory bytes reaches .x is the fetch
        // constant's SWIZZLE, decoded above.
        //
        // Expanded to RGBA8 on the CPU rather than uploaded as R8G8 with a view
        // swizzle. Two reasons: the guest lanes can be the constants 0 and 1,
        // which a component mapping cannot always express, and the descriptor
        // heap here is filled with setTexture(slot, texture) -- one default view
        // per texture -- so a per-binding view swizzle has nowhere to live.
        // 128x128 is 16K texels; the expansion costs nothing.
        out.format = plume::RenderFormat::R8G8B8A8_UNORM;
        out.block = 1;
        out.row_bytes = t.width * 4u;
        out.rows = t.height;

        const uint32_t pitch_texels = t.pitch * 32u;
        const uint32_t src_pitch = std::max(pitch_texels, t.width);
        // 2 bytes a texel, so the tiled address is the log2(2) = 1 variant --
        // the same untiler the 4-byte path calls with 2 and the DXT path calls
        // with 3 or 4.
        constexpr uint32_t kTexelBytes = 2u;
        constexpr uint32_t kTexelLog2 = 1u;
        const uint32_t needed =
            t.tiled ? (((src_pitch + 31u) & ~31u) * ((t.height + 31u) & ~31u) * kTexelBytes)
                    : (src_pitch * t.height * kTexelBytes);
        out.source_bytes = needed;
        const uint8_t* src = ResolveTextureMemory(t, base, needed, &out.source_alias);
        if (src == nullptr) return false;

        out.data.resize(size_t(t.width) * t.height * 4);
        for (uint32_t y = 0; y < t.height; ++y) {
            for (uint32_t x = 0; x < t.width; ++x) {
                const uint32_t src_off =
                    t.tiled ? uint32_t(TiledOffset2D(int32_t(x), int32_t(y), src_pitch, kTexelLog2))
                            : ((y * src_pitch + x) * kTexelBytes);
                if (src_off + kTexelBytes > needed) continue;
                const uint8_t b0 = src[src_off], b1 = src[src_off + 1];
                // Component X is the FIRST byte, Y the second, and the fetch
                // constant's endianness (kNone for every k_8_8 Daytona binds)
                // swaps that pair.
                //
                // ⚠ This was settled by experiment, not by argument, because
                // the argument gave the wrong answer. Reading the k_8_8_8_8
                // path below as "the raw value is the big-endian word, X in the
                // low bits" is self-consistent -- LoadBE32 of 0xAARRGGBB with
                // the decoded zyxw swizzle does land on R,G,B,A -- and it
                // predicts X = the SECOND byte here. Built that way, the scene
                // bleaches: every surface goes near-white, which is what a
                // palette lookup does when it is fed the mostly-opaque ALPHA
                // channel and lands on the top entry every time. The A/B is
                // banked as k88_fix2/fix4/fix5.png against k88_a/v3.png.
                //
                // So k_8_8 stores X then Y in address order. Component X is the
                // luminance the palette is indexed by; Y is alpha, and the
                // xxxy swizzle puts it exactly where the shader expects.
                const bool swapped = (t.endianness == 1);   // k8in16
                const uint8_t cx = swapped ? b1 : b0;
                const uint8_t cy = swapped ? b0 : b1;
                // Two-component formats replicate the LAST component into the
                // ones they do not have, before the swizzle (xenos.h, k_8_A) --
                // which is why rexglue gives k_8_8 the host swizzle RGGG.
                const uint8_t comp[4] = {cx, cy, cy, cy};
                uint8_t* d = out.data.data() + (size_t(y) * t.width + x) * 4;
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    const uint32_t sel = SwizzleLane(t.swizzle, lane);
                    d[lane] = (sel < 4)    ? comp[sel]
                              : (sel == 4) ? uint8_t(0)     // constant 0
                              : (sel == 5) ? uint8_t(255)   // constant 1
                                           : comp[lane & 3];
                }
            }
        }
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
    out.source_bytes = needed;
    const uint8_t* src = ResolveTextureMemory(t, base, needed, &out.source_alias);
    if (src == nullptr) return false;

    out.data.resize(size_t(t.width) * t.height * 4);
    for (uint32_t y = 0; y < t.height; ++y) {
        for (uint32_t x = 0; x < t.width; ++x) {
            const uint32_t src_off =
                t.tiled ? uint32_t(TiledOffset2D(int32_t(x), int32_t(y), src_pitch, 2))
                        : ((y * src_pitch + x) * 4u);
            if (src_off + 4 > needed) continue;
            // Data component N is the Nth BYTE, and the fetch constant's
            // swizzle says which one reaches each shader lane -- the same rule
            // the k_8_8 path above was forced to by experiment, applied here
            // too. It was not, and that was the palette defect:
            //
            // The old code read the four bytes as a big-endian ARGB word and
            // hard-coded R,G,B,A out of it, which is component x = the LOW byte
            // -- the theory k_8_8 had already disproved. For the zyxw swizzle
            // every k_8_8_8_8 texture in the game carries, it lands one channel
            // out: the shader's BLUE gets the guest's ALPHA. The palette at
            // 148C2000 is 64x1024 of shading ramps stored opaque, so its blue
            // came back 0xFF in all 65536 texels, the pixel shader adds half
            // the palette to half the index, and every palette-lookup surface
            // got a >= 0.5 blue floor: grey road to pale lavender, green grass
            // to dark blue-green. The 640x1028 device-notice art decodes to a
            // blue/yellow negative the old way and to a white Xbox 360
            // controller with correct button colours this way; the palette
            // decodes to blue stripes the old way and to varied ramps this way.
            uint8_t* d = out.data.data() + (size_t(y) * t.width + x) * 4;
            const uint8_t* px = src + src_off;
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const uint32_t sel = SwizzleLane(t.swizzle, lane);
                d[lane] = (sel < 4)    ? px[sel]
                          : (sel == 4) ? uint8_t(0)     // constant 0
                          : (sel == 5) ? uint8_t(255)   // constant 1
                                       : px[lane];      // untouched
            }
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

// ── The frame target persists; only the GUEST clears it ──────────────────────
// A console backbuffer is not destroyed by presenting it. The guest exploits
// that: on a frame where only the overlay changed it re-submits the overlay and
// nothing else, and while paused it submits nothing but the pause panel. A
// renderer that clears at the top of every frame throws away the picture those
// frames are drawn ON TOP OF, and the result is a HUD floating on the clear
// colour -- measured at 3-4 of 8 in-race captures, and a pause menu over black
// where the referee freezes the race behind it.
//
// It is worse than one lost clear per Swap, too: EndFrame also runs at every
// D3DDevice_Resolve, and the guest resolves several times per frame. Each of
// those splits one guest frame into several of ours, so a scene pass and the
// HUD pass that follows it landed in different targets even within one frame.
//
// So the target is never cleared on our own initiative (once primed), and the
// guest's D3DDevice_ClearF is what clears it. DT_PLUME_PERSIST=0 restores the
// old clear-every-frame behaviour for an A/B (g_persist, declared at the top of
// this namespace because Init() reads its environment variable).
bool g_target_primed = false;
uint32_t g_clear_pending = 0;                 // D3DCLEAR_* not yet executed
float g_clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
float g_clear_z = 1.0f;
uint32_t g_clear_stencil = 0;
uint32_t g_clears_seen = 0;

// Set by OnResolve around its EndFrame call: forces the readback and captures
// the frame's pixels into g_resolve_pixels for the write-back to guest memory.
bool g_capture_resolve = false;
std::vector<uint8_t> g_resolve_pixels;
uint32_t g_frame_index = 0;
uint32_t g_frame_draws = 0;
uint32_t g_frame_skipped = 0;
std::vector<std::unique_ptr<plume::RenderBuffer>> g_frame_buffers;

// ── The per-frame upload arena ───────────────────────────────────────────────
// Every draw used to create four UPLOAD buffers of its own: one per vertex slot
// and three for the constant registers (b0 vertex, b1 pixel, b2 shared). At the
// ~900 draws a race frame records that is ~3,500 createBuffer calls a frame,
// each one a committed D3D12 resource with a heap and a GPU virtual address
// behind it.
//
// MEASURED, with DT_PLUME_PROF=1, in the 300-frame window right after the green
// flag (51.2fps, `record` 2,689ms of a 5,862ms wall):
//
//     cb 1761ms | vbuf 504ms | tex 186ms (probe 125ms) | vtx 111ms | cmd 79ms
//
// So 84% of `record` was creating and mapping buffers, and only 4% was the
// sampled texture probe that this work started out suspecting. The BYTES were
// never the problem -- 9 KB per draw is nothing -- the ALLOCATIONS were.
//
// So allocate once and hand out offsets. Chunks are UPLOAD buffers mapped for
// their whole life, bump-allocated within a frame and rewound at the top of the
// next one. Rewinding is safe for exactly the reason clearing g_frame_buffers
// is: EndFrame waits on the frame's fence before it returns, so no GPU work can
// still be reading last frame's bytes.
//
// DT_PLUME_ARENA=0 goes back to a buffer per allocation, which is the A/B.
constexpr uint64_t kArenaChunkBytes = 4ull << 20;
struct ArenaChunk {
    std::unique_ptr<plume::RenderBuffer> buffer;
    uint8_t* mapped = nullptr;
    uint64_t size = 0;
};
std::vector<ArenaChunk> g_arena_chunks;
size_t g_arena_chunk = 0;      // the chunk being filled
uint64_t g_arena_used = 0;     // how far into it
uint64_t g_arena_peak = 0;     // high-water mark over a frame, for the log

// Somewhere the GPU can read this frame: which buffer, at what offset, and
// where to write the bytes on the CPU.
struct UploadSpan {
    plume::RenderBuffer* buffer = nullptr;
    uint64_t offset = 0;
    uint8_t* cpu = nullptr;
    explicit operator bool() const { return cpu != nullptr; }
};

UploadSpan ArenaTake(uint64_t bytes, uint64_t align) {
    UploadSpan out;
    if (bytes == 0) return out;
    for (;;) {
        if (g_arena_chunk < g_arena_chunks.size()) {
            ArenaChunk& c = g_arena_chunks[g_arena_chunk];
            const uint64_t at = (g_arena_used + align - 1) & ~(align - 1);
            if (at + bytes <= c.size) {
                g_arena_used = at + bytes;
                g_arena_peak = std::max(g_arena_peak, at + bytes);
                out.buffer = c.buffer.get();
                out.offset = at;
                out.cpu = c.mapped + at;
                return out;
            }
            // Full. The remainder is abandoned for this frame; the next rewind
            // reclaims it.
            ++g_arena_chunk;
            g_arena_used = 0;
            continue;
        }
        ArenaChunk fresh;
        const uint64_t size = std::max<uint64_t>(kArenaChunkBytes, bytes + align);
        // VERTEX and CONSTANT because one chunk serves both, and Vulkan turns
        // these flags into usage bits a buffer cannot be bound without.
        fresh.buffer = g_device->createBuffer(plume::RenderBufferDesc::UploadBuffer(
            size, plume::RenderBufferFlag::VERTEX | plume::RenderBufferFlag::CONSTANT));
        if (fresh.buffer == nullptr) return out;
        // Mapped once and never unmapped: an UPLOAD heap is host-visible for its
        // whole life on both backends, and unmapping per draw was part of what
        // this replaces.
        fresh.mapped = static_cast<uint8_t*>(fresh.buffer->map());
        if (fresh.mapped == nullptr) return out;
        fresh.size = size;
        g_arena_chunks.push_back(std::move(fresh));
        REXLOG_WARN("plume: upload arena grew to {} chunks of {} KB", g_arena_chunks.size(),
                    size / 1024);
    }
}

// Per-frame scratch the GPU will read: out of the arena when it is on, out of a
// buffer of its own (owned by the frame, freed at its fence) when it is not.
UploadSpan TakeUpload(uint64_t bytes, uint64_t align, plume::RenderBufferFlags flags) {
    if (g_arena) return ArenaTake(bytes, align);
    UploadSpan out;
    auto buf = g_device->createBuffer(plume::RenderBufferDesc::UploadBuffer(bytes, flags));
    if (buf == nullptr) return out;
    out.cpu = static_cast<uint8_t*>(buf->map());
    if (out.cpu == nullptr) return out;
    out.buffer = buf.get();
    g_frame_buffers.push_back(std::move(buf));
    return out;
}

// The vertex and pixel constant files are 4 KB each and they change ~9 times a
// frame, not ~900: the guest sets them and then draws several times with the
// same values. So they are uploaded once per (frame, generation) and every draw
// in between points its root descriptor at the same bytes. g_const_gen is
// bumped by FlushConstants whenever it actually writes (declared beside the
// constant files themselves, since that is where it is written).
UploadSpan g_cb_vs, g_cb_ps;
uint32_t g_cb_gen = UINT32_MAX;

// Called at the top of each frame, once the previous frame's fence has been
// waited on. Also drops the constant-file spans, because they point into arena
// bytes that are about to be handed out again.
void ArenaRewind() {
    g_arena_chunk = 0;
    g_arena_used = 0;
    g_arena_peak = 0;
    g_cb_vs = UploadSpan();
    g_cb_ps = UploadSpan();
    g_cb_gen = UINT32_MAX;
}

std::unique_ptr<plume::RenderDescriptorSet> g_texture_set;
std::unique_ptr<plume::RenderDescriptorSet> g_sampler_set;
std::unique_ptr<plume::RenderSampler> g_sampler;
std::vector<std::unique_ptr<plume::RenderTexture>> g_textures;
// Textures replaced by a re-upload at a different size. The old one may still
// be referenced by draws already recorded into the open command list, so it
// cannot be freed until that frame's fence signals -- same rule, and the same
// release point, as g_frame_buffers.
std::vector<std::unique_ptr<plume::RenderTexture>> g_texture_graveyard;

// What the cache remembers about an uploaded texture. It used to remember only
// the heap slot, and that was the bug behind two defects at once: a hit never
// looked at the guest memory again, so a texture was uploaded ONCE, for all
// time. The game reuses addresses -- streamed UI art lands on a freed
// allocation, and the resolve destination the scene composites from is rewritten
// every single frame -- so the second and every later bind was served whatever
// happened to be in memory the first time. Measured: 110C5000 (1920x1080, the
// scene's resolve destination) was uploaded 49ms BEFORE the first resolve ever
// wrote it, and never again, which is why the compositing pass painted black
// over the 3D scene and left the HUD floating on it.
struct TexCache {
    uint32_t slot = 0;
    uint32_t width = 0, height = 0;
    plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
    uint32_t source_bytes = 0;
    uint32_t source_alias = 0;
    uint64_t probe = 0;        // sampled hash of the source at upload time
    // g_frame_index + 1 of the last frame the probe ran for this entry, so that
    // 0 does not match frame 0. See ProbeDue.
    uint32_t probe_frame = 0;
    uint32_t resolve_stamp = 0;   // g_resolve_written value the upload saw
    uint32_t reuploads = 0;
    // Set by OnGpuCacheInvalidate when the guest says it rewrote memory this
    // entry reads. Exact where the sampled probe is only probabilistic, and it
    // costs nothing to check: one bool on the cache-hit path.
    bool dirty = false;
    uint32_t phys = 0;         // base_address & 0x1FFFFFFF, for range compares
};
// Cache key -> entry. Keyed on the base address AND on the format and
// swizzle, not on the address alone: the upload bakes the swizzle in (k_8_8 is
// expanded channel by channel on the CPU), so two bindings of the same memory
// with different destination selection are two different host textures. Keying
// on the address alone would hand the second one the first one's picture, and
// the symptom -- one surface wearing another's palette index -- looks like a
// decode bug rather than a cache bug.
std::unordered_map<uint64_t, TexCache> g_texture_index;
// The physical span every cached texture's source lies inside, maintained on
// upload. D3DDevice_InvalidateGpuCache fires ~8 times a frame and the cache
// holds hundreds of entries; this turns the common "nowhere near a texture"
// call into one pair of comparisons instead of a walk of the whole index.
uint32_t g_tex_phys_lo = UINT32_MAX;
uint32_t g_tex_phys_hi = 0;
// Destination physical address -> the frame index whose pixels were last
// scattered there by OnResolve. A texture bound at that address is stale the
// moment this moves; it is exact, unlike the byte probe, because we are the
// ones who wrote it.
std::unordered_map<uint32_t, uint32_t> g_resolve_written;

// ── The resolve mirror: closing the loop on the GPU ──────────────────────────
// The stamp above made the compositing pass CORRECT, and it made it slow. What
// the guest does every frame is:
//
//   render the 3D scene -> D3DDevice_Resolve into a 1920x1080 surface ->
//   bind that surface as a texture -> draw it full-screen with the HUD on top
//
// and what we did to serve it was a full round trip: read our 1280x720 frame
// target back to the CPU, nearest-scale and re-tile it into guest memory
// (OnResolve's scatter), then have EnsureTexture read those 8 MB straight back
// out, un-tile them and upload them to the GPU again -- every frame, because
// the stamp correctly says the memory changed. Measured at 1920x1080: it is
// most of the difference between 42fps in a race and the 60 the game wants.
//
// The pixels never needed to make that trip. They start on the GPU, in
// g_color_target, and they end up on the GPU, in a texture sampled by a
// full-screen quad. So the mirror is a GPU-side copy of the frame target taken
// at the moment OnResolve captures, and a bind of a resolve destination is
// served by it instead of by an upload.
//
// The guest-memory scatter is NOT removed. Game logic may read those pixels,
// and the boot handshake depends on resolve output landing in guest memory;
// this only stops us reading BACK what we already have.
//
// Three things make the substitution exact rather than approximate:
//
//  * CHANNELS. Compose the scatter with the unpack that used to undo it:
//        scatter: dest_byte[sel(lane)] = frame_channel[lane]   (last lane wins)
//        unpack:  shader_lane[lane]    = dest_byte[sel(lane)]
//    so shader_lane[lane] = frame_channel[last lane' with sel(lane') == sel],
//    and a sel of 4 or 5 is the unpack's constant 0 or 1. The two are inverses,
//    so for every swizzle Daytona actually uses (zyxw and zyx1 on the two 1080p
//    destinations) the composition is the IDENTITY, except that zyx1's alpha is
//    the constant 1. That is a component mapping and nothing more, which is why
//    the mirror needs no blit and no shader: it is B8G8R8A8_UNORM like the
//    frame target, and a texture VIEW carries the mapping. MirrorComponentMap
//    derives it from the fetch constant rather than hard-coding those two.
//
//  * SIZE. The mirror is 1280x720 and the destination is 1920x1080, and that is
//    fine because the CPU path was a NEAREST SCALE of the same 1280x720 pixels:
//    normalized texcoord u addresses the same content in both. The guest samples
//    it with normalized coordinates (nothing here hands a shader a texture
//    dimension), so the sampler does the scale the scatter used to do, better.
//
//  * FRESHNESS. g_resolve_gen records which mirror generation was scattered to
//    an address; a bind is served from the mirror only while the mirror still
//    holds that generation or the one after it.
std::unique_ptr<plume::RenderTexture> g_resolve_mirror;
// Views of the mirror live as long as the heap slot that points at them.
std::vector<std::unique_ptr<plume::RenderTextureView>> g_mirror_views;
// Texture cache key (address + format + swizzle) -> heap slot showing the
// mirror through that key's component mapping. Deliberately NOT g_texture_index:
// a mirror slot owns no texture of its own, and the upload path must never find
// one of these and try to overwrite its pixels.
std::unordered_map<uint64_t, uint32_t> g_mirror_slots;
// Bumped every time the mirror is refreshed from the frame target.
uint32_t g_mirror_gen = 0;
// Destination physical address -> the mirror generation scattered there.
std::unordered_map<uint32_t, uint32_t> g_resolve_gen;
uint32_t g_mirror_hits = 0;      // binds served from the mirror
uint32_t g_mirror_stale = 0;     // ... of those, from the generation after

// DT_PLUME_TEXPROBE=0 turns off the sampled-content check (g_texprobe, declared
// at the top of this namespace because Init() reads its environment variable);
// the resolve stamp still invalidates, which is the A/B for "is this texture
// stale or is it decoded wrong".
uint32_t g_texture_reuploads = 0;
uint64_t TextureCacheKey(const DecodedTexture& t) {
    return uint64_t(t.base_address) | (uint64_t(t.format) << 32) | (uint64_t(t.swizzle) << 40);
}
// DT_PLUME_TEXCONTROL: the whole heap is the control checker, so guest textures
// must not be uploaded over it.
bool g_control_texture_only = false;

// ── Samplers ─────────────────────────────────────────────────────────────────
// One LINEAR sampler used to serve every binding. That is wrong for the format
// this stage just taught the renderer: a k_8_8 colour-index texture's .x is a
// COORDINATE into the palette, and bilinear filtering averages two indices into
// a third that names an unrelated palette entry. The fetch constant says which
// filter the guest asked for, so the samplers are built from it.
//
// The shaders already index a bindless SAMPLER heap -- SharedConstants carries
// sN_SamplerDescriptorIndex at c12 -- so this needs no pipeline change, only
// distinct entries in the heap and the right index written per binding.
std::vector<std::unique_ptr<plume::RenderSampler>> g_samplers;
std::unordered_map<uint32_t, uint32_t> g_sampler_index;   // packed state -> heap slot

plume::RenderTextureAddressMode AddressModeFromClamp(uint32_t clamp) {
    switch (clamp) {
        case 0: return plume::RenderTextureAddressMode::WRAP;           // kRepeat
        case 1: return plume::RenderTextureAddressMode::MIRROR;         // kMirroredRepeat
        case 2:                                                          // kClampToEdge
        case 4: return plume::RenderTextureAddressMode::CLAMP;          // kClampToHalfway
        case 3:                                                          // kMirrorClampToEdge
        case 5: return plume::RenderTextureAddressMode::MIRROR_ONCE;    // kMirrorClampToHalfway
        case 6:
        case 7: return plume::RenderTextureAddressMode::BORDER;
        default: return plume::RenderTextureAddressMode::WRAP;
    }
}

// The sampler heap slot for a binding's state, creating it on first use.
// Falls back to slot 0 (the original LINEAR/WRAP sampler) if anything fails,
// which is exactly the behaviour this replaces.
uint32_t EnsureSampler(const DecodedTexture& t) {
    // kUseFetchConst (3) and kBaseMap (2) are not selections a 2D fetch makes;
    // treat anything that is not kPoint as linear.
    const uint32_t mag_point = (t.mag_filter == 0) ? 1u : 0u;
    const uint32_t min_point = (t.min_filter == 0) ? 1u : 0u;
    const uint32_t key = mag_point | (min_point << 1) | (t.clamp_x << 2) | (t.clamp_y << 5);
    auto it = g_sampler_index.find(key);
    if (it != g_sampler_index.end()) return it->second;
    if (g_samplers.size() + 1 >= kSamplerDescriptorSize) return 0;

    plume::RenderSamplerDesc desc;
    desc.magFilter = mag_point ? plume::RenderFilter::NEAREST : plume::RenderFilter::LINEAR;
    desc.minFilter = min_point ? plume::RenderFilter::NEAREST : plume::RenderFilter::LINEAR;
    // Nothing here uploads mips, so the mip mode only decides how a single
    // level is sampled; NEAREST keeps it off the fence.
    desc.mipmapMode = plume::RenderMipmapMode::NEAREST;
    desc.addressU = AddressModeFromClamp(t.clamp_x);
    desc.addressV = AddressModeFromClamp(t.clamp_y);
    desc.addressW = desc.addressU;
    auto sampler = g_device->createSampler(desc);
    if (sampler == nullptr) return 0;

    const uint32_t slot = uint32_t(g_samplers.size()) + 1;   // slot 0 is the default
    g_sampler_set->setSampler(slot, sampler.get());
    g_samplers.push_back(std::move(sampler));
    g_sampler_index[key] = slot;
    REXLOG_WARN("plume: sampler state mag {} min {} clampU {} clampV {} -> heap slot {}",
                mag_point ? "POINT" : "linear", min_point ? "POINT" : "linear",
                t.clamp_x, t.clamp_y, slot);
    return slot;
}

// A sampled hash of a texture's source bytes. Not a checksum of the whole
// surface: at 1080p that is 8 MB, and this runs on every binding of every draw.
// 32 windows of 16 bytes, spread across the range, is 512 bytes -- enough to
// tell "different art" from "the same art", which is the only question a cache
// has to answer, and cheap enough to ask thousands of times a frame.
//
// It cannot see a change confined to bytes it does not sample. That is why the
// resolve destinations do not rely on it (see g_resolve_written): for those we
// know exactly when they changed, because we changed them.
uint64_t ProbeSource(const DecodedTexture& t, uint32_t alias, uint32_t source_bytes,
                     const uint8_t* base) {
    ProfMs prof(g_ms_p_probe);
    ++g_probe_calls;
    if (source_bytes == 0) return 0;
    const uint8_t* p = dt_guest_mem::Resolve(base, alias | t.base_address, source_bytes);
    if (p == nullptr) return 0;
    constexpr uint32_t kWindows = 32, kWindowBytes = 16;
    // A surface smaller than one window is hashed whole and once. Spelled out
    // because the arithmetic below is unsigned: source_bytes - kWindowBytes
    // would wrap, and the read would run off the end of the range Resolve
    // validated.
    const uint32_t window = std::min<uint32_t>(kWindowBytes, source_bytes);
    const uint32_t span = source_bytes - window;
    uint64_t h = 1469598103934665603ull;                 // FNV-1a
    for (uint32_t w = 0; w < kWindows; ++w) {
        const uint64_t at = uint64_t(span) * w / (kWindows - 1);
        for (uint32_t i = 0; i < window; ++i) {
            h = (h ^ p[at + i]) * 1099511628211ull;
        }
        if (span == 0) break;
    }
    return h;
}

// Should this cache entry's source be probed on this bind?
//
// The probe answers "is this still the same art", and art changes BETWEEN
// frames, not between two draws of the same frame. It used to run on every bind
// of every sampler slot of every draw: MEASURED at 214,139 calls per 300 frames
// in a race, i.e. ~714 a frame for the few hundred distinct textures a frame
// actually binds. Once per entry per frame asks the same question of the same
// bytes and gets the same answer, and it costs detection nothing: a rewrite is
// still caught on the first bind of the following frame.
//
// Deliberately NOT a longer backoff (probe every Nth frame once an address has
// been still for a while). It was measured at 125ms of a 2,689ms `record`, so
// the extra saving would be a fraction of a fraction, and it would buy that with
// frames of latency on exactly the guest UI rewrites this probe exists to catch.
bool ProbeDue(const TexCache& c) {
    return c.probe_frame != g_frame_index + 1u;
}

// The component mapping that makes sampling the mirror identical to sampling
// what the scatter + unpack round trip would have produced. See the resolve
// mirror block above for the derivation; this is that derivation, spelled.
plume::RenderComponentMapping MirrorComponentMap(uint32_t swizzle) {
    static const plume::RenderSwizzle kChannel[4] = {
        plume::RenderSwizzle::R, plume::RenderSwizzle::G,
        plume::RenderSwizzle::B, plume::RenderSwizzle::A,
    };
    plume::RenderSwizzle out[4];
    for (uint32_t lane = 0; lane < 4; ++lane) {
        uint32_t sel = SwizzleLane(swizzle, lane);
        if (sel == 4) { out[lane] = plume::RenderSwizzle::ZERO; continue; }   // constant 0
        if (sel == 5) { out[lane] = plume::RenderSwizzle::ONE; continue; }    // constant 1
        // 6/7 are "untouched", which the unpack reads as byte `lane` -- the
        // same arithmetic with sel = lane.
        if (sel > 5) sel = lane;
        // The scatter writes each destination byte once per lane that selects
        // it, so a swizzle that selects one byte twice leaves the LAST such
        // lane's channel there, and every lane reading that byte sees it.
        uint32_t from = lane;
        for (uint32_t later = lane + 1; later < 4; ++later) {
            if (SwizzleLane(swizzle, later) == sel) from = later;
        }
        out[lane] = kChannel[from];
    }
    return plume::RenderComponentMapping(out[0], out[1], out[2], out[3]);
}

// A heap slot showing the mirror as this binding's texture, created on first
// use. One slot per cache key, so the two swizzles the game's two 1080p
// destinations carry get two views of the SAME texture rather than two copies.
uint32_t EnsureMirrorSlot(uint64_t key, const DecodedTexture& t) {
    auto it = g_mirror_slots.find(key);
    if (it != g_mirror_slots.end()) return it->second;
    if (g_textures.size() + 1 >= kTextureDescriptorSize) return UINT32_MAX;

    auto view = g_resolve_mirror->createTextureView(
        [&] {
            plume::RenderTextureViewDesc desc =
                plume::RenderTextureViewDesc::Texture2D(kRenderTargetFormat);
            desc.componentMapping = MirrorComponentMap(t.swizzle);
            return desc;
        }());
    if (view == nullptr) return UINT32_MAX;

    const uint32_t slot = uint32_t(g_textures.size());
    g_textures.push_back(nullptr);   // the slot owns no texture; the mirror does
    g_texture_set->setTexture(slot, g_resolve_mirror.get(),
                              plume::RenderTextureLayout::SHADER_READ, view.get());
    g_mirror_views.push_back(std::move(view));
    g_mirror_slots[key] = slot;
    REXLOG_WARN("plume: resolve destination {:08X} {}x{} swizzle {} -> mirror heap slot {} "
                "({}x{} frame target, sampled instead of re-uploaded)",
                t.base_address, t.width, t.height, SwizzleName(t.swizzle), slot, kFrameW, kFrameH);
    return slot;
}

// Upload the texture bound to `sampler` and return its slot in the bindless
// heap, or UINT32_MAX. Cached by guest base address: the menu rebinds the same
// 1920x1080 surface every draw, and untiling it each time would be absurd.
//
// A cache HIT still has to ask whether the bytes are the ones that were
// uploaded. See TexCache for what happened when it did not.
//
// The decoded fetch constant comes back out through `t`: the caller needs it
// too, for EnsureSampler, and decoding it twice per slot per draw was 16 reads
// of guest memory a draw for an answer we already had.
uint32_t EnsureTexture(uint32_t sampler, const uint8_t* base, DecodedTexture& t) {
    t = ReadTextureFetchConstant(sampler, base);
    if (g_control_texture_only) return 0;   // every slot is the checker
    if (!t.valid) return UINT32_MAX;
    const uint64_t key = TextureCacheKey(t);

    // Is this a surface we resolved into, whose pixels the mirror still holds?
    // Then they are already on the GPU and the whole upload is dead work. Only
    // k_8_8_8_8 -- that is what a resolve destination is, and it is the format
    // the channel derivation above is written against.
    //
    // "Still holds" is one generation of slack: the mirror only ever moves
    // FORWARD, so a bind served one generation late shows the newer frame
    // rather than a stale one. Anything older falls through to the upload,
    // which reads guest memory and is correct by construction.
    if (g_gpuloop && g_resolve_mirror != nullptr && t.format == 6) {
        auto gen = g_resolve_gen.find(t.base_address & 0x1FFFFFFFu);
        if (gen != g_resolve_gen.end() && g_mirror_gen - gen->second <= 1) {
            const uint32_t slot = EnsureMirrorSlot(key, t);
            if (slot != UINT32_MAX) {
                ++g_mirror_hits;
                if (g_mirror_gen != gen->second) ++g_mirror_stale;
                return slot;
            }
        }
    }

    auto it = g_texture_index.find(key);
    uint32_t reuse_slot = UINT32_MAX;
    if (it != g_texture_index.end()) {
        TexCache& c = it->second;
        // Exact: did WE write new pixels there since the upload?
        uint32_t stamp = 0;
        if (auto rw = g_resolve_written.find(t.base_address & 0x1FFFFFFFu);
            rw != g_resolve_written.end()) {
            stamp = rw->second;
        }
        const char* why = nullptr;
        if (stamp != c.resolve_stamp) {
            why = "resolved into";
        } else if (c.dirty) {
            // The guest's own signal, so it is exact -- and it is checked
            // BEFORE the probe because it is also the cheaper question.
            why = "InvalidateGpuCache";
        } else if (g_texprobe && ProbeDue(c)) {
            c.probe_frame = g_frame_index + 1u;
            if (ProbeSource(t, c.source_alias, c.source_bytes, base) != c.probe) {
                why = "contents changed";
            }
        }
        if (why == nullptr) return c.slot;

        ++c.reuploads;
        ++g_texture_reuploads;
        // One line the first time an entry re-uploads for a NEW reason, plus
        // one every 100th overall. The old rule (first two per entry) printed
        // three lines for every texture the game ever streams, which at a few
        // hundred textures was most of a default run's log.
        static std::set<std::pair<uint32_t, const char*>> reasons;
        const bool novel = reasons.emplace(t.base_address, why).second;
        if (novel || (g_texture_reuploads % 100) == 0) {
            REXLOG_WARN("plume: texture {:08X} {}x{} {} -- re-uploading (this one {} times, {} "
                        "re-uploads overall)",
                        t.base_address, t.width, t.height, why, c.reuploads,
                        g_texture_reuploads);
        }
        reuse_slot = c.slot;
    }

    // Say it out loud when one address is bound two ways -- that is the case
    // the key above exists for, and if it never happens the extra key bits are
    // free rather than load-bearing.
    {
        static std::map<uint32_t, uint32_t> first_swizzle;
        auto ins = first_swizzle.emplace(t.base_address, t.swizzle);
        if (!ins.second && ins.first->second != t.swizzle) {
            REXLOG_WARN("plume: texture {:08X} bound with swizzle {} after {} -- uploading a "
                        "second copy rather than aliasing",
                        t.base_address, SwizzleName(t.swizzle),
                        SwizzleName(ins.first->second));
        }
    }

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
    //
    // Only on the FIRST upload of an address: re-uploads now happen every frame
    // for the resolve destinations, and an 8 MB byte count plus a log line at
    // 60 Hz is a measurement that pays for nothing. It no longer gets a line of
    // its own either -- it is folded into the one line the upload prints.
    size_t nonzero = 0;
    if (reuse_slot == UINT32_MAX) {
        for (uint8_t byte : up.data) nonzero += (byte != 0);
    }

    // A re-upload keeps its heap slot and, when the geometry has not changed,
    // its texture object: overwriting the pixels of a texture the open command
    // list already references is fine (the upload list is submitted and waited
    // on before that list is ever recorded further), whereas FREEING it is not.
    plume::RenderTexture* target = nullptr;
    std::unique_ptr<plume::RenderTexture> fresh;
    if (reuse_slot != UINT32_MAX) {
        const TexCache& c = g_texture_index[key];
        if (c.width == t.width && c.height == t.height && c.format == up.format) {
            target = g_textures[reuse_slot].get();
        }
    }
    if (target == nullptr) {
        fresh = g_device->createTexture(
            plume::RenderTextureDesc::Texture2D(t.width, t.height, 1, up.format));
        if (fresh == nullptr) return UINT32_MAX;
        target = fresh.get();
    }
    plume::RenderTexture* texture = target;

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
    const plume::RenderTextureBarrier to_dst(texture, plume::RenderTextureLayout::COPY_DEST);
    g_upload_cmd->barriers(plume::RenderBarrierStage::COPY, nullptr, 0, &to_dst, 1);
    g_upload_cmd->copyTextureRegion(
        plume::RenderTextureCopyLocation::Subresource(texture),
        plume::RenderTextureCopyLocation::PlacedFootprint(
            staging.get(), up.format, t.width, t.height, 1, footprint_texels));
    const plume::RenderTextureBarrier to_read(texture, plume::RenderTextureLayout::SHADER_READ);
    g_upload_cmd->barriers(plume::RenderBarrierStage::GRAPHICS, nullptr, 0, &to_read, 1);
    g_upload_cmd->end();
    const plume::RenderCommandList* lists[] = {g_upload_cmd.get()};
    g_queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0, g_upload_fence.get());
    g_queue->waitForCommandFence(g_upload_fence.get());

    uint32_t slot = reuse_slot;
    if (slot == UINT32_MAX) {
        slot = uint32_t(g_textures.size());
        g_textures.push_back(nullptr);
    }
    if (fresh != nullptr) {
        // The slot's previous texture (if any) outlives this call: draws already
        // recorded still point at it.
        if (g_textures[slot] != nullptr) {
            g_texture_graveyard.push_back(std::move(g_textures[slot]));
        }
        g_textures[slot] = std::move(fresh);
        g_texture_set->setTexture(slot, g_textures[slot].get(),
                                  plume::RenderTextureLayout::SHADER_READ);
    }

    TexCache& c = g_texture_index[key];
    c.slot = slot;
    c.width = t.width;
    c.height = t.height;
    c.format = up.format;
    c.source_bytes = up.source_bytes;
    c.source_alias = up.source_alias;
    c.probe = ProbeSource(t, up.source_alias, up.source_bytes, base);
    c.dirty = false;                                  // this upload IS the answer
    c.phys = t.base_address & 0x1FFFFFFFu;
    g_tex_phys_lo = std::min(g_tex_phys_lo, c.phys);
    g_tex_phys_hi = std::max(g_tex_phys_hi, c.phys + up.source_bytes);
    if (auto rw = g_resolve_written.find(t.base_address & 0x1FFFFFFFu);
        rw != g_resolve_written.end()) {
        c.resolve_stamp = rw->second;
    }
    // ONE line per new texture. It used to be three -- an alias probe, a
    // non-zero byte count, and this -- which at a few hundred textures was
    // most of a default run's log for information any one of them carries.
    if (reuse_slot == UINT32_MAX) {
        REXLOG_WARN("plume: uploaded texture {}x{} guest format {} as plume format {} {} base "
                    "{:08X} via {:08X} swizzle {} ({:03X}) endian {} sign {}{}{}{} numfmt {} "
                    "-- {} of {} bytes non-zero -> heap slot {}",
                    t.width, t.height, t.format, static_cast<uint32_t>(up.format),
                    t.tiled ? "tiled" : "linear", t.base_address, up.source_alias,
                    SwizzleName(t.swizzle), t.swizzle, t.endianness, t.sign[0], t.sign[1],
                    t.sign[2], t.sign[3], t.num_format, nonzero, up.data.size(), slot);
    }

    // DT_PLUME_TEXDUMP=<dir>: the staged RGBA8 bytes, exactly as the GPU will
    // see them. A palette is 64x1024 of shading ramps and no log line can say
    // what is in it; the file can be read against the referee's colours.
    if (const char* dump_dir = std::getenv("DT_PLUME_TEXDUMP")) {
        // A re-upload rewrites the file, so the dump always holds the CURRENT
        // art -- but not for the 8 MB resolve destinations, which change every
        // frame and would turn the dump into a 480 MB/s write.
        const bool too_big_to_redump = reuse_slot != UINT32_MAX && up.data.size() > (1u << 20);
        if (dump_dir[0] != '\0' && up.block == 1 && !too_big_to_redump) {
            const std::string path = fmt::format("{}/tex_{:08X}_{}x{}_f{}_s{:03X}.raw", dump_dir,
                                                 t.base_address, t.width, t.height, t.format,
                                                 t.swizzle);
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(up.data.data(), 1, up.data.size(), f);
                std::fclose(f);
            }
        }
    }
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

    // The resolve mirror: same size and same format as the frame target, so the
    // refresh is a straight copyTexture and the channel order needs no argument.
    // Failing to create it is not fatal -- the CPU round trip is still there and
    // still correct, only slower.
    if (g_gpuloop) {
        g_resolve_mirror = g_device->createTexture(
            plume::RenderTextureDesc::Texture2D(kFrameW, kFrameH, 1, kRenderTargetFormat));
        if (g_resolve_mirror == nullptr) {
            REXLOG_WARN("plume: resolve mirror failed -- resolve destinations will re-upload");
        }
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

// Copy one slot's vertex range out of guest memory, rewriting each attribute's
// bytes into the order the host input assembler must see.
//
// ⚠ The swap is 8in32 -- a blanket dword reverse -- because that is what the
// vertex fetch constant asks the GPU for, and it applies to the packed 8_8_8_8
// colour exactly as much as it does to a float. The old code reversed WITHIN
// each component instead, which is the same thing for 32-bit components and a
// no-op for 8-bit ones, so a D3DCOLOR reached the shader still in guest byte
// order: A R G B read as R G B A. Grey asphalt (80 80 80 with alpha FF) came
// out as red 1.0, green 0.5, blue 0.5 -- the magenta wash over the whole race.
//
// The reorder is not just the endian, though: see BuildAttributePermutation for
// the swizzle half. Bytes not covered by any attribute are copied through
// untouched rather than zeroed -- padding we have not identified is not the
// same thing as padding we know to be unused.
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
        uint8_t perm[16] = {};
        const uint32_t width = BuildAttributePermutation(a, perm);
        // A format with no permutation is one we do not understand well enough
        // to touch; copying it through is the only honest option.
        if (width == 0 || PermutationIsIdentity(perm, width)) continue;
        for (uint32_t v = 0; v < vertex_count; ++v) {
            const uint32_t at = v * stride + a.offset;
            if (at + width > bytes) break;       // attribute runs past the range
            uint8_t tmp[16];
            std::memcpy(tmp, &out[at], width);
            for (uint32_t b = 0; b < width; ++b) out[at + b] = tmp[perm[b]];
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

// Execute a clear the guest asked for, into the currently open command list.
// The flag layout is the console's, not the PC one: TARGET is four bits, one
// per MRT slot (d3d9types.h:192), so the colour test is a mask and not == 1.
void ApplyGuestClear(uint32_t flags) {
    if ((flags & 0x0Fu) != 0) {
        g_cmd->clearColor(0, plume::RenderColor(g_clear_color[0], g_clear_color[1],
                                                g_clear_color[2], g_clear_color[3]));
    }
    const bool depth = (flags & 0x10u) != 0;      // D3DCLEAR_ZBUFFER
    const bool stencil = (flags & 0x20u) != 0;    // D3DCLEAR_STENCIL
    if (depth || stencil) {
        g_cmd->clearDepthStencil(depth, stencil, g_clear_z, g_clear_stencil);
    }
}

// Open the frame's command list. Called lazily by the first draw of a frame
// rather than at the previous Swap, so a frame in which the game draws nothing
// costs nothing and does not present a stale picture.
//
// It no longer clears: see g_persist above. The one clear it still does is the
// PRIMING clear, because a freshly created target holds whatever the allocator
// left in it and the guest's first clear may not arrive before the first draw.
bool BeginFrame() {
    if (g_frame_open) return true;
    if (!EnsureOffscreen()) return false;

    // The previous frame's fence was waited on inside its EndFrame, so nothing
    // is still reading the arena's bytes.
    ArenaRewind();
    g_cmd->begin();
    const plume::RenderTextureBarrier to_write[] = {
        plume::RenderTextureBarrier(g_color_target.get(), plume::RenderTextureLayout::COLOR_WRITE),
        plume::RenderTextureBarrier(g_depth_target.get(), plume::RenderTextureLayout::DEPTH_WRITE),
    };
    g_cmd->barriers(plume::RenderBarrierStage::GRAPHICS, nullptr, 0, to_write, 2);
    g_cmd->setFramebuffer(g_framebuffer.get());
    if (!g_persist || !g_target_primed) {
        g_cmd->clearColor(0, plume::RenderColor(kClearR, kClearG, kClearB, 1.0f));
        g_cmd->clearDepthStencil();
        g_target_primed = true;
        g_clear_pending = 0;
    } else if (g_clear_pending != 0) {
        ApplyGuestClear(g_clear_pending);
        g_clear_pending = 0;
    }
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
// Presents and guest Swaps over the same 300-EndFrame window. These two numbers
// are the standing check on the fix below: one present per Swap is the whole
// point, and 2 presents per Swap is the flicker coming back.
uint32_t g_presents = 0;
uint32_t g_swaps = 0;
// ... of which came from PresentTarget, i.e. Swaps that arrived with the frame
// already closed by a resolve. Counted because a fallback that never fires is
// untested code, and one that fires constantly means the Swap is not where the
// frame is being finished.
uint32_t g_presents_closed = 0;
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
        bool built = false;
        {
            ProfMs prof(g_ms_p_vtx);
            built = BuildSlotBytes(c.attributes, c.slot_consts[sl], c.slots[sl].stride,
                                   start_vertex, vertex_count, base, bytes, src_ea);
        }
        if (!built) {
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
            // The palette lookup's ROW is TEXCOORD1.y, and 1024 rows of shading
            // ramp is a lot of picture riding on two floats we never printed.
            // Reported from the HOST-ORDER bytes, i.e. exactly what the input
            // assembler will hand the shader -- decoding the guest copy again
            // would only re-test the decoder.
            if (c.recorded_draws <= 2) {
                const uint32_t stride = c.slots[sl].stride;
                for (const auto& a : c.attributes) {
                    if (a.fetch_const != c.slot_consts[sl]) continue;
                    std::string vals;
                    for (uint32_t v = 0; v < vertex_count && v < 6; ++v) {
                        float f[4] = {};
                        const uint32_t at = v * stride + a.offset;
                        for (uint32_t k = 0; k < 4 && (at + k * 4 + 4) <= bytes.size(); ++k)
                            std::memcpy(&f[k], bytes.data() + at + k * 4, sizeof(float));
                        vals += fmt::format("({:.4f},{:.4f},{:.4f},{:.4f}) ", f[0], f[1], f[2],
                                            f[3]);
                    }
                    REXLOG_WARN("plume: draw {} ps {} attr usage {}.{} fmt {} off {} raw floats: {}",
                                c.recorded_draws, s.entry != nullptr ? s.entry->name : "?",
                                a.usage, a.usage_index, a.format, a.offset, vals);
                }
            }
        }
        Step("createBuffer(vertex)");
        ProfMs prof_vbuf(g_ms_p_vbuf);
        // The GPU reads these bytes when the frame is submitted, not now, so
        // they have to outlive this function: the arena owns them until the
        // next rewind, or the frame owns the buffer until its fence.
        const UploadSpan vb = TakeUpload(bytes.size(), 16, plume::RenderBufferFlag::VERTEX);
        if (!vb) return;
        std::memcpy(vb.cpu, bytes.data(), bytes.size());
        views.emplace_back(vb.buffer->at(vb.offset), uint32_t(bytes.size()));
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
            REXLOG_WARN("plume: sampler {} -> texture {}x{} format {} {} base {:08X} pitch {} "
                        "swizzle {}",
                        sampler, t.width, t.height, t.format,
                        t.tiled ? "TILED" : "linear", t.base_address, t.pitch * 32,
                        SwizzleName(t.swizzle));
        }
        // Positive control against REX_DUMP_ALUCONST's ground truth for c72..c75:
        // {2,0,0,0 / 0,2,0,0 / 0,0,-1,0 / -1,-1,0,1}.
        for (uint32_t r = 72; r <= 75; ++r) {
            REXLOG_WARN("plume: captured vertex c{} = {:.4f} {:.4f} {:.4f} {:.4f}", r,
                        g_vs_constants[r][0], g_vs_constants[r][1],
                        g_vs_constants[r][2], g_vs_constants[r][3]);
        }
    }
    // A root CBV's address must be 256-byte aligned on D3D12, so that is the
    // alignment the arena is asked for regardless of backend.
    auto upload_constants = [&](const void* src, uint64_t bytes, UploadSpan& out) -> bool {
        out = TakeUpload(bytes, 256, plume::RenderBufferFlag::CONSTANT);
        if (!out) return false;
        std::memcpy(out.cpu, src, size_t(bytes));
        return true;
    };
    UploadSpan shared_cb;
    bool cb_ok = false;
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
    uint32_t unresolved = 0;
    {
    ProfMs prof_tex(g_ms_p_tex);
    for (uint32_t s_i = 0; s_i < 8; ++s_i) {
        DecodedTexture t;
        const uint32_t slot = EnsureTexture(s_i, base, t);
        if (slot == UINT32_MAX) {
            // ⚠ Leaving the index at 0 is not "no texture" -- 0 is the control
            // CHECKER, and a shader that adds its sampler instead of modulating
            // by it paints the checker over the picture. ps_4a5273cf8fbd0839,
            // which every 3D surface uses, does exactly that:
            //     r1.xyz = tex2D(texture_color_ram, ...)      // sampler 1
            //     r0.xyz = saturate(r1.xyz * 2 + r0.xyz * 2)
            // so an unresolved sampler 1 saturates to magenta or green over
            // the whole scene. Counted here so it is a number, not a theory.
            unresolved |= 1u << s_i;
            continue;
        }
        shared[s_i / 4][s_i % 4] = slot;
        shared[12 + s_i / 4][s_i % 4] = EnsureSampler(t);
    }
    }
    // Once per combination: this runs on every draw and the answer is a
    // property of the combination, not of the draw.
    if (!c.logged_samplers) {
        c.logged_samplers = true;
        REXLOG_WARN("plume: combo {} ({}) textures s0={} s1={} s2={} s3={}{}",
                    uint32_t(&c - g_combos), s.entry->name,
                    shared[0][0], shared[0][1], shared[0][2], shared[0][3],
                    unresolved != 0
                        ? fmt::format(" -- UNRESOLVED sampler mask {:02X}, those read heap slot 0, "
                                      "the control checker", unresolved)
                        : "");
    }
    {
        ProfMs prof_cb(g_ms_p_cb);
        // The two constant FILES are re-uploaded only when the guest has changed
        // them since this frame's last upload; the SHARED block carries this
        // draw's own bindless indices, so it is per draw.
        cb_ok = true;
        if (g_cb_gen != g_const_gen || !g_cb_vs || !g_cb_ps) {
            cb_ok = upload_constants(g_vs_constants, sizeof(g_vs_constants), g_cb_vs) &&
                    upload_constants(g_ps_constants, sizeof(g_ps_constants), g_cb_ps);
            g_cb_gen = cb_ok ? g_const_gen : UINT32_MAX;
        }
        cb_ok = cb_ok && upload_constants(shared, sizeof(shared), shared_cb);
    }
    if (!cb_ok) {
        REXLOG_WARN("plume: could not upload the shader constant buffers");
        return;
    }

    const plume::RenderViewport viewport(0.0f, 0.0f, float(kFrameW), float(kFrameH));
    const plume::RenderRect scissor(0, 0, int32_t(kFrameW), int32_t(kFrameH));

    // From here it is pure recording into the frame's already-open list. No
    // clear: the frame was cleared once in BeginFrame, and clearing per draw is
    // exactly what stopped the previous stage from ever producing a picture.
    ++g_frame_draws;
    {
    ProfMs prof_cmd(g_ms_p_cmd);
    Step("setPipeline");
    g_cmd->setPipeline(c.pipeline.get());
    Step("setGraphicsPipelineLayout");
    g_cmd->setGraphicsPipelineLayout(g_pipeline_layout.get());
    // Root descriptor order matches how the layout was built: b0, b1, b2.
    Step("setGraphicsRootDescriptor x3");
    g_cmd->setGraphicsRootDescriptor(g_cb_vs.buffer->at(g_cb_vs.offset), 0);
    g_cmd->setGraphicsRootDescriptor(g_cb_ps.buffer->at(g_cb_ps.offset), 1);
    g_cmd->setGraphicsRootDescriptor(shared_cb.buffer->at(shared_cb.offset), 2);
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
    }

    // The constant buffers are read at submit time like the vertex data, and
    // TakeUpload has already put them where they will outlive this call.

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

// Close the frame: submit it, optionally copy it into the backbuffer and
// present, and (for the first few frames) measure what actually landed.
//
// The measurement is not redundant with the window. A window shows a person
// what happened; the readback is what an unattended run, or a run on a machine
// with no display, can still report -- and it is the thing that caught a
// renderer reporting success while drawing nothing.
//
// ── Why `present` is a parameter ─────────────────────────────────────────────
// Closing a frame and PUTTING IT ON THE SCREEN are two different events, and
// conflating them is what made the HUD flicker.
//
// EndFrame runs twice per guest frame: once at the guest's Swap, and once at
// the mid-frame D3DDevice_Resolve that copies the finished 3D scene out of
// EDRAM (OnResolve has to close the frame there, because the pixels the guest
// asked for only exist once the recording is submitted). The guest then draws
// its HUD/overlay pass and Swaps. So a frame that presented at BOTH points put
// two different pictures on the screen: the bare 3D scene at the resolve, and
// the composited scene+HUD at the Swap.
//
// MEASURED: 300 EndFrames took a metronomic 2500ms -- 120.0/s against a 60Hz
// guest, i.e. exactly 2 presents per guest frame -- and a 20-shot in-race burst
// classified 13 composited / 7 bare. The bare seven were not corrupt frames.
// They were legitimate mid-frame states that had no business being presented.
//
// Everything else a resolve-EndFrame does it still does: the readback that
// captures g_resolve_pixels for the guest, the mirror refresh, the fence wait,
// the buffer and arena lifecycle. Only the acquire/copy/present is skipped.
void EndFrame(bool present) {
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
                    "{:.0f}ms; {} binds served from the mirror ({} a generation late); "
                    "{} presents ({} of them from an already-closed frame) for {} guest Swaps",
                    wall, wall > 0 ? 300000.0 / wall : 0.0, g_ms_endframe, g_ms_scatter,
                    g_scatter_count, g_ms_record,
                    wall - g_ms_endframe - g_ms_scatter - g_ms_record,
                    g_mirror_hits, g_mirror_stale, g_presents, g_presents_closed, g_swaps);
        if (g_prof) {
            REXLOG_WARN("plume: record breakdown over 300 frames -- vtx {:.0f}ms, vbuf {:.0f}ms, "
                        "tex {:.0f}ms (probe {:.0f}ms in {} calls), cb {:.0f}ms, cmd {:.0f}ms, "
                        "rest {:.0f}ms; arena {} chunks, {} KB used at this frame's peak",
                        g_ms_p_vtx, g_ms_p_vbuf, g_ms_p_tex, g_ms_p_probe, g_probe_calls,
                        g_ms_p_cb, g_ms_p_cmd,
                        g_ms_record - g_ms_p_vtx - g_ms_p_vbuf - g_ms_p_tex - g_ms_p_cb -
                            g_ms_p_cmd,
                        g_arena_chunks.size(), g_arena_peak / 1024);
        }
        g_last_endframe = now;
        g_ms_endframe = g_ms_scatter = g_ms_record = 0.0;
        g_ms_p_vtx = g_ms_p_vbuf = g_ms_p_tex = g_ms_p_probe = g_ms_p_cb = g_ms_p_cmd = 0.0;
        g_probe_calls = 0;
        g_scatter_count = 0;
        g_mirror_hits = g_mirror_stale = 0;
        g_presents = g_presents_closed = g_swaps = 0;
    }

    const bool report = g_capture_resolve ||
        g_frame_index <= kReportedFrames || (g_frame_index % kReportInterval) == 0;

    // DT_PLUME_SWAPPRESENT=0 restores the old present-at-every-EndFrame, which
    // is the A/B for the flicker.
    const bool do_present = g_swap_present ? present : true;
    uint32_t swap_index = 0;
    plume::RenderTexture* backbuffer = nullptr;
    if (do_present && g_swapchain != nullptr && !g_swapchain->needsResize() &&
        g_swapchain->acquireTexture(nullptr, &swap_index)) {
        backbuffer = g_swapchain->getTexture(swap_index);
    }

    // One barrier batch: the frame target stops being a render target, and the
    // backbuffer becomes a copy destination.
    // The mirror joins that batch on a capturing frame. Ordering inside one
    // command list is what makes this safe: the draws that SAMPLE the mirror
    // were recorded above, the copy that overwrites it is recorded here, and
    // the SHADER_READ -> COPY_DEST transition is the barrier between them.
    const bool refresh_mirror = g_capture_resolve && g_resolve_mirror != nullptr;
    std::vector<plume::RenderTextureBarrier> to_copy;
    to_copy.emplace_back(g_color_target.get(), plume::RenderTextureLayout::COPY_SOURCE);
    if (backbuffer != nullptr) {
        to_copy.emplace_back(backbuffer, plume::RenderTextureLayout::COPY_DEST);
    }
    if (refresh_mirror) {
        to_copy.emplace_back(g_resolve_mirror.get(), plume::RenderTextureLayout::COPY_DEST);
    }
    g_cmd->barriers(plume::RenderBarrierStage::COPY, nullptr, 0, to_copy.data(),
                    uint32_t(to_copy.size()));

    if (refresh_mirror) {
        // Same dimensions, same format: no scale, no conversion, no shader.
        g_cmd->copyTexture(g_resolve_mirror.get(), g_color_target.get());
        const plume::RenderTextureBarrier to_read(g_resolve_mirror.get(),
                                                  plume::RenderTextureLayout::SHADER_READ);
        g_cmd->barriers(plume::RenderBarrierStage::GRAPHICS, nullptr, 0, &to_read, 1);
        ++g_mirror_gen;
    }

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
        ++g_presents;
    }
#ifdef _WIN32
    // Unconditional: a frame that does not present still has to keep the
    // window's message queue drained or it is marked unresponsive.
    PumpWindow();
#endif

    const size_t buffers = g_frame_buffers.size();
    g_frame_buffers.clear();
    // Textures a re-upload replaced. The fence above is what makes this safe.
    g_texture_graveyard.clear();

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
                // What THIS EndFrame did, not what the swapchain is capable of.
                // Half of them close a frame mid-guest-frame and do not present,
                // and a label that said "presented" on those would be the kind
                // of true-sounding number this spike has been misled by before.
                backbuffer != nullptr ? ", presented"
                                      : (g_swapchain != nullptr ? ", not presented (mid-frame "
                                                                  "resolve)"
                                                                : ""));

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

// Put the frame target on the screen without touching the recording.
//
// A guest Swap is the only moment a picture is finished, but by the time it
// arrives the frame may already have been CLOSED -- OnResolve closes it, and
// the guest's last resolve of a frame can come after its last draw, leaving
// nothing for the Swap's EndFrame to do but return. Presenting the target as it
// stands is exactly right then: it holds the composited frame, byte for byte
// what the resolve-EndFrame submitted. The alternative, skipping the present,
// leaves the PREVIOUS guest frame on the screen, which is a stutter traded for
// a flicker.
//
// It borrows g_upload_cmd rather than g_cmd: g_cmd is between begin() and end()
// for most of a frame and calling begin() on an open list is invalid, whereas
// nothing is uploading at a Swap with no frame open.
void PresentTarget() {
    if (g_offscreen != OffscreenState::kReady || !g_target_primed) return;
    if (g_swapchain == nullptr || g_swapchain->needsResize()) return;
    uint32_t swap_index = 0;
    if (!g_swapchain->acquireTexture(nullptr, &swap_index)) return;
    plume::RenderTexture* backbuffer = g_swapchain->getTexture(swap_index);
    if (backbuffer == nullptr) return;

    // The frame target has been in COPY_SOURCE since the EndFrame that closed
    // the frame and it is not transitioned back until the next BeginFrame, so
    // only the backbuffer needs a barrier here.
    g_upload_cmd->begin();
    const plume::RenderTextureBarrier to_dst(backbuffer, plume::RenderTextureLayout::COPY_DEST);
    g_upload_cmd->barriers(plume::RenderBarrierStage::COPY, nullptr, 0, &to_dst, 1);
    g_upload_cmd->copyTexture(backbuffer, g_color_target.get());
    const plume::RenderTextureBarrier to_present(backbuffer,
                                                 plume::RenderTextureLayout::PRESENT);
    g_upload_cmd->barriers(plume::RenderBarrierStage::GRAPHICS, nullptr, 0, &to_present, 1);
    g_upload_cmd->end();

    const plume::RenderCommandList* lists[] = {g_upload_cmd.get()};
    g_queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0, g_upload_fence.get());
    g_queue->waitForCommandFence(g_upload_fence.get());
    g_swapchain->present(swap_index, nullptr, 0);
    ++g_presents;
    ++g_presents_closed;
#ifdef _WIN32
    PumpWindow();
#endif
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

// D3DDevice_ClearF (sub_8222C390) -- the ONE place the guest clears anything.
// D3DDevice_Clear (sub_8222C4C8) unpacks its D3DCOLOR into a float4 and tail-
// calls this, so hooking here catches both entry points (every observed call
// arrives with lr=8222C53C, i.e. from inside D3DDevice_Clear).
//
//   D3DDevice_ClearF(D3DDevice*, DWORD Flags, CONST D3DRECT* pRect,
//                    CONST D3DVECTOR4* pColor, float Z, DWORD Stencil)
//   (XDK 20209 d3d9.h:2176)
//
// pRect == nullptr means the whole render target -- which is what the game
// passes, so a sub-rect clear is a case this deliberately only LOGS.
void OnClear(uint32_t flags, uint32_t rect_ea, uint32_t color_ea, float z, uint32_t stencil,
             const uint8_t* base) {
    if (!Enabled() || !g_init_ok) return;
    std::lock_guard<std::mutex> lock(g_mutex);

    if ((flags & 0x0Fu) != 0 && color_ea != 0) {
        // A D3DVECTOR4 of big-endian floats, already in the 0..1 range the
        // console clears with -- D3DDevice_Clear does the D3DCOLOR unpack.
        if (const uint8_t* p = dt_guest_mem::Resolve(base, color_ea, 16)) {
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t bits = LoadBE32(p + i * 4);
                float f;
                std::memcpy(&f, &bits, 4);
                g_clear_color[i] = f;
            }
        }
    }
    g_clear_z = z;
    g_clear_stencil = stencil;

    ++g_clears_seen;
    if (g_clears_seen <= 8 || (g_clears_seen % 600) == 0) {
        REXLOG_WARN("plume: CLEAR #{} flags {:#x}{}{}{} colour ({:.3f},{:.3f},{:.3f},{:.3f}) "
                    "z={:.3f} stencil={} rect={:08X} (frame {} {})",
                    g_clears_seen, flags,
                    (flags & 0x0Fu) ? " target" : "", (flags & 0x10u) ? " z" : "",
                    (flags & 0x20u) ? " stencil" : "",
                    g_clear_color[0], g_clear_color[1], g_clear_color[2], g_clear_color[3],
                    z, stencil, rect_ea, g_frame_index,
                    g_frame_open ? "open" : "closed");
    }

    if (!g_persist) return;                 // BeginFrame clears unconditionally
    if (g_frame_open) {
        // Mid-frame: execute it now, so draws recorded before it survive and
        // draws after it start from the cleared target -- guest order.
        ApplyGuestClear(flags);
    } else {
        // Between frames: the clear belongs to the frame the next draw opens.
        // OR, not assign: two clears before one draw must both happen.
        g_clear_pending |= flags;
    }
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
        // present=false: this is a MID-FRAME close. The guest has finished its
        // 3D scene and wants the pixels, but it has not drawn its HUD/overlay
        // pass yet, so what is in the target is not what the frame will look
        // like. Presenting it here is what put bare-3D frames on the screen.
        EndFrame(false);
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
    // g_resolve_written, not a local static, because EnsureTexture reads it: a
    // texture bound at an address we have just scattered into is stale, and
    // this stamp is the exact signal for that (see TexCache).
    auto& stamp = g_resolve_written[phys];
    // Which mirror generation these pixels are. Recorded before the dedup
    // early-out below, so it is the answer to "does the mirror still hold what
    // this address holds" whether or not this call is the one that scatters.
    g_resolve_gen[phys] = g_mirror_gen;
    if (stamp == g_frame_index + 1u) return;   // +1: 0 must not match a fresh entry
    stamp = g_frame_index + 1u;

    // Nearest-scale our frame into the destination, packed as the exact
    // inverse of ReadTexture's k_8_8_8_8 unpack -- so it moves with it. The
    // unpack is now "data component N is byte N, the fetch constant's swizzle
    // says which reaches each shader lane", so the pack asks the same question
    // backwards: the byte lane L selects has to receive shader channel L.
    ScopedMs scatter_timer(g_ms_scatter);
    ++g_scatter_count;

    // Everything this loop used to recompute two million times a frame that
    // does not vary from texel to texel:
    //
    //   * the horizontal sample map (x -> source column),
    //   * the destination byte offset, which for a tiled surface is a
    //     TiledOffset2D call per texel,
    //   * which source byte each destination byte takes -- four SwizzleLane
    //     calls per texel, for a permutation that is fixed for the surface.
    //
    // All three depend only on the surface's geometry and swizzle, so they are
    // built once and cached. Measured before: the scatter was 4031ms of the
    // 5618ms it took to run 300 boot frames (72%, two 1080p destinations per
    // frame) and 2042ms of 4969ms in a race (41%).
    //
    // 1080p costs 8.3 MB of offset table. The table is keyed on GEOMETRY ONLY.
    // Keying it on the swizzle as well looked tidier and was wrong: the game's
    // two 1080p destinations are identical in geometry but differ in swizzle
    // (zyx1 and zyxw), so the cache alternated and rebuilt 61 times a second
    // -- slower than the loop it replaced. The rebuild counter in the log line
    // is what caught that, so it stays.
    struct ScatterMap {
        uint32_t pitch = 0, w = 0, h = 0, needed = 0;
        bool tiled = false;
        std::vector<uint32_t> off;   // (y*w + x) -> destination byte offset
        std::vector<uint32_t> sx;    // x -> source column
    };
    static ScatterMap map;
    if (map.pitch != pitch_texels || map.w != t.width || map.h != t.height ||
        map.needed != needed || map.tiled != t.tiled) {
        static uint32_t rebuilds = 0;
        ++rebuilds;
        map.pitch = pitch_texels;
        map.w = t.width;
        map.h = t.height;
        map.needed = needed;
        map.tiled = t.tiled;
        map.sx.resize(t.width);
        for (uint32_t x = 0; x < t.width; ++x) map.sx[x] = x * kFrameW / t.width;
        map.off.assign(size_t(t.width) * t.height, UINT32_MAX);
        for (uint32_t y = 0; y < t.height; ++y) {
            uint32_t* orow = map.off.data() + size_t(y) * t.width;
            for (uint32_t x = 0; x < t.width; ++x) {
                const uint32_t off = t.tiled
                    ? uint32_t(TiledOffset2D(int32_t(x), int32_t(y), pitch_texels, 2))
                    : ((y * pitch_texels + x) * 4u);
                orow[x] = (off + 4 > needed) ? UINT32_MAX : off;   // same guard as before
            }
        }
        REXLOG_WARN("plume: scatter map #{} built -- {}x{} pitch {} {} ({:.1f} MB); a map "
                    "rebuilt every frame would cost more than it saves",
                    rebuilds, t.width, t.height, pitch_texels, t.tiled ? "tiled" : "linear",
                    double(map.off.size() * 4) / (1024.0 * 1024.0));
    }

    // Destination byte `sel` receives shader channel `lane`, and channel
    // `lane` lives at byte kBgra[lane] of the captured B8G8R8A8 pixel. Lanes
    // the swizzle does not select stay 0xFF and are not written -- exactly
    // what the per-texel `if (sel < 4)` did. Four calls per surface, not four
    // per texel, so this is per-scatter rather than cached.
    static const uint8_t kBgra[4] = {2, 1, 0, 3};
    uint8_t take[4] = {0xFF, 0xFF, 0xFF, 0xFF};
    for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint32_t sel = SwizzleLane(t.swizzle, lane);
        if (sel < 4) take[sel] = kBgra[lane];
    }

    const bool full_permutation = take[0] != 0xFF && take[1] != 0xFF &&
                                  take[2] != 0xFF && take[3] != 0xFF;
    for (uint32_t y = 0; y < t.height; ++y) {
        const uint32_t sy = y * kFrameH / t.height;
        const uint8_t* srow = g_resolve_pixels.data() + size_t(sy) * kFrameW * 4;
        const uint32_t* orow = map.off.data() + size_t(y) * t.width;
        const uint32_t* sxr = map.sx.data();
        for (uint32_t x = 0; x < t.width; ++x) {
            const uint32_t off = orow[x];
            if (off == UINT32_MAX) continue;
            const uint8_t* sp = srow + sxr[x] * 4;   // B8G8R8A8 as stored
            uint8_t* dp = dst + off;
            if (full_permutation) {
                const uint8_t out[4] = {sp[take[0]], sp[take[1]], sp[take[2]], sp[take[3]]};
                std::memcpy(dp, out, 4);
            } else {
                for (uint32_t b = 0; b < 4; ++b) {
                    if (take[b] != 0xFF) dp[b] = sp[take[b]];
                }
            }
        }
    }
    if (resolves <= 8 || (resolves % 600) == 0) {
        REXLOG_WARN("plume: RESOLVE #{} -- frame -> guest {:08X} {}x{} {} (fetch at +{:#x})",
                    resolves, t.base_address, t.width, t.height,
                    t.tiled ? "tiled" : "linear", used_off);
    }
}

// ── D3DDevice_Resolve's Flags: does Daytona clear THROUGH the resolve? ───────
//
// XDK 20209 d3d9types.h:222 gives Resolve's Flags two clear bits:
//
//   D3DRESOLVE_CLEARRENDERTARGET  0x0100   clear the RT after resolving
//   D3DRESOLVE_CLEARDEPTHSTENCIL  0x0200   clear depth/stencil after resolving
//
// with the values riding on Resolve's own pClearColor / ClearZ / ClearStencil
// parameters. So a title can clear EDRAM entirely through Resolve and never
// call Clear once -- and with the frame target now persisting across Swap,
// such a title presents as "the target never clears", which is indistinguish-
// able from a bug in the persistence.
//
// This MEASURES first. Building the clear-after-resolve path for a bit the
// game never sets would be dead code that can only rot; leaving the
// measurement in is cheap insurance that says so out loud if it ever changes.
//
// Read at D3DDevice_ResolveEx (sub_8222DA70) rather than at D3DDevice_Resolve,
// because the latter is a shim that shifts its arguments and tail-calls this
// one: Flags stays in r4 either way, so this sees BOTH entry points.
uint32_t g_resolve_flag_calls = 0;
struct ResolveFlagSeen { uint32_t flags = 0; uint32_t count = 0; };
constexpr uint32_t kMaxResolveFlags = 24;
ResolveFlagSeen g_resolve_flags[kMaxResolveFlags];
uint32_t g_resolve_flag_count = 0;
uint32_t g_resolve_flag_overflow = 0;
uint32_t g_resolve_clears = 0;   // resolve-clears folded into g_clear_pending

const char* ResolveSourceName(uint32_t flags) {
    switch (flags & 0x7u) {
        case 0: return "RT0";
        case 1: return "RT1";
        case 2: return "RT2";
        case 3: return "RT3";
        case 4: return "DEPTHSTENCIL";
        default: return "?";
    }
}

// MEASURED (boot -> menu -> circuit select -> race): TWO distinct values, and
// three resolves in four carry the clear.
//
//   0x0000  x14166   source RT0
//   0x0300  x42492   source RT0  CLEARRENDERTARGET  CLEARDEPTHSTENCIL
//
// with pClearColor reading (1.000, 1.000, 1.000, 0.000). So the persisted frame
// target is not merely allowed to be cleared by the guest's ClearF -- the guest
// clears it three times as often through the resolve, and ignoring that leaves
// every pass compositing on top of the previous pass's leftovers.
//
// The clear is recorded as PENDING rather than executed here, which is both
// cheaper and more correct: a resolve-clear happens AFTER the copy-out, and
// OnResolve (which runs later in the same guest call, from the D3DDevice_Resolve
// hook) is what captures and closes the frame. BeginFrame executes it on the
// next draw -- exactly the path a between-frames ClearF already takes.
void OnResolveFlags(uint32_t flags, uint32_t stack_ea, float clear_z, const uint8_t* base) {
    if (!Enabled()) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_resolve_flag_calls;

    // pClearColor rides in the outgoing parameter slot at r1+92 and ClearStencil
    // at r1+108 -- read off D3DDevice_Resolve's shim, which stores its incoming
    // r10 there (`stw r31,92(r1)`) rather than guessed from an ABI. ClearZ is a
    // float parameter, so it is in f1, which is why it arrives as an argument.
    float clear[4] = {0, 0, 0, 0};
    bool have_colour = false;
    uint32_t colour_ea = 0, stencil = 0;
    if ((flags & 0x0300u) != 0 && stack_ea != 0) {
        if (const uint8_t* slot = dt_guest_mem::Resolve(base, stack_ea + 92, 4)) {
            colour_ea = LoadBE32(slot);
            if (const uint8_t* p = dt_guest_mem::Resolve(base, colour_ea, 16)) {
                for (uint32_t i = 0; i < 4; ++i) {
                    const uint32_t bits = LoadBE32(p + i * 4);
                    std::memcpy(&clear[i], &bits, 4);
                }
                have_colour = true;
            }
        }
        if (const uint8_t* slot = dt_guest_mem::Resolve(base, stack_ea + 108, 4)) {
            stencil = LoadBE32(slot);
        }

        if (g_resolveclear) {
            // Translate resolve-clear bits into the D3DCLEAR_* vocabulary
            // ApplyGuestClear already speaks. TARGET is four bits (one per MRT)
            // and this resolve names one source, so only the colour bit for
            // RT0 is set -- ApplyGuestClear tests the mask, not equality.
            uint32_t pending = 0;
            if (flags & 0x0100u) {
                pending |= 0x01u;                       // D3DCLEAR_TARGET0
                if (have_colour) {
                    for (uint32_t i = 0; i < 4; ++i) g_clear_color[i] = clear[i];
                }
            }
            if (flags & 0x0200u) {
                pending |= 0x10u | 0x20u;               // ZBUFFER | STENCIL
                g_clear_z = clear_z;
                g_clear_stencil = stencil;
            }
            // Always pending, never immediate: the clear follows the copy-out,
            // and the copy-out has not happened yet at this point in the call.
            g_clear_pending |= pending;
            ++g_resolve_clears;
        }
    }

    for (uint32_t i = 0; i < g_resolve_flag_count; ++i) {
        if (g_resolve_flags[i].flags == flags) {
            ++g_resolve_flags[i].count;
            return;
        }
    }
    if (g_resolve_flag_count >= kMaxResolveFlags) {
        ++g_resolve_flag_overflow;
        return;
    }
    g_resolve_flags[g_resolve_flag_count++] = {flags, 1};

    REXLOG_WARN("plume: RESOLVE FLAGS {:#06x} (new, call #{}) -- source {} fragments {:#x} "
                "exp-bias {}{}{} (frame {})",
                flags, g_resolve_flag_calls, ResolveSourceName(flags), (flags >> 4) & 0x7u,
                int32_t(flags) >> 26,
                (flags & 0x0100u) ? "  *** CLEARRENDERTARGET ***" : "",
                (flags & 0x0200u) ? "  *** CLEARDEPTHSTENCIL ***" : "", g_frame_index);
    if (flags & 0x0300u) {
        REXLOG_WARN("plume:   resolve-clear colour ptr {:08X} -> {} z={:.3f} stencil={} -- {}",
                    colour_ea,
                    have_colour ? fmt::format("({:.3f},{:.3f},{:.3f},{:.3f})", clear[0], clear[1],
                                              clear[2], clear[3])
                                : std::string("(unreadable)"),
                    clear_z, stencil,
                    g_resolveclear ? "APPLIED to the frame target"
                                   : "IGNORED (DT_PLUME_RESOLVECLEAR=0)");
    }
}

// ── D3DDevice_InvalidateGpuCache (sub_82221818) ──────────────────────────────
//
//   void D3DDevice_InvalidateGpuCache(D3DDevice*, void* pBaseAddress, DWORD Size,
//                                     DWORD Flags)      (XDK 20209 d3d9.h:2286)
//
// r4/r5 verified against the recompiled body, not taken on the trace map's
// word: it saves r4 into r30 and r5 into r29, masks r30 with 0x1FFFFFFF (the
// physical address) and computes r29 = r30 + r29, i.e. an [address, address +
// size) range. That is an EXACT dirty signal, delivered by the game, for the
// case the sampled content probe is structurally blind to: a small patch to a
// UI atlas that lands entirely between the probe's 32 windows.
//
// Measured before it is wired, for the same reason as the resolve flags: a
// hook on a call the game never makes usefully is worse than no hook.
uint64_t g_gpucache_calls = 0;
struct GpuCacheRange { uint32_t base = 0; uint32_t size = 0; uint32_t count = 0; uint32_t hits = 0; };
constexpr uint32_t kMaxGpuCacheRanges = 96;
GpuCacheRange g_gpucache_ranges[kMaxGpuCacheRanges];
uint32_t g_gpucache_range_count = 0;
uint32_t g_gpucache_range_overflow = 0;
uint32_t g_gpucache_dirtied = 0;      // cache entries marked dirty by it

void OnGpuCacheInvalidate(uint32_t base_addr, uint32_t size) {
    if (!Enabled()) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_gpucache_calls;

    // Distinct (base,size) census, bounded. The first few of each are logged so
    // an unattended run records WHEN they happen relative to the screen it is
    // on; after that the census carries the counts and Dump prints it.
    bool novel = false;
    GpuCacheRange* rec = nullptr;
    for (uint32_t i = 0; i < g_gpucache_range_count; ++i) {
        if (g_gpucache_ranges[i].base == base_addr && g_gpucache_ranges[i].size == size) {
            rec = &g_gpucache_ranges[i];
            break;
        }
    }
    if (rec == nullptr) {
        if (g_gpucache_range_count < kMaxGpuCacheRanges) {
            rec = &g_gpucache_ranges[g_gpucache_range_count++];
            rec->base = base_addr;
            rec->size = size;
            novel = true;
        } else {
            ++g_gpucache_range_overflow;
        }
    }
    if (rec != nullptr) ++rec->count;
    if (novel && g_gpucache_range_count <= 8) {
        REXLOG_WARN("plume: INVALIDATE GPU CACHE {:08X}+{:#x} (new range #{}, call #{}, frame {})",
                    base_addr, size, g_gpucache_range_count, g_gpucache_calls, g_frame_index);
    }

    // Mark every cached texture whose source bytes the guest just rewrote, so
    // the next bind re-uploads without waiting for the sampled probe to happen
    // to land on a changed byte. The probe stays as the backstop for memory the
    // guest rewrites WITHOUT telling us.
    //
    // The address is a GPU address; the physical one is the low 29 bits, which
    // is what the function's own body masks off. The span check first: this
    // fires thousands of times a second and the index holds hundreds of
    // entries, and almost none of these calls are anywhere near a texture.
    if (size == 0 || g_tex_phys_lo >= g_tex_phys_hi) return;
    const uint32_t lo = base_addr & 0x1FFFFFFFu;
    const uint32_t hi = lo + size;
    if (hi <= g_tex_phys_lo || lo >= g_tex_phys_hi) return;
    for (auto& [key, c] : g_texture_index) {
        if (c.dirty || c.source_bytes == 0) continue;
        if (hi <= c.phys || lo >= c.phys + c.source_bytes) continue;
        c.dirty = true;
        ++g_gpucache_dirtied;
        if (rec != nullptr) ++rec->hits;
        if (g_gpucache_dirtied <= 8) {
            REXLOG_WARN("plume: INVALIDATE GPU CACHE {:08X}+{:#x} dirties cached texture {:08X}"
                        "+{:#x} ({}x{}) -- exactly the change the sampled probe can miss",
                        base_addr, size, c.phys, c.source_bytes, c.width, c.height);
        }
    }
}

// D3DDevice_Swap: the guest says the picture is finished. This is the ONE
// place a frame goes on the screen.
void OnFrameEnd() {
    if (!Enabled() || !g_init_ok) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_swaps;
    if (g_frame_open) {
        EndFrame(true);
    } else if (g_swap_present) {
        // Already closed by a resolve. The recording is gone but the picture is
        // not: present the target as it stands.
        PresentTarget();
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
    // The two guest-side dirty/clear signals, measured rather than assumed.
    REXLOG_WARN("   --- resolve Flags ({} calls, {} distinct{}, {} clears folded into the frame "
                "target) ---", g_resolve_flag_calls, g_resolve_flag_count,
                g_resolve_flag_overflow ? ", TABLE FULL" : "", g_resolve_clears);
    for (uint32_t i = 0; i < g_resolve_flag_count; ++i) {
        REXLOG_WARN("   {:#06x}  x{:<8} source {}{}{}", g_resolve_flags[i].flags,
                    g_resolve_flags[i].count, ResolveSourceName(g_resolve_flags[i].flags),
                    (g_resolve_flags[i].flags & 0x0100u) ? "  CLEARRENDERTARGET" : "",
                    (g_resolve_flags[i].flags & 0x0200u) ? "  CLEARDEPTHSTENCIL" : "");
    }
    // The texture span is the POSITIVE CONTROL for "dirtied 0": without it,
    // a span that was never populated and a guest that never invalidates a
    // texture produce the same zero, and only one of those is a finding.
    REXLOG_WARN("   --- InvalidateGpuCache ({} calls, {} distinct ranges{}, {} cache entries "
                "dirtied; cached textures span {:08X}..{:08X}) ---", g_gpucache_calls,
                g_gpucache_range_count, g_gpucache_range_overflow ? ", TABLE FULL" : "",
                g_gpucache_dirtied,
                g_tex_phys_lo == UINT32_MAX ? 0 : g_tex_phys_lo, g_tex_phys_hi);
    // Busiest ranges only: the table holds 96 and printing all of them made
    // this census the largest single item in a default log.
    {
        std::vector<uint32_t> order(g_gpucache_range_count);
        for (uint32_t i = 0; i < g_gpucache_range_count; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [](uint32_t a, uint32_t b) {
            return g_gpucache_ranges[a].count > g_gpucache_ranges[b].count;
        });
        const uint32_t show = std::min<uint32_t>(8, g_gpucache_range_count);
        for (uint32_t i = 0; i < show; ++i) {
            const GpuCacheRange& r = g_gpucache_ranges[order[i]];
            REXLOG_WARN("   {:08X}+{:<#9x} x{:<8} dirtied {}", r.base, r.size, r.count, r.hits);
        }
        if (g_gpucache_range_count > show) {
            REXLOG_WARN("   (+{} quieter ranges)", g_gpucache_range_count - show);
        }
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
    // The arena's chunks are mapped for their whole life; the drain above is
    // what makes destroying them here safe.
    ArenaRewind();
    g_arena_chunks.clear();
    g_shaders.clear();
    for (uint32_t i = 0; i < g_combo_count; ++i) g_combos[i].pipeline.reset();
    g_pipeline_layout.reset();
    g_quad_indices.reset();
    g_fan_indices.reset();
    g_index_state = IndexState::kNone;
    g_textures.clear();
    g_texture_index.clear();
    g_tex_phys_lo = UINT32_MAX;
    g_tex_phys_hi = 0;
    // The views go before the texture they view, and both before the heap they
    // are written into.
    g_mirror_views.clear();
    g_mirror_slots.clear();
    g_resolve_mirror.reset();
    g_texture_set.reset();
    g_sampler_set.reset();
    g_sampler.reset();
    g_samplers.clear();
    g_sampler_index.clear();
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

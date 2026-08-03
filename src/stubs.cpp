// Daytona USA (XBLA) -- Game-specific Xbox 360 API stubs
// These override or supplement the ReXGlue SDK's built-in kernel stubs.

#include "daytona_init.h"
#include "daytona_debug.h"
#include "daytona_symbols.h"
#include "dt_d3d9_trace.h"
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/system/xtypes.h>
#include <cstdio>
#include <cstdlib>

// Return a zero result from a stub (r3 = 0)
#define DT_STUB_RETURN(name, val) \
    extern "C" REX_FUNC(name) { (void)base; ctx.r3.u64 = (val); }

#define DT_STUB(name) DT_STUB_RETURN(name, 0)

// ── XUI / Overlay stubs (no Xbox Live — all no-ops) ──────────────────────────
// Wrappers: UiShowFriends, UiShowAchievements, UiShowMarketplace, UiShowGamerCard
DT_STUB(__imp__XamShowGamerCardUI)
DT_STUB(__imp__XamShowAchievementsUI)
DT_STUB(__imp__XamShowMarketplaceUI)
DT_STUB(__imp__XamShowFriendsUI)
DT_STUB(__imp__XamShowMessageComposeUI)

// ── Network stubs (offline — XNetConnect/GetConnectStatus no-op) ─────────────
// Wrappers: NetXNetConnect, NetGetConnectStatus
DT_STUB(__imp__NetDll_XNetUnregisterInAddr)
DT_STUB(__imp__NetDll_XNetConnect)
DT_STUB(__imp__NetDll_XNetGetConnectStatus)

// ── Leaderboard / Achievement stubs (no Xbox Live) ───────────────────────────
DT_STUB(__imp__XUserWriteAchievements)
DT_STUB(__imp__XUserCreateStatsEnumeratorByRank)
DT_STUB(__imp__XUserCreateStatsEnumeratorByXuid)
DT_STUB(__imp__XamShowGamerCardUIForXUID)

// ── Kernel I/O stubs (device driver path unused on PC) ───────────────────────
DT_STUB(__imp__IoCheckShareAccess)
DT_STUB(__imp__IoCompleteRequest)
DT_STUB(__imp__IoDeleteDevice)
DT_STUB(__imp__IoInvalidDeviceRequest)
DT_STUB(__imp__IoRemoveShareAccess)
DT_STUB(__imp__IoSetShareAccess)

// ── Kernel Object Manager stubs ───────────────────────────────────────────────
DT_STUB(__imp__ObIsTitleObject)
DT_STUB(__imp__ObReferenceObject)

namespace {

bool ShouldLog(uint32_t count) {
    return count <= 16 || (count % 1000) == 0;
}

void LogGuestCall(const char* name, uint32_t count, const PPCContext& ctx) {
    REXLOG_INFO("Daytona hook {} call #{} lr={:08X} r3={:08X} r4={:08X} r5={:08X}",
                name,
                count,
                static_cast<uint32_t>(ctx.lr),
                ctx.r3.u32,
                ctx.r4.u32,
                ctx.r5.u32);
}

void ReadGuestString(uint8_t* base, uint32_t address, char* out, size_t out_size) {
    if (out_size == 0) return;
    size_t i = 0;
    for (; i + 1 < out_size && address != 0; ++i) {
        const uint8_t c = REX_LOAD_U8(address + static_cast<uint32_t>(i));
        if (c == 0) break;
        out[i] = c >= 32 && c < 127 ? static_cast<char>(c) : '.';
    }
    out[i] = 0;
}

const char* DeclUsageName(uint8_t usage) {
    switch (usage) {
        case 0:  return "POSITION";
        case 1:  return "BLENDWEIGHT";
        case 2:  return "BLENDINDICES";
        case 3:  return "NORMAL";
        case 4:  return "PSIZE";
        case 5:  return "TEXCOORD";
        case 6:  return "TANGENT";
        case 7:  return "BINORMAL";
        case 8:  return "TESSFACTOR";
        case 10: return "COLOR";
        case 11: return "FOG";
        case 12: return "DEPTH";
        case 13: return "SAMPLE";
        default: return "?";
    }
}

// Dump a guest D3DVERTEXELEMENT9[] (12 bytes/element on the 360: WORD Stream,
// WORD Offset, DWORD Type, BYTE Method, BYTE Usage, BYTE UsageIndex, pad),
// terminated by Stream == 0xFF.
//
// This is the vertex format the native renderer has to reproduce. It cannot be
// read statically: the game builds the array on the stack at call time (see
// sub_8212F598, which stores Stream/Offset with sth and Method/Usage/UsageIndex
// with stb into a stack frame). Logged at WARN so it survives the shipped
// log_level=warn.
void DumpVertexElements(uint32_t elements_ea, uint8_t* base) {
    static bool dumped = false;
    if (dumped || elements_ea == 0) return;
    dumped = true;   // Daytona creates exactly one declaration; see dt_d3d9_trace
    REXLOG_WARN("===== D3DVERTEXELEMENT9[] @ {:08X} =====", elements_ea);
    for (uint32_t i = 0; i < 64; ++i) {
        const uint32_t p      = elements_ea + i * 12;
        const uint16_t stream = REX_LOAD_U16(p);
        if (stream == 0x00FF || stream == 0xFFFF) {
            REXLOG_WARN("   [{:2}] END (stream=0x{:X})", i, stream);
            break;
        }
        REXLOG_WARN("   [{:2}] stream={:<2} offset={:<4} type=0x{:08X} method={} usage={} ({}) index={}",
                    i, stream, REX_LOAD_U16(p + 2), REX_LOAD_U32(p + 4),
                    REX_LOAD_U8(p + 8), REX_LOAD_U8(p + 9),
                    DeclUsageName(REX_LOAD_U8(p + 9)), REX_LOAD_U8(p + 10));
    }
    REXLOG_WARN("===== end D3DVERTEXELEMENT9[] =====");
}

}  // namespace

#define DT_TRACE_HOOK(name, id)                         \
    REX_HOOK_RAW(name) {                                \
        uint32_t n = daytona_debug::RecordHook(id, ctx); \
        if (ShouldLog(n)) {                             \
            LogGuestCall(#name, n, ctx);                \
        }                                               \
        __imp__##name(ctx, base);                       \
    }

#define DT_TRACE_HOOK_AFTER(name, id)                   \
    REX_HOOK_RAW(name) {                                \
        __imp__##name(ctx, base);                       \
        uint32_t n = daytona_debug::RecordHook(id, ctx); \
        if (ShouldLog(n)) {                             \
            LogGuestCall(#name, n, ctx);                \
        }                                               \
    }

#define DT_TRACE_HOOK_AROUND(name, enter_id, exit_id)       \
    REX_HOOK_RAW(name) {                                    \
        uint32_t n = daytona_debug::RecordHook(enter_id, ctx); \
        if (ShouldLog(n)) {                                 \
            LogGuestCall(#name " enter", n, ctx);           \
        }                                                   \
        __imp__##name(ctx, base);                           \
        n = daytona_debug::RecordHook(exit_id, ctx);         \
        if (ShouldLog(n)) {                                 \
            LogGuestCall(#name " exit", n, ctx);            \
        }                                                   \
    }

#define DT_TRACE_HOOK_AFTER_EXCEPT(name, id)                \
    REX_HOOK_RAW(name) {                                    \
        uint32_t n = daytona_debug::RecordHook(id, ctx);    \
        if (ShouldLog(n)) {                                 \
            LogGuestCall(#name " enter", n, ctx);           \
        }                                                   \
        __imp__##name(ctx, base);                           \
    }

// ── Graphics / GPU trace hooks ────────────────────────────────────────────────
DT_TRACE_HOOK(sub_822346A0, daytona_debug::HookId::kGpuInterrupt)  // GpuInterruptHandler
// sub_82234FA8 = GPU descriptor submit. The old DESC diagnostic hook dumped
// guest memory at the raw register values (r3,r4,r8,r9,r10); when a register
// held a non-pointer like 0x81000000 (>=0x70000000 so it passed the weak
// safe_load guard, but unmapped) the dump itself faulted — a PHANTOM crash that
// looked like a descriptor bug. The game's own submit is fine; pass through.
REX_HOOK_RAW(sub_82234FA8) {
    __imp__sub_82234FA8(ctx, base);
}
DT_TRACE_HOOK(sub_824140B8, daytona_debug::HookId::kGpuCacheFlush)  // GpuCacheFlush
DT_TRACE_HOOK(sub_82130D68, daytona_debug::HookId::kBootPresentA)   // BootPresentA
REX_HOOK_RAW(sub_82131088) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kBootPresentB, ctx);
    if (ShouldLog(n)) {
        REXLOG_INFO(
            "Daytona boot present B enter #{} lr={:08X} r3={:08X} r4={:08X} r5={:08X}",
            n,
            static_cast<uint32_t>(ctx.lr),
            ctx.r3.u32,
            ctx.r4.u32,
            ctx.r5.u32);
    }

    __imp__sub_82131088(ctx, base);

    if (ShouldLog(n)) {
        REXLOG_INFO(
            "Daytona boot present B exit #{} result={:08X} lr={:08X}",
            n,
            ctx.r3.u32,
            static_cast<uint32_t>(ctx.lr));
    }
}
DT_TRACE_HOOK(sub_8223CF68, daytona_debug::HookId::kRenderPresentWorker)  // RenderPresentWorker
DT_TRACE_HOOK(sub_82233C78, daytona_debug::HookId::kFrameFenceReset)      // FrameFenceReset
REX_HOOK_RAW(sub_822337C0) {
    __imp__sub_822337C0(ctx, base);
}
DT_TRACE_HOOK(sub_82235B20, daytona_debug::HookId::kCommandFlush)  // CommandFlush
DT_TRACE_HOOK(sub_82235D58, daytona_debug::HookId::kCommandAlloc)  // CommandAlloc
REX_HOOK_RAW(sub_82233C80) {
    // spike/d3d9-hook-trace phase 0: this is D3DDevice_Swap (confirmed against
    // XDK 20500 d3d9.lib), so it is the frame tick. Drives the trace dump cadence
    // and doubles as the "did we render at all" control.
    dt_d3d9_trace::OnFrame();
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kVdSwapCaller, ctx);

    if (ShouldLog(n)) {
        REXLOG_INFO(
            "Daytona VdSwap caller enter #{} lr={:08X} r3={:08X} r4={:08X} r5={:08X} r6={:08X} r7={:08X} r8={:08X}",
            n,
            static_cast<uint32_t>(ctx.lr),
            ctx.r3.u32,
            ctx.r4.u32,
            ctx.r5.u32,
            ctx.r6.u32,
            ctx.r7.u32,
            ctx.r8.u32);
    }

    __imp__sub_82233C80(ctx, base);

    if (ShouldLog(n)) {
        REXLOG_INFO(
            "Daytona VdSwap caller exit #{} result={:08X}",
            n,
            ctx.r3.u32);
    }
}
DT_TRACE_HOOK_AFTER(sub_822297D8, daytona_debug::HookId::kCreateDeviceLike)   // D3DDevice-like create
DT_TRACE_HOOK_AFTER(sub_8221E940, daytona_debug::HookId::kSetRenderStateA)    // SetRenderState (A)
DT_TRACE_HOOK_AFTER(sub_8221E950, daytona_debug::HookId::kSetRenderStateB)    // SetRenderState (B)
DT_TRACE_HOOK_AFTER(sub_8221E658, daytona_debug::HookId::kSetRenderStateC)    // SetRenderState (C)
DT_TRACE_HOOK_AFTER(sub_822337E8, daytona_debug::HookId::kBackbufferSetup)    // BackbufferSetup
DT_TRACE_HOOK_AFTER(sub_8222C4C8, daytona_debug::HookId::kSceneSetup)         // SceneSetup
DT_TRACE_HOOK_AFTER(sub_8222EBC0, daytona_debug::HookId::kRenderTargetSetup)  // RenderTargetSetup
DT_TRACE_HOOK_AFTER(sub_8212F028, daytona_debug::HookId::kBeginSceneLike)     // BeginScene
DT_TRACE_HOOK_AFTER(sub_82223F28, daytona_debug::HookId::kClearLike)          // Clear
DT_TRACE_HOOK(sub_824DC8A8, daytona_debug::HookId::kMemcpyLikeDc8)            // Memcpy-like (DC8)
DT_TRACE_HOOK(sub_824EB2B8, daytona_debug::HookId::kMemcpyLikeEb2)            // Memcpy-like (EB2)
// ── Boot / Init sequence hooks ────────────────────────────────────────────────
DT_TRACE_HOOK_AROUND(sub_82242A30,
                     daytona_debug::HookId::kPostStateGateEnter,
                     daytona_debug::HookId::kPostStateGateExit)   // PostStateGate
DT_TRACE_HOOK_AROUND(sub_8212C7A8,
                     daytona_debug::HookId::kBootLogoSetupEnter,
                     daytona_debug::HookId::kBootLogoSetupExit)   // BootLogoSetup
DT_TRACE_HOOK_AROUND(sub_82130870,
                     daytona_debug::HookId::kBootPollEnter,
                     daytona_debug::HookId::kBootPollExit)        // BootPoll
DT_TRACE_HOOK_AFTER(sub_8212F000, daytona_debug::HookId::kEndSceneLike)       // EndScene
DT_TRACE_HOOK_AFTER(sub_8222AF38, daytona_debug::HookId::kBootCreateTarget)   // BootCreateTarget
DT_TRACE_HOOK_AFTER(sub_8222AE18, daytona_debug::HookId::kBootCreateTarget2)  // BootCreateTarget2
DT_TRACE_HOOK_AFTER(sub_822203B8, daytona_debug::HookId::kBootBindTarget)     // BootBindTarget
DT_TRACE_HOOK_AFTER(sub_822200E0, daytona_debug::HookId::kBootSetTarget)      // BootSetTarget
DT_TRACE_HOOK_AFTER(sub_8212E068, daytona_debug::HookId::kBootShaderPath)     // BootShaderPath
DT_TRACE_HOOK_AFTER(sub_82220730, daytona_debug::HookId::kBootAllocA)         // BootAllocA
DT_TRACE_HOOK_AFTER(sub_82221780, daytona_debug::HookId::kBootClearAllocA)    // BootClearAllocA
DT_TRACE_HOOK_AFTER(sub_822207F8, daytona_debug::HookId::kBootUnlockAllocA)   // BootUnlockAllocA
// sub_82223EA8 = D3DDevice_CreateVertexDeclaration, identified against XDK 20500
// d3d9.lib: its body walks a 12-byte-stride element array until Stream == 0xFF,
// then allocates 56 + 12*count and calls XGSetVertexDeclaration (sub_82223DB8).
// r3 on entry is the D3DVERTEXELEMENT9[]; it is clobbered by the return value,
// so capture it before calling through. The trace shows Daytona binds only ONE
// declaration game-wide, so this dump is the whole vertex format.
REX_HOOK_RAW(sub_82223EA8) {
    const uint32_t elements = ctx.r3.u32;
    __imp__sub_82223EA8(ctx, base);
    DumpVertexElements(elements, base);
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kBootMakeDraw, ctx);
    if (ShouldLog(n)) {
        LogGuestCall("CreateVertexDeclaration", n, ctx);
    }
}
DT_TRACE_HOOK_AFTER(sub_82220808, daytona_debug::HookId::kBootAllocB)         // BootAllocB
DT_TRACE_HOOK_AFTER(sub_822217D0, daytona_debug::HookId::kBootClearAllocB)    // BootClearAllocB
DT_TRACE_HOOK_AFTER(sub_822208B8, daytona_debug::HookId::kBootUnlockAllocB)   // BootUnlockAllocB
DT_TRACE_HOOK_AFTER(sub_8212C408, daytona_debug::HookId::kBootObjectSetup)    // BootObjectSetup
DT_TRACE_HOOK_AFTER(sub_82129DE0, daytona_debug::HookId::kBootUiSetupA)       // BootUiSetupA
DT_TRACE_HOOK_AFTER(sub_82126280, daytona_debug::HookId::kBootUiSetupB)       // BootUiSetupB
DT_TRACE_HOOK_AFTER(sub_8212A850, daytona_debug::HookId::kBootUiSetupC)       // BootUiSetupC
DT_TRACE_HOOK_AFTER(sub_82128638, daytona_debug::HookId::kBootUiSetupD)       // BootUiSetupD
DT_TRACE_HOOK_AROUND(sub_8212A928,
                     daytona_debug::HookId::kBootUiSetupF,
                     daytona_debug::HookId::kBootUiSetupFExit)    // BootUiSetupF
DT_TRACE_HOOK_AROUND(sub_8212C7B8,
                     daytona_debug::HookId::kBootUiSetupG,
                     daytona_debug::HookId::kBootUiSetupGExit)    // BootUiSetupG

// ── UI texture hooks ──────────────────────────────────────────────────────────
DT_TRACE_HOOK_AFTER(sub_82241750, daytona_debug::HookId::kUiTextureSizeA)         // UiTextureSizeA
DT_TRACE_HOOK_AFTER(sub_82225658, daytona_debug::HookId::kUiTextureFinalizeA)     // UiTextureFinalizeA
DT_TRACE_HOOK_AFTER(sub_82241740, daytona_debug::HookId::kUiTextureSizeB)         // UiTextureSizeB
DT_TRACE_HOOK_AFTER(sub_82225470, daytona_debug::HookId::kUiTextureFinalizeB)     // UiTextureFinalizeB

REX_HOOK_RAW(sub_8212F990) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kMainShaderSetup, ctx);
    if (ShouldLog(n)) {
        LogGuestCall("sub_8212F990 enter", n, ctx);
    }
    // Do not clear the built-in shader source pointers here. sub_8212F990
    // still walks those strings before the compile calls, so zeroing them can
    // stall inside the initializer. Let the original function run so the next
    // failing child call can be observed.
    __imp__sub_8212F990(ctx, base);
    n = daytona_debug::RecordHook(daytona_debug::HookId::kMainShaderSetup, ctx);
    if (ShouldLog(n)) {
        LogGuestCall("sub_8212F990 exit", n, ctx);
    }
}

REX_HOOK_RAW(sub_8212AD80) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kBootUiSetupE, ctx);
    if (ShouldLog(n)) {
        LogGuestCall("sub_8212AD80 bypass enter", n, ctx);
    }
    // This boot UI helper compiles built-in D3DX HLSL shaders. The current
    // guest D3DX path returns 0x8000FFFF and the title asserts before first
    // present. Leave the zeroed shader fields in place so boot can continue.
    n = daytona_debug::RecordHook(daytona_debug::HookId::kBootUiSetupEExit, ctx);
    if (ShouldLog(n)) {
        LogGuestCall("sub_8212AD80 bypass exit", n, ctx);
    }
}

REX_HOOK_RAW(sub_822429C8) {
    char path[160]{};
    ReadGuestString(base, ctx.r3.u32, path, sizeof(path));
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kUiTextureLoad, ctx);
    if (ShouldLog(n)) {
        REXLOG_INFO("Daytona hook sub_822429C8 enter #{} path='{}' len={} r5={:08X} r7={:08X} r8={:08X} r10={:08X}",
                    n,
                    path,
                    ctx.r4.u32,
                    ctx.r5.u32,
                    ctx.r7.u32,
                    ctx.r8.u32,
                    ctx.r10.u32);
    }
    __imp__sub_822429C8(ctx, base);
    if (ShouldLog(n)) {
        REXLOG_INFO("Daytona hook sub_822429C8 exit #{} result={:08X}", n, ctx.r3.u32);
    }
}

REX_HOOK_RAW(sub_82176F48) {
    REXLOG_INFO("Daytona parent sub_82176F48 enter lr={:08X} r3={:08X} r4={:08X} r5={:08X}",
                static_cast<uint32_t>(ctx.lr),
                ctx.r3.u32,
                ctx.r4.u32,
                ctx.r5.u32);

    __imp__sub_82176F48(ctx, base);

    REXLOG_INFO("Daytona parent sub_82176F48 exit result={:08X} lr={:08X}",
                ctx.r3.u32,
                static_cast<uint32_t>(ctx.lr));
}


REX_HOOK_RAW(sub_822F8D48) {
    const uint32_t obj = ctx.r3.u32;
    const uint32_t obj0 = obj ? REX_LOAD_U32(obj + 0) : 0;
    const uint32_t state = obj ? REX_LOAD_U32(obj + 4) : 0;

    if (state) {
        const uint32_t flags = REX_LOAD_U32(state + 40);
        if ((flags & 0x10000000u) == 0) {
            REX_STORE_U32(state + 40, flags | 0x10000000u);
        }

        const uint32_t expected = REX_LOAD_U32(state + 52);
        if (obj0 == expected) {
            REX_STORE_U32(state + 52, expected ^ 1u);
        }
    }

    __imp__sub_822F8D48(ctx, base);
}

// Daytona F9288 fix: sub_822F9288 (a boot shader-build routine) only reaches its
// build path when this memzero helper returns 0 at its two F9288 call sites
// (lr 0x822F9304 / 0x822F9484, len 0x9C0). Force that result.
REX_HOOK_RAW(sub_824DFC40) {
    const uint32_t lr = static_cast<uint32_t>(ctx.lr);
    const bool f9288_site =
        (lr == 0x822F9304u || lr == 0x822F9484u) &&
        ctx.r4.u32 == 0 &&
        ctx.r5.u32 == 0x000009C0u;
    __imp__sub_824DFC40(ctx, base);
    if (f9288_site) {
        ctx.r3.u64 = 0;
    }
}

// Trampoline: single `b 0x824C78D0` instruction.
extern "C" REX_FUNC(sub_824C8D80) {
    sub_824C78D0(ctx, base);
}

// Daytona manual missing leaf sub_82342258.
// The recompiler skipped this valid PPC leaf routine, but guest code calls it
// indirectly as a comparator from sub_824DEC68.
extern "C" REX_FUNC(sub_82342258) {
    const uint32_t left_obj = REX_LOAD_U32(ctx.r3.u32 + 0);
    const uint32_t right_obj = REX_LOAD_U32(ctx.r4.u32 + 0);

    auto key_from_word = [](uint32_t word) -> uint32_t {
        // PPC: rlwinm rX,rX,19,16,31
        return ((word << 19) | (word >> 13)) & 0xFFFFu;
    };

    uint32_t left_max = 0;
    for (uint32_t node = REX_LOAD_U32(left_obj + 0x20); node != 0; node = REX_LOAD_U32(node + 4)) {
        const uint32_t payload = REX_LOAD_U32(node + 0);
        const uint32_t word = REX_LOAD_U32(payload + 0);
        const uint32_t key = key_from_word(word);
        if (key > left_max) {
            left_max = key;
        }
    }

    uint32_t right_max = 0;
    for (uint32_t node = REX_LOAD_U32(right_obj + 0x20); node != 0; node = REX_LOAD_U32(node + 4)) {
        const uint32_t payload = REX_LOAD_U32(node + 0);
        const uint32_t word = REX_LOAD_U32(payload + 0);
        const uint32_t key = key_from_word(word);
        if (key > right_max) {
            right_max = key;
        }
    }

    if (left_max == right_max) {
        ctx.r3.s64 = 0;
    } else {
        // Matches the small helper at 0x823422CC: descending order.
        ctx.r3.s64 = (right_max > left_max) ? 1 : -1;
    }
}


// License flag: set to true for full version (trial mode when false)
static constexpr bool kFullVersion = true;

// Override the default license mask stub.
extern "C" REX_FUNC(__imp__XamContentGetLicenseMask) {
    uint32_t mask_addr = ctx.r3.u32;
    if (mask_addr) {
        REX_STORE_U32(mask_addr, kFullVersion ? 0xFFFFFFFF : 0);
    }
    ctx.r3.u64 = 0;
}

// ── Audio domain hooks ────────────────────────────────────────────────────────
// AudioInitDriver: XAudioGetSpeakerConfig + XAudioRegisterRenderDriverClient
REX_HOOK_RAW(AudioInitDriver) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kAudioInitDriver, ctx);
    if (ShouldLog(n))
        LogGuestCall("AudioInitDriver enter", n, ctx);
    __imp__sub_8242E438(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona AudioInitDriver exit #{} result={:08X}", n, ctx.r3.u32);
}

// AudioSubmitRenderFrame: XAudioSubmitRenderDriverFrame
REX_HOOK_RAW(AudioSubmitRenderFrame) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kAudioSubmitRenderFrame, ctx);
    const bool log = n <= 8 || (n % 5000) == 0;
    if (log)
        LogGuestCall("AudioSubmitRenderFrame", n, ctx);
    __imp__sub_8242E038(ctx, base);
}

// ── Input domain hooks ────────────────────────────────────────────────────────
// InputGetRawState: XamInputGetState + XamInputRawState + XNotifyGetNext
REX_HOOK_RAW(InputGetRawState) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kInputGetRawState, ctx);
    const bool log = n <= 8 || (n % 2000) == 0;
    if (log)
        LogGuestCall("InputGetRawState", n, ctx);
    __imp__sub_8245D500(ctx, base);
}

// InputGetCapabilitiesEx: XamInputGetCapabilitiesEx
REX_HOOK_RAW(InputGetCapabilitiesEx) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kInputGetCapabilitiesEx, ctx);
    if (ShouldLog(n))
        LogGuestCall("InputGetCapabilitiesEx", n, ctx);
    __imp__sub_8245D348(ctx, base);
}

// ── Video (Vd*) domain hooks ──────────────────────────────────────────────────
// GpuSetDisplayMode: VdGetCurrentDisplayInformation + VdSetDisplayMode + VdSetDisplayModeOverride
REX_HOOK_RAW(GpuSetDisplayMode) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kGpuSetDisplayMode, ctx);
    if (ShouldLog(n))
        LogGuestCall("GpuSetDisplayMode enter", n, ctx);
    __imp__sub_82237F18(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona GpuSetDisplayMode exit #{} result={:08X}", n, ctx.r3.u32);
}

// GpuInitEngines: VdInitializeEngines + VdSetGraphicsInterruptCallback
REX_HOOK_RAW(GpuInitEngines) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kGpuInitEngines, ctx);
    if (ShouldLog(n))
        LogGuestCall("GpuInitEngines enter", n, ctx);
    __imp__sub_82238028(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona GpuInitEngines exit #{} result={:08X}", n, ctx.r3.u32);
}

// GpuInitRingBuffer: VdInitializeRingBuffer + VdEnableRingBufferRPtrWriteBack
REX_HOOK_RAW(GpuInitRingBuffer) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kGpuInitRingBuffer, ctx);
    if (ShouldLog(n))
        LogGuestCall("GpuInitRingBuffer enter", n, ctx);
    __imp__sub_82235EA0(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona GpuInitRingBuffer exit #{} result={:08X}", n, ctx.r3.u32);
}

// ── Network domain hooks ──────────────────────────────────────────────────────
// NetXNetStartup: NetDll_XNetStartup + XamGetSystemVersion
REX_HOOK_RAW(NetXNetStartup) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kNetXNetStartup, ctx);
    if (ShouldLog(n))
        LogGuestCall("NetXNetStartup enter", n, ctx);
    __imp__sub_8240DA60(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona NetXNetStartup exit #{} result={:08X}", n, ctx.r3.u32);
}

// NetWsaStartup: NetDll_WSAStartup + XamGetSystemVersion
REX_HOOK_RAW(NetWsaStartup) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kNetWsaStartup, ctx);
    if (ShouldLog(n))
        LogGuestCall("NetWsaStartup enter", n, ctx);
    __imp__sub_8240DC38(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona NetWsaStartup exit #{} result={:08X}", n, ctx.r3.u32);
}

// ── XUI / Overlay domain hooks ────────────────────────────────────────────────
// UiShowMessageBoxEx: XamShowMessageBoxUIEx — used for in-game error dialogs
REX_HOOK_RAW(UiShowMessageBoxEx) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kUiShowMessageBoxEx, ctx);
    if (ShouldLog(n))
        LogGuestCall("UiShowMessageBoxEx enter", n, ctx);
    __imp__sub_82210B78(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona UiShowMessageBoxEx exit #{} result={:08X}", n, ctx.r3.u32);
}

// UiShowSignin: XamShowSigninUI
REX_HOOK_RAW(UiShowSignin) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kUiShowSignin, ctx);
    if (ShouldLog(n))
        LogGuestCall("UiShowSignin enter", n, ctx);
    __imp__sub_8220F1D0(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona UiShowSignin exit #{} result={:08X}", n, ctx.r3.u32);
}

// ── User / Profile domain hooks ───────────────────────────────────────────────
// UserGetSigninInfo: XamUserGetSigninInfo
REX_HOOK_RAW(UserGetSigninInfo) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kUserGetSigninInfo, ctx);
    if (ShouldLog(n))
        LogGuestCall("UserGetSigninInfo", n, ctx);
    __imp__sub_8220EC88(ctx, base);
}

// VoiceCheckHeadset: XamVoiceHeadsetPresent + XamVoiceIsActiveProcess + XamUserReadProfileSettings
REX_HOOK_RAW(VoiceCheckHeadset) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kVoiceCheckHeadset, ctx);
    if (ShouldLog(n))
        LogGuestCall("VoiceCheckHeadset", n, ctx);
    __imp__sub_82461C58(ctx, base);
}

// ── Session domain hooks ──────────────────────────────────────────────────────
// SessionCreate: XamSessionCreateHandle + XamSessionRefObjByHandle + XMsgStartIORequest
REX_HOOK_RAW(SessionCreate) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kSessionCreate, ctx);
    if (ShouldLog(n))
        LogGuestCall("SessionCreate enter", n, ctx);
    __imp__sub_82459FD0(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona SessionCreate exit #{} result={:08X}", n, ctx.r3.u32);
}

// ── Platform / system domain hooks ───────────────────────────────────────────
// SysPlatformInfo: DbgPrint + ExGetXConfigSetting + XGetAVPack + XGetLanguage — read platform config
REX_HOOK_RAW(SysPlatformInfo) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kSysPlatformInfo, ctx);
    if (ShouldLog(n))
        LogGuestCall("SysPlatformInfo enter", n, ctx);
    __imp__sub_82210CB0(ctx, base);
    if (ShouldLog(n))
        REXLOG_INFO("Daytona SysPlatformInfo exit #{} result={:08X}", n, ctx.r3.u32);
}

// LaunchTitle: XamLoaderLaunchTitle
REX_HOOK_RAW(LaunchTitle) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kLaunchTitle, ctx);
    if (ShouldLog(n))
        LogGuestCall("LaunchTitle", n, ctx);

    char launch_path[256];
    ReadGuestString(base, ctx.r3.u32, launch_path, sizeof(launch_path));
    // Daytona's boot shell relaunches via XamLoaderLaunchTitle (e.g.
    // 'XLNI_DASH_ARCADE'). The recompiled standalone title can't relaunch, so we
    // exit cleanly. (Ignoring it and continuing → heap corruption; see notes.)
    REXLOG_INFO("Daytona requested title launch/exit path='{}'; closing PC port", launch_path);
    std::fflush(nullptr);
    std::_Exit(0);
}

// ── Crypto hook ───────────────────────────────────────────────────────────────
// CryptoSha: XeCryptSha
REX_HOOK_RAW(CryptoSha) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kCryptoSha, ctx);
    if (ShouldLog(n))
        LogGuestCall("CryptoSha", n, ctx);
    __imp__sub_824F0110(ctx, base);
}

// ── Render pipeline orchestrator hooks ───────────────────────────────────────
// FrameRender (sub_8212B828): called once per frame from FrameEntry.
// Logs on first 5 calls and every 1000 thereafter to trace frame-render cadence.
REX_HOOK_RAW(FrameRender) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kFrameRender, ctx);
    const bool log = n <= 5 || (n % 1000) == 0;
    if (log)
        LogGuestCall("FrameRender", n, ctx);
    __imp__sub_8212B828(ctx, base);
    if (log)
        REXLOG_INFO("Daytona FrameRender exit #{} result={:08X}", n, ctx.r3.u32);
}

// SceneRender (sub_8212B3C8): dispatches to Scene3dRender; checks render-object dirty flags.
REX_HOOK_RAW(SceneRender) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kSceneRender, ctx);
    const bool log = n <= 5 || (n % 1000) == 0;
    if (log)
        LogGuestCall("SceneRender", n, ctx);
    __imp__sub_8212B3C8(ctx, base);
}

// Scene3dRender (sub_8222CB88): builds the 3D draw list; calls RenderTargetSetup.
REX_HOOK_RAW(Scene3dRender) {
    uint32_t n = daytona_debug::RecordHook(daytona_debug::HookId::kScene3dRender, ctx);
    const bool log = n <= 5 || (n % 1000) == 0;
    if (log)
        LogGuestCall("Scene3dRender", n, ctx);
    __imp__sub_8222CB88(ctx, base);
}

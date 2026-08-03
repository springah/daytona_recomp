#pragma once

#include <array>
#include <cstdint>

#include <rex/ppc/context.h>

namespace daytona_debug {

enum class AppStage : uint32_t {
    kStarting,
    kPreSetup,
    kPostSetup,
    kPreLaunchModule,
    kPostLaunchModule,
    kGuestThreadExit,
};

enum class HookId : size_t {
    kD3DInterruptCallback,
    kD3DAddCallsToPrimary,
    kGpuCacheFlush,
    kBootPresentA,
    kBootPresentB,
    kD3DWorkerThread,
    kD3DSyncToPresentation,
    kFrameFenceWait,
    kD3DKickOff,
    kD3DBeginRingBig,
    kD3DDeviceSwap,
    kCreateDeviceLike,
    kD3DSetRSPresentInterval,
    kSetRenderStateB,
    kD3DSetRSHalfPixelOffset,
    kBootPoll,
    kBackbufferSetup,
    kSceneSetup,
    kD3DResolve,
    kBeginSceneLike,
    kD3DSetShaderGPRAlloc,
    kMemcpyLikeDc8,
    kMemcpyLikeEb2,
    kPostStateGateEnter,
    kPostStateGateExit,
    kBootLogoSetupEnter,
    kBootLogoSetupExit,
    kBootPollEnter,
    kBootPollExit,
    kEndSceneLike,
    kD3DCreateSurface,
    kD3DCreateTexture,
    kBootBindTarget,
    kD3DSetDepthStencil,
    kBootShaderPath,
    kBootAllocA,
    kBootClearAllocA,
    kBootUnlockAllocA,
    kBootMakeDraw,
    kBootAllocB,
    kBootClearAllocB,
    kBootUnlockAllocB,
    kBootObjectSetup,
    kBootUiSetupA,
    kBootUiSetupB,
    kBootUiSetupC,
    kBootUiSetupD,
    kBootUiSetupE,
    kBootUiSetupEExit,
    kBootUiSetupF,
    kBootUiSetupFExit,
    kBootUiSetupG,
    kBootUiSetupGExit,
    kUiTextureSizeA,
    kUiTextureLoad,
    kD3DCreateVertexShader,
    kUiTextureSizeB,
    kD3DCreatePixelShader,
    kMainShaderSetup,

    // Audio domain
    kAudioInitDriver,
    kAudioSubmitRenderFrame,

    // Input domain
    kInputGetRawState,
    kInputGetCapabilitiesEx,

    // Video domain
    kD3DVideoModeInitialize,
    kD3DDriverInitialize,
    kD3DSetRingBufferParams,

    // Network domain
    kNetXNetStartup,
    kNetWsaStartup,

    // XUI / overlay domain
    kUiShowMessageBoxEx,
    kUiShowSignin,

    // User / profile domain
    kUserGetSigninInfo,
    kVoiceCheckHeadset,

    // Session domain
    kSessionCreate,

    // Platform / system domain
    kSysPlatformInfo,
    kLaunchTitle,

    // Crypto
    kCryptoSha,

    // Render pipeline orchestrators
    kFrameRender,
    kSceneRender,
    kD3DEndTiling,

    kCount,
};

struct HookStats {
    const char* name = "";
    uint32_t count = 0;
    uint32_t lr = 0;
    uint32_t r3 = 0;
    uint32_t r4 = 0;
    uint32_t r5 = 0;
    uint32_t thread_hash = 0;
    double last_seen_seconds = 0.0;
};

struct HookEvent {
    uint64_t sequence = 0;
    const char* name = "";
    uint32_t count = 0;
    uint32_t lr = 0;
    uint32_t r3 = 0;
    uint32_t r4 = 0;
    uint32_t r5 = 0;
    uint32_t thread_hash = 0;
    double age_seconds = 0.0;
};

struct Snapshot {
    AppStage stage = AppStage::kStarting;
    double uptime_seconds = 0.0;
    std::array<HookStats, static_cast<size_t>(HookId::kCount)> hooks{};
    std::array<HookEvent, 24> events{};
};

void SetStage(AppStage stage);
uint32_t RecordHook(HookId id, const PPCContext& ctx);
Snapshot GetSnapshot();
const char* StageName(AppStage stage);

// Dump every hook's count at WARN. The per-hook LogGuestCall lines are INFO and
// the shipped daytona.ini sets log_level=warn, so without this an unattended run
// yields no input / signin / session / storage evidence -- which is exactly what
// is needed to tell "waiting for START" apart from "blocked at a gate".
void LogCensus(const char* reason);

}  // namespace daytona_debug

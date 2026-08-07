// Daytona USA (XBLA) - ReXGlue Static Recompilation
// Title ID: 58410B1D

#include "daytona_init.h"
#include "daytona_debug.h"
#include "dt_d3d9_trace.h"
#include "dt_null_audio.h"
#include "dt_plume.h"
#include "dt_shader_dump.h"

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/graphics/flags.h>
#include <rex/logging/api.h>
#include <rex/platform.h>
#include <rex/rex_app.h>
#include <rex/input/input_system.h>
#include <rex/system/xthread.h>
#include <rex/ui/flags.h>
#include <imgui.h>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace {

std::string Trim(std::string_view value) {
    const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && is_space(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() && is_space(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string StripInlineComment(std::string_view value) {
    const size_t comment = value.find_first_of("#;");
    if (comment != std::string_view::npos) {
        value = value.substr(0, comment);
    }
    return Trim(value);
}

std::optional<int32_t> ParseInt(std::string_view value) {
    const std::string text = Trim(value);
    int32_t out = 0;
    auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    if (ec != std::errc() || ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return out;
}

std::optional<double> ParseDouble(std::string_view value) {
    try {
        size_t parsed = 0;
        const std::string text = Trim(value);
        double out = std::stod(text, &parsed);
        if (parsed != text.size()) {
            return std::nullopt;
        }
        return out;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> ParseBool(std::string_view value) {
    const std::string lowered = Lower(Trim(value));
    if (lowered == "true" || lowered == "yes" || lowered == "on" || lowered == "1") {
        return true;
    }
    if (lowered == "false" || lowered == "no" || lowered == "off" || lowered == "0") {
        return false;
    }
    return std::nullopt;
}

std::optional<std::pair<int32_t, int32_t>> ParseResolution(std::string_view value) {
    std::string text = Lower(Trim(value));
    std::replace(text.begin(), text.end(), 'X', 'x');
    const size_t sep = text.find('x');
    if (sep == std::string::npos) {
        return std::nullopt;
    }
    auto width = ParseInt(text.substr(0, sep));
    auto height = ParseInt(text.substr(sep + 1));
    if (!width || !height || *width <= 0 || *height <= 0) {
        return std::nullopt;
    }
    return std::pair<int32_t, int32_t>(*width, *height);
}

int32_t ClampInt(int32_t value, int32_t min, int32_t max) {
    return std::clamp(value, min, max);
}

std::filesystem::path ResolveIniPath(const std::filesystem::path& exe_dir,
                                     std::string_view value) {
    std::filesystem::path path = Trim(value);
    if (path.empty()) {
        return {};
    }
    return path.is_absolute() ? path : exe_dir / path;
}

void SetCvar(std::string_view name, std::string_view value) {
    rex::cvar::SetFlagByName(name, value);
}

void SetCvarInt(std::string_view name, int32_t value) {
    SetCvar(name, std::to_string(value));
}

void SetCvarDouble(std::string_view name, double value) {
    SetCvar(name, std::to_string(value));
}

void SetCvarBool(std::string_view name, bool value) {
    SetCvar(name, value ? "true" : "false");
}

void WriteDefaultIni(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        return;
    }

    std::ofstream file(path);
    if (!file) {
        return;
    }

    file
        << "# Daytona USA Recompiled PC settings\n"
        << "# Edit this file, then restart the game.\n"
        << "\n"
        << "[Display]\n"
        << "# Window size as WxH (host window). game_resolution is the guest\n"
        << "# render resolution; leave it at 1280x720 for authentic output.\n"
        << "resolution = 1920x1080\n"
        << "game_resolution = 1280x720\n"
        << "fullscreen = false\n"
        << "monitor = 0\n"
        << "vsync = true\n"
        << "refresh_rate = 60\n"
        << "letterbox = true\n"
        << "safe_area_x = 100\n"
        << "safe_area_y = 100\n"
        << "\n"
        << "[Graphics]\n"
        << "internal_resolution_scale = 1\n"
        << "# off, app, 1, 2, 4, 8, 16\n"
        << "anisotropic_filtering = app\n"
        << "# bilinear is always available. CAS/FSR options depend on SDK build support.\n"
        << "upscaler = bilinear\n"
        << "cas_sharpness = 0.0\n"
        << "fsr_quality = auto\n"
        << "dither = false\n"
        << "\n"
        << "[Storage]\n"
        << "cache_path = cache\n"
        << "# Leave empty to use the platform save folder.\n"
        << "user_data_path =\n"
        << "\n"
        << "[Diagnostics]\n"
        << "debug_overlay = false\n"
        << "log_level = warn\n"
        << "log_max_files = 5\n"
        << "log_max_file_size_mb = 5\n";
}

std::unordered_map<std::string, std::string> ReadIni(const std::filesystem::path& path) {
    std::unordered_map<std::string, std::string> values;
    std::ifstream file(path);
    if (!file) {
        return values;
    }

    std::string section;
    std::string line;
    while (std::getline(file, line)) {
        line = StripInlineComment(line);
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            section = Lower(Trim(std::string_view(line).substr(1, line.size() - 2)));
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = Lower(Trim(line.substr(0, eq)));
        std::string value = Trim(line.substr(eq + 1));
        if (!section.empty()) {
            key = section + "." + key;
        }
        values[std::move(key)] = std::move(value);
    }
    return values;
}

int32_t ParseAnisotropic(std::string_view value) {
    std::string text = Lower(Trim(value));
    if (text == "app" || text == "auto" || text == "default") {
        return -1;
    }
    if (text == "off" || text == "none" || text == "0") {
        return 0;
    }
    if (!text.empty() && text.back() == 'x') {
        text.pop_back();
    }
    auto parsed = ParseInt(text);
    if (!parsed) {
        return 3;
    }
    switch (ClampInt(*parsed, 1, 16)) {
        case 1: return 1;
        case 2: return 2;
        case 3:
        case 4: return 3;
        case 5:
        case 6:
        case 7:
        case 8: return 4;
        default: return 5;
    }
}

struct DaytonaIniSettings {
    bool debug_overlay = false;
};

DaytonaIniSettings LoadDaytonaIni(const std::filesystem::path& exe_dir,
                                  rex::PathConfig& paths) {
    const std::filesystem::path ini_path = exe_dir / "daytona.ini";
    WriteDefaultIni(ini_path);
    const auto values = ReadIni(ini_path);

    DaytonaIniSettings settings;

    auto set_resolution = [&](std::string_view key, std::string_view width_cvar,
                              std::string_view height_cvar) {
        if (auto it = values.find(std::string(key)); it != values.end()) {
            if (auto resolution = ParseResolution(it->second)) {
                SetCvarInt(width_cvar, ClampInt(resolution->first, 640, 8192));
                SetCvarInt(height_cvar, ClampInt(resolution->second, 480, 8192));
            }
        }
    };

    set_resolution("display.resolution", "window_width", "window_height");
    set_resolution("display.game_resolution", "video_mode_width", "video_mode_height");

    if (auto it = values.find("display.fullscreen"); it != values.end()) {
        if (auto value = ParseBool(it->second)) SetCvarBool("fullscreen", *value);
    }
    if (auto it = values.find("display.monitor"); it != values.end()) {
        if (auto value = ParseInt(it->second)) SetCvarInt("monitor", ClampInt(*value, 0, 16));
    }
    if (auto it = values.find("display.vsync"); it != values.end()) {
        if (auto value = ParseBool(it->second)) SetCvarBool("vsync", *value);
    }
    if (auto it = values.find("display.refresh_rate"); it != values.end()) {
        if (auto value = ParseDouble(it->second)) {
            SetCvarDouble("video_mode_refresh_rate", std::clamp(*value, 24.0, 240.0));
        }
    }
    if (auto it = values.find("display.letterbox"); it != values.end()) {
        if (auto value = ParseBool(it->second)) SetCvarBool("present_letterbox", *value);
    }
    if (auto it = values.find("display.safe_area_x"); it != values.end()) {
        if (auto value = ParseInt(it->second)) {
            SetCvarInt("present_safe_area_x", ClampInt(*value, 0, 100));
        }
    }
    if (auto it = values.find("display.safe_area_y"); it != values.end()) {
        if (auto value = ParseInt(it->second)) {
            SetCvarInt("present_safe_area_y", ClampInt(*value, 0, 100));
        }
    }

    if (auto it = values.find("graphics.internal_resolution_scale"); it != values.end()) {
        if (auto value = ParseInt(it->second)) {
            SetCvarInt("resolution_scale", ClampInt(*value, 1, 8));
        }
    }
    if (auto it = values.find("graphics.anisotropic_filtering"); it != values.end()) {
        SetCvarInt("anisotropic_override", ParseAnisotropic(it->second));
    }
    if (auto it = values.find("graphics.upscaler"); it != values.end()) {
        SetCvar("present_effect", Lower(Trim(it->second)));
    }
    if (auto it = values.find("graphics.cas_sharpness"); it != values.end()) {
        if (auto value = ParseDouble(it->second)) {
            SetCvarDouble("present_cas_additional_sharpness", std::clamp(*value, 0.0, 1.0));
        }
    }
    if (auto it = values.find("graphics.fsr_quality"); it != values.end()) {
        SetCvar("present_fsr_quality_mode", Lower(Trim(it->second)));
    }
    if (auto it = values.find("graphics.dither"); it != values.end()) {
        if (auto value = ParseBool(it->second)) SetCvarBool("present_dither", *value);
    }

    if (auto it = values.find("storage.cache_path"); it != values.end()) {
        if (auto path = ResolveIniPath(exe_dir, it->second); !path.empty()) {
            paths.cache_root = path;
        }
    }
    if (auto it = values.find("storage.user_data_path"); it != values.end()) {
        if (auto path = ResolveIniPath(exe_dir, it->second); !path.empty()) {
            paths.user_data_root = path;
        }
    }

    if (auto it = values.find("diagnostics.debug_overlay"); it != values.end()) {
        if (auto value = ParseBool(it->second)) settings.debug_overlay = *value;
    }
    if (auto it = values.find("diagnostics.log_level"); it != values.end()) {
        SetCvar("log_level", Lower(Trim(it->second)));
    }
    if (auto it = values.find("diagnostics.log_max_files"); it != values.end()) {
        if (auto value = ParseInt(it->second)) SetCvarInt("log_max_files", ClampInt(*value, 1, 50));
    }
    if (auto it = values.find("diagnostics.log_max_file_size_mb"); it != values.end()) {
        if (auto value = ParseInt(it->second)) {
            SetCvarInt("log_max_file_size_mb", ClampInt(*value, 1, 256));
        }
    }

    return settings;
}

class DaytonaDebugDialog : public rex::ui::ImGuiDialog {
public:
    explicit DaytonaDebugDialog(rex::ui::ImGuiDrawer* drawer) : ImGuiDialog(drawer) {}

protected:
    void OnDraw(ImGuiIO& io) override {
        auto snapshot = daytona_debug::GetSnapshot();
        UpdateVdSwapRate(snapshot, io.DeltaTime);

        ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(260.0f, 80.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowBgAlpha(0.55f);
        if (!ImGui::Begin("Daytona##perf", nullptr, ImGuiWindowFlags_NoResize)) {
            ImGui::End();
            return;
        }

        const float fps = io.Framerate;
        const float ms  = fps > 0.0f ? 1000.0f / fps : 0.0f;
        ImGui::Text("Host  %.1f fps  %.2f ms", fps, ms);
        ImGui::Text("Swap  %.1f fps  stage: %s",
                    vd_swap_rate_,
                    daytona_debug::StageName(snapshot.stage));

        ImGui::End();
    }

private:
    void UpdateVdSwapRate(const daytona_debug::Snapshot& snapshot, float delta_time) {
        rate_accumulator_ += std::max(delta_time, 0.0f);
        if (rate_accumulator_ < 0.25f) return;

        const uint32_t count = snapshot.hooks[static_cast<size_t>(daytona_debug::HookId::kD3DDeviceSwap)].count;
        vd_swap_rate_ = static_cast<float>(count - prev_vd_swap_count_) / rate_accumulator_;
        prev_vd_swap_count_ = count;
        rate_accumulator_ = 0.0f;
    }

    uint32_t prev_vd_swap_count_ = 0;
    float vd_swap_rate_ = 0.0f;
    float rate_accumulator_ = 0.0f;
};

}  // namespace

class DaytonaApp : public rex::ReXApp {
public:
    using rex::ReXApp::ReXApp;

    static std::unique_ptr<rex::ui::WindowedApp> Create(rex::ui::WindowedAppContext& ctx) {
        return std::unique_ptr<DaytonaApp>(new DaytonaApp(ctx, "daytona", PPCImageConfig));
    }

protected:
    void OnConfigurePaths(rex::PathConfig& paths) override {
        const std::filesystem::path exe_dir = rex::filesystem::GetExecutableFolder();
        // Leave config_path at the SDK default (<exe_dir>/daytona.ini) so the
        // "Save to config" overlay and any auto-loaded cvars use .ini, matching
        // the sibling recomps (futari/gh/guwange/sdoj/sotn) -- no .toml.
        paths.cache_root = exe_dir / "cache";
        ini_settings_ = LoadDaytonaIni(exe_dir, paths);
        // Default the game data root to an "assets" folder next to the
        // executable when no --game_data_root was provided, so the game runs on
        // a plain double-click (no launcher .bat needed), matching the sibling
        // recomps (futari/gh/guwange/sotn). An explicit --game_data_root (cvar
        // non-empty) still takes precedence.
        if (paths.game_data_root.empty()) {
            paths.game_data_root = exe_dir / "assets";
        }
    }

    void OnPreSetup(rex::RuntimeConfig& config) override {
        daytona_debug::SetStage(daytona_debug::AppStage::kPreSetup);
        // By-name set: this cvar's storage moved into the rexgpu-xenos plugin
        // DLL with the SDK's v0.9 GPU split, so the link-time REXCVAR_SET
        // macro can no longer reach it. The v0.9 deferred-registration cvar
        // semantics apply the value when the plugin registers the flag.
        rex::cvar::SetFlagByName("gpu_allow_invalid_fetch_constants", "true");

        // Since the SDK's v0.9 GPU-plugin split an empty gpu_plugin means
        // headless. This branch defaults to `plume`, which does NOT draw: it
        // supplies the three things a GPU is besides a renderer (the vblank tick
        // and its interrupt, the register window at 0x7FC80000, and draining the
        // ring) and discards the draws, because dt_plume has already recorded
        // them off the D3D9 hooks. See src/gpu_plume/plugin.cpp.
        //
        // `--gpu_plugin=xenos` is the no-rebuild A/B against Xenos emulation and
        // pairs with DT_PLUME=0. An explicit --gpu_plugin (cvar non-empty by
        // now) still takes precedence over both.
        if (config.gpu_plugin.empty()) {
            // DT_PLUME_NOGPU=1 leaves it empty, i.e. genuinely headless: no GPU
            // plugin of any kind. That is a DIAGNOSTIC, not a shipping mode --
            // it is the experiment that established what the plume plugin has to
            // supply. With nothing supplying the vblank tick, the register
            // window and the ring drain, the guest gets through boot and shader
            // creation and then HANGS before the first draw. Expect that; it is
            // the measurement, not a regression.
            if (std::getenv("DT_PLUME_NOGPU") != nullptr) {
                REXLOG_WARN("DT_PLUME_NOGPU: no GPU plugin at all -- the guest is "
                            "expected to hang before the first draw");
            } else {
                config.gpu_plugin = "plume";
            }
        }

        // DT_NOAUDIO=1 swaps the SDL audio backend for the SDK's no-op one.
        //
        // Why this exists: the guest treats audio-device init failure as FATAL --
        // it calls KeBugCheckEx and the process dies at startup, after the window
        // appears but before anything renders, which reads as a graphics crash.
        // On this machine that happens whenever the display sleeps and takes the
        // HDMI audio endpoint with it, so any unattended or overnight run dies a
        // few minutes in for a reason that has nothing to do with the port.
        //
        // ReXApp::SetupPresentation() assigns the SDL backend and THEN calls
        // OnPreSetup(config_), so overriding here wins without touching the SDK
        // that the other six ports share.
        //
        // Deliberately opt-in, not an automatic fallback: silently substituting a
        // silent audio system would turn "my sound is broken" into an invisible
        // config state. No env => stock SDL audio, unchanged.
        if (std::getenv("DT_NOAUDIO")) {
            config.audio_factory = REX_AUDIO_BACKEND(dt_null_audio::NullAudioSystem);
            REXLOG_WARN("DT_NOAUDIO set -- using the silent null audio backend. "
                        "There will be NO SOUND; this is for unattended runs.");
        }

        // Non-game runtime settings are baked into the binary (not exposed in
        // the .ini, which carries only user-facing game settings): a full
        // content license, and keyboard/mouse input on with the mouse left
        // uncaptured.
        SetCvarInt("license_mask", 1);
        SetCvarBool("mnk_mode", true);
        SetCvarBool("mnk_mouse_capture", false);

        // Daytona relies on CPU readback of render-to-texture resolves for
        // correct rendering (missing/garbled geometry otherwise). The readback
        // cvars are split per backend, and each backend only defines (links) its
        // own pair, so select by the compile-time BACKEND macro (not the OS):
        // a Windows+Vulkan build still defines _WIN32 but links no d3d12_* cvars.
        // NB: the shared `readback_memexport` already defaults true, so the
        // memexport set is belt-and-suspenders; `readback_resolve` defaults to
        // "none" (disabled), so the resolve set is what actually restores parity
        // and was the regression when the D3D12-only sets were _WIN32-guarded.
        // DIAGNOSTIC escape (Deck perf investigation 2026-06-22): DT_NO_READBACK=1
        // skips forcing the per-frame CPU readback (expensive on the Deck APU) to
        // measure its cost + whether daytona still renders correctly without it
        // (post-SDK-fixes). No env => current behavior.
        if (!std::getenv("DT_NO_READBACK")) {
#if REX_HAS_D3D12
        rex::cvar::SetFlagByName("d3d12_readback_memexport", "true");
        rex::cvar::SetFlagByName("d3d12_readback_resolve", "true");
#elif REX_HAS_VULKAN
        rex::cvar::SetFlagByName("vulkan_readback_memexport", "true");
        rex::cvar::SetFlagByName("vulkan_readback_resolve", "true");
#endif
        }

#if REX_HAS_VULKAN
        // Daytona's 3D scenes z-fight badly on the default Vulkan render-target
        // path: the host depth buffer can't reproduce Xenos float24 depth
        // precision, so coplanar surfaces (road markings, guardrails, billboards)
        // flicker. The pixel-shader-interlock (FSI) path emulates EDRAM (including
        // float24 depth) accurately and largely resolves it, so force it on.
        //
        // ...EXCEPT on macOS/MoltenVK, where the FSI/EDRAM path renders the 3D
        // scene dark/underlit AND adds a doubled "top-band" artifact across the
        // top ~15% of the frame (a MoltenVK SPIRV-Cross->MSL execution bug in the
        // EDRAM path; native Vulkan is unaffected). The default host render-target
        // path is bright and clean there, so leave Mac on it. Set DT_FORCE_FSI=1
        // to override and re-test FSI on Mac once that MoltenVK bug is fixed.
#if REX_PLATFORM_MAC
        if (std::getenv("DT_FORCE_FSI")) {
            rex::cvar::SetFlagByName("render_target_path_vulkan", "fsi");
        }
#else
        // DIAGNOSTIC escape (Deck perf investigation 2026-06-22): DT_RT_HOST=1 skips
        // forcing the FSI/EDRAM path (expensive on the Deck) so it uses the cheaper
        // host render-target path (like Mac) -- to measure perf + whether z-fighting
        // is now tolerable/fixed. No env => current behavior (FSI forced).
        if (!std::getenv("DT_RT_HOST")) {
            rex::cvar::SetFlagByName("render_target_path_vulkan", "fsi");
        }
#endif

        // MoltenVK computes implicit texture LOD/gradients inconsistently across a
        // quad's two triangles, drawing diagonal mip-seam lines along quad diagonals
        // (worst on reflective surfaces like the cars' glass). Forcing pixel-shader
        // texture fetches to explicit (non-gradient) LOD removes the seams. Native
        // Vulkan is unaffected, so this is macOS-only.
#if REX_PLATFORM_MAC
        rex::cvar::SetFlagByName("pixel_shader_explicit_lod", "true");
#endif
#endif
    }

    void OnCreateDialogs(rex::ui::ImGuiDrawer* drawer) override {
        if (ini_settings_.debug_overlay) {
            debug_dialog_ = std::make_unique<DaytonaDebugDialog>(drawer);
        }
    }

    void OnPostSetup() override {
        daytona_debug::SetStage(daytona_debug::AppStage::kPostSetup);
        // Keyboard input is handled by the SDK's built-in MnK driver: WASD = left
        // analog stick (steering), arrows = D-pad, Space = A, Shift = B, Return =
        // Start, etc. (all rebindable via the keybind_* cvars). We intentionally do
        // NOT register a game-specific keyboard driver: the old one (removed:
        // src/keyboard_driver.*) was a stale Crazy-Taxi mapping that put WASD AND
        // arrows on the D-pad, and since the InputSystem MERGES every connected
        // driver it double-drove the stick (MnK) and the D-pad (ours) at once.
    }

    void OnPreLaunchModule() override {
        daytona_debug::SetStage(daytona_debug::AppStage::kPreLaunchModule);
        // Replace the SDK's default window title with the port's own version
        // (runs after the SDK set its default, before guest code starts).
        if (auto* wnd = window()) {
#define DAYTONA_STR2(x) #x
#define DAYTONA_STR(x) DAYTONA_STR2(x)
            wnd->SetTitle("Daytona USA v" DAYTONA_STR(DAYTONA_VERSION));
#undef DAYTONA_STR
#undef DAYTONA_STR2
        }
    }

    void OnPostLaunchModule(rex::system::XThread* thread) override {
        (void)thread;
        daytona_debug::SetStage(daytona_debug::AppStage::kPostLaunchModule);
    }

    void OnGuestThreadExit(rex::system::XThread* thread) override {
        (void)thread;
        daytona_debug::SetStage(daytona_debug::AppStage::kGuestThreadExit);
    }

    void OnShutdown() override {
        // spike/d3d9-hook-trace phase 0: emit the D3D9 call profile. Dumps also
        // fire periodically, so a hard kill still leaves usable data in the log.
        dt_d3d9_trace::Dump("shutdown");
        dt_shader_dump::Dump("shutdown");
        dt_plume::Dump("shutdown");
        dt_plume::Shutdown();
        daytona_debug::LogCensus("shutdown");
        debug_dialog_.reset();
    }

private:
    DaytonaIniSettings ini_settings_;
    std::unique_ptr<DaytonaDebugDialog> debug_dialog_;
};

REX_DEFINE_APP(daytona, DaytonaApp::Create)

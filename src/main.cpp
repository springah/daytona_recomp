// Daytona USA (XBLA) - ReXGlue Static Recompilation
// Title ID: 58410B1D

#include "daytona_init.h"
#include "daytona_debug.h"

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/graphics/flags.h>
#include <rex/logging/api.h>
#include <rex/rex_app.h>
#include <rex/input/input_system.h>
#include <rex/system/xthread.h>
#include <rex/ui/flags.h>
#include <imgui.h>

#include <algorithm>
#include <charconv>
#include <cctype>
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
        << "resolution = 1280x720\n"
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

        const uint32_t count = snapshot.hooks[static_cast<size_t>(daytona_debug::HookId::kVdSwapCaller)].count;
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
        paths.config_path = exe_dir / "daytona.toml";
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
        (void)config;
        daytona_debug::SetStage(daytona_debug::AppStage::kPreSetup);
        REXCVAR_SET(gpu_allow_invalid_fetch_constants, true);

        // Daytona relies on CPU readback of render-to-texture resolves for
        // correct rendering (missing/garbled geometry otherwise). The readback
        // cvars are split per backend, and each backend only defines (links) its
        // own pair, so select by the compile-time BACKEND macro (not the OS):
        // a Windows+Vulkan build still defines _WIN32 but links no d3d12_* cvars.
        // NB: the shared `readback_memexport` already defaults true, so the
        // memexport set is belt-and-suspenders; `readback_resolve` defaults to
        // "none" (disabled), so the resolve set is what actually restores parity
        // and was the regression when the D3D12-only sets were _WIN32-guarded.
#if REX_HAS_D3D12
        REXCVAR_SET(d3d12_readback_memexport, true);
        REXCVAR_SET(d3d12_readback_resolve, true);
#elif REX_HAS_VULKAN
        REXCVAR_SET(vulkan_readback_memexport, true);
        REXCVAR_SET(vulkan_readback_resolve, true);
#endif

#if REX_HAS_VULKAN
        // Daytona's 3D scenes z-fight badly on the default Vulkan render-target
        // path: the host depth buffer can't reproduce Xenos float24 depth
        // precision, so coplanar surfaces (road markings, guardrails, billboards)
        // flicker. The pixel-shader-interlock path emulates EDRAM (including
        // float24 depth) accurately and largely resolves it. The SDK auto-falls-
        // back to the default path on GPUs without fragment-shader-interlock, so
        // this is safe to set unconditionally. (Heavier than the default path,
        // but fine for a title this old.)
        REXCVAR_SET(render_target_path_vulkan, "fsi");
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
        debug_dialog_.reset();
    }

private:
    DaytonaIniSettings ini_settings_;
    std::unique_ptr<DaytonaDebugDialog> debug_dialog_;
};

REX_DEFINE_APP(daytona, DaytonaApp::Create)

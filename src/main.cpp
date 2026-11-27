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
#include <rex/ui/imgui_drawer.h>
#include <rex/ui/keybinds.h>
#include <rex/ui/overlay/settings_overlay.h>
#include <imgui.h>
// For the overlay-scale workaround in the pre-draw hook: walking the ImGui
// window list needs the internal header. See the lambda for why.
#include <imgui_internal.h>

#include <atomic>
#include <cstring>

// For the Start+Select -> settings chord: the SDK's input driver is SDL, so
// gamepad button events already flow through SDL's queue; an event watch sees
// them without touching the driver, and a synthetic F4 key event walks the
// same path a keyboard press would.
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_video.h>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <vector>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

// One folder for everything the game writes, for launchers that keep a game's
// data with it (RecompLauncher passes its {data} folder). Empty, the default,
// changes nothing: daytona.ini, logs/ and cache/ stay beside the executable,
// saves in the platform folder, the game in assets/.
REXCVAR_DEFINE_STRING(data_root, "", "Daytona",
                      "Keep settings, saves, cache and logs here, and play the game from its "
                      "game/ folder");

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

// Ini settings whose cvar is owned by the Xenos GPU plugin (vsync,
// resolution_scale, anisotropic_override). That DLL is only loaded for
// --gpu_plugin=xenos, so under the default plume renderer the cvar does not
// exist and a plain SetFlagByName fails without a word -- which is how these
// keys used to be "read" and then dropped. They are deferred instead (applied
// if the plugin registers them), and the ones that ask for something other than
// the shipped default are noted for OnPreSetup to report.
struct GpuPluginIniSetting {
    std::string text;   // "key = value" as the ini has it
    std::string cvar;
};
std::vector<GpuPluginIniSetting> g_ini_gpu_plugin_only;

void SetGpuPluginCvar(std::string_view ini_key, std::string_view ini_value, std::string_view name,
                      std::string_view value, bool is_shipped_default) {
    if (is_shipped_default) {
        SetCvar(name, value);
        return;
    }
    rex::cvar::SetFlagByNameOrDefer(name, value);
    g_ini_gpu_plugin_only.push_back(
        {std::string(ini_key) + " = " + Trim(ini_value), std::string(name)});
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
        << "# Emulated renderer only (--gpu_plugin=xenos); the default native\n"
        << "# renderer ignores it.\n"
        << "vsync = true\n"
        << "# The game runs at a fixed 60Hz and paces its own logic off that, so\n"
        << "# this is ignored: changing it would change GAME SPEED, not smoothness.\n"
        << "# Kept so existing settings files stay valid.\n"
        << "refresh_rate = 60\n"
        << "letterbox = true\n"
        << "safe_area_x = 100\n"
        << "safe_area_y = 100\n"
        << "\n"
        << "[Graphics]\n"
        << "# internal_resolution_scale and anisotropic_filtering apply to the\n"
        << "# emulated renderer only (--gpu_plugin=xenos). The default native\n"
        << "# renderer draws a 1280x720 frame and scales it to the window.\n"
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

// Set while parsing the ini, reported once logging exists. 0 = nothing to say.
double g_ini_refresh_rate_ignored = 0.0;

// Set when --data_root chose the log file (see OnConfigurePaths), so the
// overlay's "Save to config" does not pin this run's file name in the ini.
bool g_log_file_from_data_root = false;

// The next logs/daytona_NNN.log in a folder, named as the SDK names the ones
// it writes beside the executable.
std::filesystem::path NextLogPath(const std::filesystem::path& logs_dir) {
    std::error_code ec;
    std::filesystem::create_directories(logs_dir, ec);
    int last = 0;
    for (const auto& entry : std::filesystem::directory_iterator(logs_dir, ec)) {
        const std::string stem = entry.path().stem().string();
        if (!entry.is_regular_file(ec) || !stem.starts_with("daytona_")) continue;
        if (auto number = ParseInt(std::string_view(stem).substr(8))) last = std::max(last, *number);
    }
    char name[32];
    std::snprintf(name, sizeof(name), "daytona_%03d.log", last + 1);
    return logs_dir / name;
}

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
        if (auto value = ParseBool(it->second)) {
            SetGpuPluginCvar("vsync", it->second, "vsync", *value ? "true" : "false", *value);
        }
    }
    // display.refresh_rate is deliberately NOT wired to video_mode_refresh_rate
    // any more. That cvar is the GUEST's video mode -- a console property -- and
    // the guest paces itself off the vblank derived from it, 1:1. Measured: at
    // 30 the whole game ran at half speed. A player on a 144Hz panel setting
    // refresh_rate = 144, which is exactly what a key of that name in [Display]
    // invites, got a game running 2.4x too fast.
    //
    // The guest tick is fixed at 60Hz in the plume plugin now. An ini key that
    // silently changes game speed is worse than one that does nothing, so this
    // one is accepted and ignored rather than removed -- existing ini files stay
    // valid, and it says so instead of failing.
    if (auto it = values.find("display.refresh_rate"); it != values.end()) {
        if (auto value = ParseDouble(it->second)) {
            // Reported from OnPreSetup, not here: the ini is parsed before
            // logging exists, so a warning at this point goes nowhere. Silently
            // ignoring a setting the user changed on purpose is the same defect
            // as silently substituting a silent audio backend.
            if (*value != 60.0) g_ini_refresh_rate_ignored = *value;
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
            const int32_t scale = ClampInt(*value, 1, 8);
            SetGpuPluginCvar("internal_resolution_scale", it->second, "resolution_scale",
                             std::to_string(scale), scale == 1);
        }
    }
    if (auto it = values.find("graphics.anisotropic_filtering"); it != values.end()) {
        const int32_t level = ParseAnisotropic(it->second);
        SetGpuPluginCvar("anisotropic_filtering", it->second, "anisotropic_override",
                         std::to_string(level), level == -1);
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

// Settings-overlay "Save to config". The SDK's own SaveConfig rewrites the
// whole file as flat `cvar = value` lines, which erased every [Section] of the
// hand-edited ini, and the flat names are not the ones LoadDaytonaIni reads.
// This writes each ini-backed setting back into its own [Section] key (comments
// and everything else in the file kept), and any other changed cvar as a flat
// line above the first section, which the SDK's LoadConfig reads at startup.
struct IniBinding {
    const char* section;   // lower case, as ReadIni keys it
    const char* key;
    const char* cvar;
    const char* cvar2;     // the height of the two WxH settings
};
constexpr IniBinding kIniBindings[] = {
    {"display", "resolution", "window_width", "window_height"},
    {"display", "game_resolution", "video_mode_width", "video_mode_height"},
    {"display", "fullscreen", "fullscreen", nullptr},
    {"display", "vsync", "vsync", nullptr},
    {"display", "letterbox", "present_letterbox", nullptr},
    {"display", "safe_area_x", "present_safe_area_x", nullptr},
    {"display", "safe_area_y", "present_safe_area_y", nullptr},
    {"graphics", "internal_resolution_scale", "resolution_scale", nullptr},
    {"graphics", "anisotropic_filtering", "anisotropic_override", nullptr},
    {"graphics", "upscaler", "present_effect", nullptr},
    {"graphics", "cas_sharpness", "present_cas_additional_sharpness", nullptr},
    {"graphics", "fsr_quality", "present_fsr_quality_mode", nullptr},
    {"graphics", "dither", "present_dither", nullptr},
    {"diagnostics", "log_level", "log_level", nullptr},
    {"diagnostics", "log_max_files", "log_max_files", nullptr},
    {"diagnostics", "log_max_file_size_mb", "log_max_file_size_mb", nullptr},
};

// The ini text for a binding's current cvar value; nullopt when the cvar is not
// registered in this run (a plugin's cvar) or has no usable value, in which
// case that ini line is left exactly as it is.
std::optional<std::string> IniValueFromCvars(const IniBinding& b) {
    if (rex::cvar::GetFlagInfo(b.cvar) == nullptr) return std::nullopt;
    const std::string value = rex::cvar::GetFlagByName(b.cvar);
    if (b.cvar2 != nullptr) {
        if (rex::cvar::GetFlagInfo(b.cvar2) == nullptr) return std::nullopt;
        const std::string second = rex::cvar::GetFlagByName(b.cvar2);
        if (value.empty() || value == "0" || second.empty() || second == "0") return std::nullopt;
        return value + "x" + second;
    }
    if (std::string_view(b.key) == "anisotropic_filtering") {
        static constexpr const char* kNames[] = {"off", "1", "2", "4", "8", "16"};
        const auto level = ParseInt(value);
        if (!level) return std::nullopt;
        if (*level < 0) return std::string("app");
        if (*level > 5) return std::nullopt;
        return std::string(kNames[*level]);
    }
    if (value.empty()) return std::nullopt;
    return value;
}

constexpr const char* kFlatMarker = "# Changed in the in-game settings overlay (read at startup)";

void SaveDaytonaIni(const std::filesystem::path& path) {
    std::vector<std::string> lines;
    bool crlf = false;
    {
        std::ifstream in(path, std::ios::binary);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                crlf = true;
                line.pop_back();
            }
            lines.push_back(std::move(line));
        }
    }

    // "section.key" -> text to write; erased as the file is walked, so what is
    // left at the end is a setting the file did not have yet.
    std::map<std::string, std::string> wanted;
    std::set<std::string> bound_cvars;
    for (const IniBinding& b : kIniBindings) {
        bound_cvars.insert(b.cvar);
        if (b.cvar2 != nullptr) bound_cvars.insert(b.cvar2);
        if (auto value = IniValueFromCvars(b)) {
            wanted[std::string(b.section) + "." + b.key] = std::move(*value);
        }
    }

    std::vector<std::string> out;
    std::string section;
    size_t first_section = std::string::npos;
    bool drop_blank = false;
    for (const std::string& raw : lines) {
        const std::string text = Trim(raw);
        if (text == kFlatMarker) {   // the flat block of an earlier save, regenerated below
            drop_blank = true;
            continue;
        }
        if (drop_blank && text.empty()) {
            drop_blank = false;
            continue;
        }
        const bool in_flat_block = drop_blank;
        drop_blank = false;
        if (text.size() >= 2 && text.front() == '[' && text.back() == ']') {
            section = Lower(Trim(std::string_view(text).substr(1, text.size() - 2)));
            if (first_section == std::string::npos) first_section = out.size();
            out.push_back(raw);
            continue;
        }
        const size_t eq = raw.find('=');
        const bool is_setting = !text.empty() && text.front() != '#' && text.front() != ';' &&
                                eq != std::string::npos;
        if (is_setting && section.empty()) {
            drop_blank = in_flat_block;   // a blank line closing that block goes too
            continue;   // a flat cvar line from an earlier save: regenerated below
        }
        if (is_setting) {
            const std::string id = section + "." + Lower(Trim(raw.substr(0, eq)));
            if (auto it = wanted.find(id); it != wanted.end()) {
                const size_t comment = raw.find_first_of("#;", eq + 1);
                std::string line = raw.substr(0, eq + 1) + " " + it->second;
                if (comment != std::string::npos) line += "  " + raw.substr(comment);
                out.push_back(std::move(line));
                wanted.erase(it);
                continue;
            }
        }
        out.push_back(raw);
    }

    // Every other changed cvar goes above the first section.
    std::vector<std::string> flat;
    for (const std::string& name : rex::cvar::ListModifiedFlags()) {
        if (bound_cvars.count(name) != 0) continue;
        // Where this run keeps its files comes from the command line every time.
        if (name == "data_root" || (name == "log_file" && g_log_file_from_data_root)) continue;
        flat.push_back(name + " = " + rex::cvar::GetFlagByName(name));
    }
    if (!flat.empty()) {
        flat.insert(flat.begin(), kFlatMarker);
        flat.push_back("");
    }
    out.insert(out.begin() + (first_section == std::string::npos ? out.size() : first_section),
               flat.begin(), flat.end());

    // Settings the file lacked: added only if they are not at their default.
    for (const auto& [id, value] : wanted) {
        const size_t dot = id.find('.');
        const std::string sec = id.substr(0, dot);
        const std::string key = id.substr(dot + 1);
        bool changed = false;
        for (const IniBinding& b : kIniBindings) {
            if (sec == b.section && key == b.key && rex::cvar::HasNonDefaultValue(b.cvar)) {
                changed = true;
            }
        }
        if (!changed) continue;
        out.push_back("");
        out.push_back("[" + sec + "]");
        out.push_back(key + " = " + value);
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        REXLOG_ERROR("daytona.ini: could not write {}", path.string());
        return;
    }
    for (const std::string& line : out) file << line << (crlf ? "\r\n" : "\n");
    REXLOG_WARN("daytona.ini: settings saved to {}", path.string());
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

// The settings overlay plus a "Quit game" button (GitHub issue #5: fullscreen
// had no obvious way out but Alt+F4). A small second window pinned to the top
// right, with a confirm step, because the overlay is opened mid-race too. A
// subclass, so the SDK's dialog is untouched.
class DaytonaSettingsDialog : public rex::ui::SettingsDialog {
public:
    DaytonaSettingsDialog(rex::ui::ImGuiDrawer* drawer, std::filesystem::path config_path,
                          std::function<void()> quit)
        : SettingsDialog(drawer, std::move(config_path)), quit_(std::move(quit)) {}

protected:
    void OnDraw(ImGuiIO& io) override {
        SettingsDialog::OnDraw(io);

        const float scale = float(rex::cvar::Query<double>("overlay_scale"));
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 8.0f * scale,
                                       viewport->WorkPos.y + 8.0f * scale),
                                ImGuiCond_Always, ImVec2(1.0f, 0.0f));
        ImGui::SetNextWindowBgAlpha(0.85f);
        if (!ImGui::Begin("Quit##daytona", nullptr,
                          ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                              ImGuiWindowFlags_AlwaysAutoResize |
                              ImGuiWindowFlags_NoSavedSettings)) {
            ImGui::End();
            return;
        }
        ImGui::SetWindowFontScale(scale);
        if (!confirm_) {
            if (ImGui::Button("Quit game")) confirm_ = true;
            ImGui::TextDisabled("Pad: hold Start+Back for 3 s");
        } else {
            ImGui::TextUnformatted("Quit Daytona USA?");
            if (ImGui::Button("Quit")) quit_();
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) confirm_ = false;
        }
        ImGui::End();
    }

private:
    std::function<void()> quit_;
    bool confirm_ = false;
};

}  // namespace

class DaytonaApp : public rex::ReXApp {
public:
    using rex::ReXApp::ReXApp;

    static std::unique_ptr<rex::ui::WindowedApp> Create(rex::ui::WindowedAppContext& ctx) {
        return std::unique_ptr<DaytonaApp>(new DaytonaApp(ctx, "daytona", PPCImageConfig));
    }

    ~DaytonaApp() override {
        // Detach from the renderer BEFORE the overlay members destruct: it
        // holds raw pointers to them (the drawer, and `this` inside the
        // pre-draw hook), and its setters serialize against a present in
        // flight, so after these two calls nothing can touch them again.
        dt_plume::SetOverlayPreDraw(nullptr);
        dt_plume::SetImGuiDrawer(nullptr);
        rex::ui::UnregisterBind("bind_settings");
    }

protected:
    void OnConfigurePaths(rex::PathConfig& paths) override {
        const std::filesystem::path exe_dir = rex::filesystem::GetExecutableFolder();
        // --data_root moves everything below from beside the executable into
        // that folder: daytona.ini, cache/, logs/, the saves (saves/) and the
        // game (game/). Explicit --game_data_root, --user_data_root and
        // --log_file still win. Without it, nothing here changes.
        const std::filesystem::path data_root = rex::to_path(REXCVAR_GET(data_root));
        const std::filesystem::path base_dir = data_root.empty() ? exe_dir : data_root;
        if (!data_root.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(data_root, ec);
            paths.config_path = data_root / "daytona.ini";
            if (rex::cvar::GetFlagByName("user_data_root").empty()) {
                paths.user_data_root = data_root / "saves";
            }
            if (rex::cvar::GetFlagByName("log_file").empty()) {
                // The SDK opens log_file by its narrow name, as it does the
                // ones it names itself; a folder the code page cannot spell
                // keeps the logs beside the executable instead.
                try {
                    SetCvar("log_file", NextLogPath(data_root / "logs").string());
                    g_log_file_from_data_root = true;
                } catch (const std::exception&) {
                }
            }
        }
        // Leave config_path at the SDK default (<exe_dir>/daytona.ini) so the
        // "Save to config" overlay and any auto-loaded cvars use .ini, matching
        // the sibling recomps (futari/gh/guwange/sdoj/sotn) -- no .toml.
        // Remembered for the settings dialog, which is app-owned here (see the
        // overlay block): its "Save to config" writes the same file.
        overlay_config_path_ = paths.config_path.empty() ? base_dir / "daytona.ini"
                                                         : paths.config_path;
        paths.cache_root = base_dir / "cache";
        ini_settings_ = LoadDaytonaIni(base_dir, paths);
        // Default the game data root to an "assets" folder next to the
        // executable when no --game_data_root was provided, so the game runs on
        // a plain double-click (no launcher .bat needed), matching the sibling
        // recomps (futari/gh/guwange/sotn). An explicit --game_data_root (cvar
        // non-empty) still takes precedence.
        if (paths.game_data_root.empty()) {
            paths.game_data_root = data_root.empty() ? exe_dir / "assets" : data_root / "game";
        }
    }

    void OnPreSetup(rex::RuntimeConfig& config) override {
        daytona_debug::SetStage(daytona_debug::AppStage::kPreSetup);

        // The first point at which the log exists. See LoadDaytonaIni.
        if (g_ini_refresh_rate_ignored != 0.0) {
            REXLOG_WARN("daytona.ini: refresh_rate = {} is IGNORED. The guest runs at a fixed 60Hz "
                        "and paces its own logic off that tick, so honouring this would change "
                        "GAME SPEED rather than smoothness -- at 30 the whole game runs at half "
                        "speed. Your window is unaffected.", g_ini_refresh_rate_ignored);
        }
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

        if (config.gpu_plugin == "plume") {
            for (const GpuPluginIniSetting& setting : g_ini_gpu_plugin_only) {
                REXLOG_WARN("daytona.ini: {} has no effect on the native renderer (it belongs "
                            "to the Xenos GPU plugin, --gpu_plugin=xenos only)", setting.text);
            }
        }

        // An explicit --gpu_plugin=xenos means "render through Xenos emulation".
        // Since the native renderer became the default, that flag ALONE ran both
        // renderers against the same window, and the process died silently ~4s
        // after launch with nothing in the log (measured 2026-08-09; the old
        // "no-rebuild A/B" recipe predates the default flip). The referee
        // configuration is xenos + no dt_plume, so choosing xenos now implies
        // it -- loudly, because silent config states are how this port gets
        // unfixable bug reports.
        if (config.gpu_plugin == "xenos") {
            const char* p = std::getenv("DT_PLUME");
            if (p != nullptr && std::atoi(p) != 0) {
                REXLOG_ERROR("--gpu_plugin=xenos with DT_PLUME=1 forced: BOTH renderers will "
                             "fight over one window. This combination is known to die silently "
                             "seconds after launch. Unset DT_PLUME unless you are debugging "
                             "exactly this conflict.");
            } else {
                dt_plume::ForceDisable();
                REXLOG_WARN("--gpu_plugin=xenos: rendering through Xenos emulation; the native "
                            "renderer is OFF for this run. (This is the referee configuration; "
                            "no DT_PLUME=0 needed.)");
            }
        }

        // Audio backend.
        //
        // The guest treats audio-device init failure as FATAL -- it calls
        // KeBugCheckEx and the process dies at startup, after the window appears
        // but before anything renders, which reads as a graphics crash. The
        // stock SDL backend reports exactly that failure when no default
        // playback device can be opened, and that is not an exotic state: a
        // machine whose display has slept has usually taken its HDMI audio
        // endpoint with it, and a player who unplugged their headphones has no
        // default device either. Either way the game died at the title.
        //
        // So the default is ResilientAudioSystem: SDL when a device opens, the
        // silent driver when it does not, and an ERROR line that says so. The
        // old objection to falling back -- that a silent substitution turns "my
        // sound is broken" into an invisible config state -- is right, and it
        // argues for being loud about it, not for exiting.
        //
        // DT_NOAUDIO=1 still forces silence outright, and remains what unattended
        // runs should use: it never opens a device at all, so it cannot be
        // affected by one disappearing mid-run.
        //
        // ReXApp::SetupPresentation() assigns the SDL backend and THEN calls
        // OnPreSetup(config_), so overriding here wins without touching the SDK
        // that the other six ports share.
        if (std::getenv("DT_NOAUDIO")) {
            config.audio_factory = REX_AUDIO_BACKEND(dt_null_audio::NullAudioSystem);
            REXLOG_WARN("DT_NOAUDIO set -- using the silent null audio backend. "
                        "There will be NO SOUND; this is for unattended runs.");
        } else {
            config.audio_factory = REX_AUDIO_BACKEND(dt_null_audio::ResilientAudioSystem);
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

    bool OnSaveSettings() override {
        SaveDaytonaIni(overlay_config_path_);
        return true;
    }

    // Quit the way the title-bar close button does (OnClosing). Callable from
    // any thread: the overlay button runs on the render thread, the pad hold
    // and DT_QUIT_AFTER on SDL's timer thread.
    void RequestQuit(const char* source) {
        REXLOG_WARN("Quit requested ({})", source);
        app_context().CallInUIThread([this] {
            if (rex::ui::Window* w = window()) w->RequestClose();
        });
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

        // One window, not two. The native renderer owns its own swapchain and
        // used to create its own window for it, so the port put a second window
        // on screen beside this one -- and this is the one that matters: input
        // is attached to it, F11 belongs to it, and closing it shuts the game
        // down. Hand the renderer this window's native handle and it presents
        // into it instead.
        //
        // Here rather than in OnPreSetup because the window does not exist until
        // SetupPresentation has run, and before the first frame because the
        // swapchain is built from whatever handle the renderer has by then.
        if (rex::ui::Window* w = window()) {
            dt_plume::SetHostWindow(w->GetNativeWindowHandle());
        }

        // The GPU plugin is loaded by now, so its cvars exist if it has them.
        // Under plume OnPreSetup has already said these keys do nothing.
        for (const GpuPluginIniSetting& setting : g_ini_gpu_plugin_only) {
            if (rex::cvar::GetFlagInfo(setting.cvar) != nullptr) {
                REXLOG_WARN("daytona.ini: {} applied ({} = {})", setting.text, setting.cvar,
                            rex::cvar::GetFlagByName(setting.cvar));
            }
        }

        // DT_SAVE_INI=1: do what the overlay "Save to config" button does, once,
        // at boot. Lets an unattended run prove the save round trip against the
        // real cvars without a click (see DT_OVERLAY_OPEN below).
        if (const char* save = std::getenv("DT_SAVE_INI"); save && save[0] == '1') {
            SaveDaytonaIni(overlay_config_path_);
        }

        // The overlay. The SDK stands up its ImGui stack in exactly two modes:
        // presenter mode (a graphics provider draws it) and fully-headless
        // detached mode (config.graphics == nullptr). Under plume there is a
        // THIRD state -- a GPU plugin exists for the vblank/ring bookkeeping
        // but declines presentation -- and in that state the SDK creates
        // NEITHER the drawer NOR the F4 bind. That is why F4 "did nothing":
        // there was no bind registered at all. So the app owns the whole
        // stack: its own ImGuiDrawer + the renderer's ImmediateDrawer, its own
        // settings dialog, its own binds; dt_plume composites it every present.
        if (imgui_drawer() == nullptr) {
            overlay_immediate_ = dt_plume::CreateImmediateDrawer();
            overlay_imgui_ = std::make_unique<rex::ui::ImGuiDrawer>(window(), 64);
            overlay_imgui_->SetImmediateDrawer(overlay_immediate_.get());
            dt_plume::SetImGuiDrawer(overlay_imgui_.get());
            // The dialog list is mutated on the RENDER thread (pre-draw hook),
            // never in the key callback: AddDialog/RemoveDialog from the UI
            // thread would race the Draw that iterates the list. The bind just
            // counts requests; parity decides open vs close, so a double-press
            // between two presents nets out instead of desyncing.
            dt_plume::SetOverlayPreDraw([this] {
                // SDK bug workaround: SettingsDialog applies overlay_scale via
                // SetWindowFontScale on its OUTER window only -- the BeginChild
                // panes (category tree, settings list) never get it, so their
                // layout scales but their text stays tiny. FontWindowScale
                // persists per ImGui window, and the drawer's context is
                // current on this thread from the previous Draw, so stamping
                // the dialog's child windows here fixes the panes without
                // touching the pinned SDK. One frame of pop-in on first open.
                if (overlay_settings_ != nullptr) {
                    if (ImGuiContext* ctx = ImGui::GetCurrentContext()) {
                        const float scale = float(rex::cvar::Query<double>("overlay_scale"));
                        for (ImGuiWindow* w : ctx->Windows) {
                            if (w->Name != nullptr &&
                                std::strncmp(w->Name, "Settings##rex/", 14) == 0) {
                                w->FontWindowScale = scale;
                            }
                        }
                    }
                }
                const int toggles = overlay_toggle_requests_.exchange(0);
                if ((toggles & 1) == 0) return;
                if (overlay_settings_) {
                    overlay_settings_.reset();
                } else {
                    overlay_settings_ = std::make_unique<DaytonaSettingsDialog>(
                        overlay_imgui_.get(), overlay_config_path_,
                        [this] { RequestQuit("overlay button"); });
                    overlay_settings_->SetSaveHandler(
                        [path = overlay_config_path_] { SaveDaytonaIni(path); });
                }
            });
            rex::ui::RegisterBind("bind_settings", "F4", "Toggle settings overlay",
                                  [this] { overlay_toggle_requests_.fetch_add(1); });
            // DT_OVERLAY_OPEN=1: open the settings dialog at boot. Unattended
            // drills verify the overlay from a screenshot without injecting a
            // keypress -- focus-stealing is exactly what modern Windows (and
            // this project's own input-injection rules) refuse.
            if (const char* open = std::getenv("DT_OVERLAY_OPEN"); open && open[0] == '1') {
                overlay_toggle_requests_.fetch_add(1);
            }
        }

        // Start+Select opens settings on a pad, because a Deck in Game Mode
        // has no F4. Gamepad button events are already in SDL's queue (the
        // SDK's input driver is SDL); the watch synthesizes an F4 key press
        // down the normal event path, so the SDK's own keybind does the rest.
        // An event watch, not a hook: watches see events without eating them,
        // so the guest still gets Start and Select as normal buttons.
        //
        // Holding the chord for kQuitHoldMs quits (GitHub issue #5): the
        // overlay has no gamepad navigation, so a pad cannot press its "Quit
        // game" button. A one-shot SDL timer measures the hold, because no
        // event arrives while two buttons are simply held; releasing either
        // button (or losing the pad) cancels it.
        SDL_AddEventWatch(
            [](void* userdata, SDL_Event* e) -> bool {
                static constexpr Uint32 kQuitHoldMs = 3000;
                static bool start_down = false;
                static bool back_down = false;
                static bool fired = false;
                static SDL_TimerID quit_timer = 0;
                if (e->type == SDL_EVENT_GAMEPAD_REMOVED) {
                    start_down = back_down = fired = false;
                    if (quit_timer != 0) {
                        SDL_RemoveTimer(quit_timer);
                        quit_timer = 0;
                    }
                    return true;
                }
                if (e->type != SDL_EVENT_GAMEPAD_BUTTON_DOWN &&
                    e->type != SDL_EVENT_GAMEPAD_BUTTON_UP) {
                    return true;
                }
                const bool down = (e->type == SDL_EVENT_GAMEPAD_BUTTON_DOWN);
                if (e->gbutton.button == SDL_GAMEPAD_BUTTON_START) start_down = down;
                if (e->gbutton.button == SDL_GAMEPAD_BUTTON_BACK) back_down = down;
                if (start_down && back_down) {
                    if (!fired) {
                        fired = true;   // edge-trigger: once per chord press
                        SDL_Event key = {};
                        key.key.scancode = SDL_SCANCODE_F4;
                        key.key.key = SDLK_F4;
                        if (SDL_Window* const* w = SDL_GetWindows(nullptr)) {
                            if (w[0] != nullptr) key.key.windowID = SDL_GetWindowID(w[0]);
                        }
                        key.type = SDL_EVENT_KEY_DOWN;
                        key.key.down = true;
                        SDL_PushEvent(&key);
                        key.type = SDL_EVENT_KEY_UP;
                        key.key.down = false;
                        SDL_PushEvent(&key);
                        quit_timer = SDL_AddTimer(
                            kQuitHoldMs,
                            [](void* app, SDL_TimerID, Uint32) -> Uint32 {
                                static_cast<DaytonaApp*>(app)->RequestQuit("Start+Back held");
                                return 0;   // one shot
                            },
                            userdata);
                    }
                } else {
                    fired = false;
                    if (quit_timer != 0) {
                        SDL_RemoveTimer(quit_timer);
                        quit_timer = 0;
                    }
                }
                return true;
            },
            this);

        // DT_QUIT_AFTER=<seconds>: quit through RequestQuit after that long.
        // An unattended run ends by the real quit path instead of being killed,
        // which is also how that path is checked without a click or a pad.
        if (const char* after = std::getenv("DT_QUIT_AFTER")) {
            if (const int seconds = std::atoi(after); seconds > 0) {
                SDL_AddTimer(
                    static_cast<Uint32>(seconds) * 1000u,
                    [](void* app, SDL_TimerID, Uint32) -> Uint32 {
                        static_cast<DaytonaApp*>(app)->RequestQuit("DT_QUIT_AFTER");
                        return 0;
                    },
                    this);
            }
        }
    }

    std::unique_ptr<rex::ui::ImmediateDrawer> OnCreateImmediateDrawer() override {
        // Detached-overlay mode: the plume plugin declines presentation, so the
        // SDK asks the app for a renderer for its own UI. See dt_plume's
        // overlay pass.
        return dt_plume::CreateImmediateDrawer();
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

    // The app-owned overlay stack (see the block in the setup hook). Member
    // order IS teardown order in reverse: the dialog goes before the drawer it
    // registered with, the drawer before the window it listens to (a base
    // member, so destroyed after all of these).
    std::unique_ptr<rex::ui::ImmediateDrawer> overlay_immediate_;
    std::unique_ptr<rex::ui::ImGuiDrawer> overlay_imgui_;
    std::unique_ptr<rex::ui::SettingsDialog> overlay_settings_;
    std::atomic<int> overlay_toggle_requests_{0};
    std::filesystem::path overlay_config_path_;
    std::unique_ptr<DaytonaDebugDialog> debug_dialog_;
};

REX_DEFINE_APP(daytona, DaytonaApp::Create)

#include "dt_shader_dump.h"

#include <rex/logging.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_set>

namespace dt_shader_dump {
namespace {

// A container is a header (9 dwords) plus tables plus microcode. Anything
// outside this range means we misread r3 -- refuse rather than write garbage.
constexpr uint32_t kMinBlob = 64;
constexpr uint32_t kMaxBlob = 1u << 20;   // 1 MB

std::mutex g_mutex;
std::unordered_set<uint64_t> g_seen;
uint32_t g_vs_unique = 0;
uint32_t g_ps_unique = 0;
std::atomic<uint32_t> g_vs_calls{0};
std::atomic<uint32_t> g_ps_calls{0};
std::atomic<uint32_t> g_rejected{0};
bool g_dir_ready = false;

uint64_t Fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

}  // namespace

bool Enabled() {
    static const bool on = std::getenv("DT_DUMP_SHADERS") != nullptr;
    return on;
}

void Record(bool is_vertex,
            uint32_t function_ea,
            uint32_t virtual_size,
            uint32_t physical_size,
            const uint8_t* raw) {
    (is_vertex ? g_vs_calls : g_ps_calls).fetch_add(1, std::memory_order_relaxed);
    if (!Enabled() || raw == nullptr) return;

    const uint64_t total64 = static_cast<uint64_t>(virtual_size) + physical_size;
    if (total64 < kMinBlob || total64 > kMaxBlob) {
        // Log the first few so a wrong register shows up immediately instead of
        // silently producing an empty shader set.
        if (g_rejected.fetch_add(1, std::memory_order_relaxed) < 8) {
            REXLOG_WARN("shader dump REJECTED {} ea={:08X} virtual={:08X} physical={:08X} -- "
                        "implausible size, is r3 really the container pointer?",
                        is_vertex ? "VS" : "PS", function_ea, virtual_size, physical_size);
        }
        return;
    }
    const size_t total = static_cast<size_t>(total64);
    const uint64_t hash = Fnv1a(raw, total);

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_seen.insert(hash).second) return;   // already dumped this exact blob

    if (!g_dir_ready) {
        std::error_code ec;
        std::filesystem::create_directories("shaders", ec);
        g_dir_ready = true;
        REXLOG_WARN("DT_DUMP_SHADERS set -- writing shader containers to ./shaders/");
    }

    char name[64];
    std::snprintf(name, sizeof(name), "shaders/%s_%016llx.bin",
                  is_vertex ? "vs" : "ps", static_cast<unsigned long long>(hash));
    std::ofstream out(name, std::ios::binary);
    if (!out) {
        REXLOG_WARN("shader dump: could not open {}", name);
        return;
    }
    out.write(reinterpret_cast<const char*>(raw), static_cast<std::streamsize>(total));

    (is_vertex ? g_vs_unique : g_ps_unique) += 1;
    REXLOG_WARN("shader dump {} #{} ea={:08X} virtual={} physical={} total={} -> {}",
                is_vertex ? "VS" : "PS", is_vertex ? g_vs_unique : g_ps_unique,
                function_ea, virtual_size, physical_size, total, name);
}

void Dump(const char* reason) {
    std::lock_guard<std::mutex> lock(g_mutex);
    REXLOG_WARN("===== shader inventory ({}) =====", reason);
    REXLOG_WARN("   CreateVertexShader calls={}  unique={}",
                g_vs_calls.load(std::memory_order_relaxed), g_vs_unique);
    REXLOG_WARN("   CreatePixelShader  calls={}  unique={}",
                g_ps_calls.load(std::memory_order_relaxed), g_ps_unique);
    if (!Enabled()) {
        REXLOG_WARN("   (DT_DUMP_SHADERS not set -- counted only, nothing written)");
    }
    const uint32_t rej = g_rejected.load(std::memory_order_relaxed);
    if (rej) REXLOG_WARN("   REJECTED {} blobs on implausible size", rej);
    REXLOG_WARN("===== end shader inventory =====");
}

}  // namespace dt_shader_dump

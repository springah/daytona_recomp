#include "dt_shader_patchtest.h"

#include <rex/logging.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace dt_shader_patchtest {
namespace {

struct Snapshot {
    uint64_t hash = 0;
    bool is_vertex = false;
    uint32_t container_ea = 0;
    uint32_t object_ea = 0;
    std::vector<uint8_t> code;   // the container's physical region = the microcode
    uint32_t live_ea = 0;        // where the microcode was found to actually live
    bool located = false;
    bool detailed = false;
};

std::mutex g_mutex;
std::vector<Snapshot> g_snaps;
uint32_t g_checks = 0;

// Guest pointers seen so far in this title sit in 0x8xxxxxxx (image) and
// 0xA0000000-0xCFFFFFFF (heap). Anything else is not worth chasing.
bool PlausiblePointer(uint32_t v) {
    return (v >= 0x82000000u && v < 0x83500000u) || (v >= 0xA0000000u && v < 0xD0000000u);
}

uint32_t LoadBE32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3];
}

}  // namespace

bool Enabled() {
    static const bool on = std::getenv("DT_PATCHTEST") != nullptr;
    return on;
}

void Remember(uint64_t hash, bool is_vertex, uint32_t container_ea,
              uint32_t virtual_size, uint32_t physical_size, const uint8_t* raw) {
    if (!Enabled() || raw == nullptr || physical_size == 0) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const auto& s : g_snaps) {
        if (s.hash == hash) return;
    }
    Snapshot s;
    s.hash = hash;
    s.is_vertex = is_vertex;
    s.container_ea = container_ea;
    // The microcode is the container's physical region: container + virtualSize,
    // physicalSize bytes (XenosRecomp reads code at virtualSize + physicalOffset).
    s.code.assign(raw + virtual_size, raw + virtual_size + physical_size);
    g_snaps.push_back(std::move(s));
}

void RememberObject(uint64_t hash, uint32_t object_ea) {
    if (!Enabled()) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    for (auto& s : g_snaps) {
        if (s.hash == hash) {
            s.object_ea = object_ea;
            REXLOG_WARN("patchtest: {} {:016X} container@{:08X} object@{:08X} code {} bytes",
                        s.is_vertex ? "VS" : "PS", hash, s.container_ea, object_ea, s.code.size());
            return;
        }
    }
}

void Check(const char* reason, const uint8_t* (*resolve)(uint32_t ea, void* ctx), void* ctx) {
    if (!Enabled()) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    ++g_checks;
    REXLOG_WARN("===== patchtest ({}) check #{} over {} snapshots =====",
                reason, g_checks, g_snaps.size());

    for (auto& s : g_snaps) {
        // The container address is a REUSED SCRATCH BUFFER -- three shaders came
        // back with the same one -- so the live microcode has to be found via the
        // shader object instead. Walk the object's leading dwords, treat each
        // plausible pointer as a candidate, and confirm by matching the first
        // 32 bytes of the code we captured at creation.
        if (!s.located) {
            const uint8_t* obj = s.object_ea ? resolve(s.object_ea, ctx) : nullptr;
            if (obj == nullptr) {
                REXLOG_WARN("   {} {:016X}: no shader object address", s.is_vertex ? "VS" : "PS", s.hash);
                continue;
            }
            const size_t probe = s.code.size() < 32 ? s.code.size() : 32;
            for (uint32_t i = 0; i < 32 && !s.located; ++i) {
                const uint32_t cand = LoadBE32(obj + i * 4);
                if (!PlausiblePointer(cand)) continue;
                const uint8_t* p = resolve(cand, ctx);
                if (p != nullptr && std::memcmp(p, s.code.data(), probe) == 0) {
                    s.live_ea = cand;
                    s.located = true;
                    REXLOG_WARN("   {} {:016X}: microcode located at {:08X} (object+{})",
                                s.is_vertex ? "VS" : "PS", s.hash, cand, i * 4);
                }
            }
            if (!s.located) {
                REXLOG_WARN("   {} {:016X}: microcode NOT found via object@{:08X} -- "
                            "no leading pointer matches the captured code",
                            s.is_vertex ? "VS" : "PS", s.hash, s.object_ea);
                continue;
            }
        }

        const uint8_t* now = resolve(s.live_ea, ctx);
        if (now == nullptr) continue;
        const size_t n = s.code.size() / 4;
        size_t changed = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&s.code[i * 4], now + i * 4, 4) != 0) ++changed;
        }
        if (changed == 0) {
            REXLOG_WARN("   {} {:016X} @{:08X}: UNCHANGED ({} dwords) -- NOT patched",
                        s.is_vertex ? "VS" : "PS", s.hash, s.live_ea, n);
            continue;
        }
        REXLOG_WARN("   {} {:016X} @{:08X}: {} of {} dwords CHANGED -- PATCHED",
                    s.is_vertex ? "VS" : "PS", s.hash, s.live_ea, changed, n);
        if (s.detailed) continue;
        s.detailed = true;
        for (size_t i = 0; i < n; ++i) {
            uint32_t before = LoadBE32(&s.code[i * 4]);
            uint32_t after = LoadBE32(now + i * 4);
            if (before == after) continue;
            REXLOG_WARN("      dword {:<4} {:08X} -> {:08X}   [insn {:<3} word {}]",
                        i, before, after, i / 3, i % 3);
        }
    }
    REXLOG_WARN("===== end patchtest =====");
}

}  // namespace dt_shader_patchtest

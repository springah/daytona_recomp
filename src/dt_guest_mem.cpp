#include "dt_guest_mem.h"

#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xmemory.h>

namespace dt_guest_mem {

bool Readable(uint32_t ea, uint32_t size) {
    if (ea == 0 || size == 0) return false;
    // A range that wraps would pass a naive high-address check.
    if (size - 1 > UINT32_MAX - ea) return false;

    auto* memory = REX_KERNEL_MEMORY();
    if (memory == nullptr) return false;

    // LookupHeap is the reserved-vs-mapped question; QueryRangeAccess is the
    // committed-and-readable one. Both are needed -- an address inside a heap's
    // range can still sit on an uncommitted page.
    auto* heap = memory->LookupHeap(ea);
    if (heap == nullptr) return false;

    const auto access = heap->QueryRangeAccess(ea, ea + size - 1);
    return (static_cast<uint32_t>(access) &
            static_cast<uint32_t>(rex::memory::PageAccess::kReadOnly)) != 0;
}

uint32_t PhysicalToVirtual(uint32_t phys, uint32_t size) {
    // Physical memory is 512 MB; anything above that is not a physical address
    // and must not be silently masked into one.
    if (phys >= 0x20000000u || size == 0) return 0;

    // Ordered by how the 360 lays them out: 64 KB pages, 16 MB pages, uncached,
    // then the cached title window.
    static constexpr uint32_t kAliases[] = {0xA0000000u, 0xC0000000u, 0xE0000000u, 0x80000000u};
    for (uint32_t alias : kAliases) {
        const uint32_t va = alias | phys;
        if (!Readable(va, size)) continue;
        static uint32_t reported = 0;
        if (reported != alias) {
            reported = alias;
            REXLOG_WARN("dt_guest_mem: physical {:08X} reads through the {:08X} alias "
                        "({} bytes) -- this is where Daytona's GPU-visible memory lives",
                        phys, alias, size);
        }
        return va;
    }
    return 0;
}

const uint8_t* Resolve(const uint8_t* base, uint32_t ea, uint32_t size) {
    if (base == nullptr) return nullptr;
    if (!Readable(ea, size)) return nullptr;
    // Matches REX_RAW_ADDR / REX_PHYS_HOST_OFFSET: the physical window is offset
    // by 0x1000 in the host mapping on Windows and macOS.
#if defined(_WIN32) || defined(__APPLE__)
    const uint32_t phys_offset = (ea >= 0xE0000000u) ? 0x1000u : 0u;
#else
    const uint32_t phys_offset = 0u;
#endif
    return base + ea + phys_offset;
}

}  // namespace dt_guest_mem

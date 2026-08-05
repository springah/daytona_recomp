// Daytona USA -- guarded guest-memory reads for the native-renderer spike.
//
// WHY THIS EXISTS. The renderer work reads guest structures whose layout is
// still being reverse-engineered: the vertex fetch constants, the live shader
// microcode buffer, the vertex data a draw sources. When an offset is wrong the
// address that comes out is not merely useless, it is arbitrary -- and the guest
// address space is a 4 GB RESERVATION, mostly uncommitted, so dereferencing it
// takes the whole process down with no message.
//
// That is exactly what happened: SnapshotDrawVertices read a fetch constant at
// device+1152+fc*8, got a nonzero garbage BaseAddress, and segfaulted. The code
// already guarded the BaseAddress==0 case; it had no answer for garbage.
//
// The rule this encodes: a DIAGNOSTIC MUST NEVER BE ABLE TO KILL THE RUN. A
// wrong offset has to produce a logged miss that names the bad address, because
// that log is the thing that tells you the offset is wrong. A crash tells you
// nothing and costs a race run to reproduce.

#pragma once

#include <cstdint>

namespace dt_guest_mem {

// Is the whole range [ea, ea+size) committed and readable guest memory?
//
// Asks the SDK's memory system (heap lookup + page-protection query) rather
// than probing, so it is honest about reserved-but-uncommitted pages, which is
// precisely the case that faults. Returns false for ea==0, size==0 and for a
// range that would wrap the 32-bit address space.
bool Readable(uint32_t ea, uint32_t size);

// Guest ea -> host pointer, mirroring REX_RAW_ADDR (which needs `base` in scope
// and so cannot be used from a free function), but nullptr unless the caller's
// whole intended read is readable.
//
// `size` is the number of bytes the caller is about to touch, not a hint: pass
// what will actually be read. A 4-byte check in front of a 2 KB scan is barely
// better than no check at all.
const uint8_t* Resolve(const uint8_t* base, uint32_t ea, uint32_t size);

}  // namespace dt_guest_mem

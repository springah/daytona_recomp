// Daytona USA -- does the guest D3D9 runtime PATCH shader microcode?
//
// Every vfetch instruction in every vertex shader reads stride=0, offset=0,
// format=0 at CreateVertexShader time (third instruction dword is literally
// 0x00000000; verified against the raw dwords, so not a decode bug). Two
// explanations, and they lead to completely different renderer designs:
//
//   (a) The 360 D3D runtime patches the vertex declaration INTO the vfetch
//       instructions at bind/draw time, and what we dumped is pre-patch. Then
//       the vertex layout is a static per-shader property, recoverable once.
//
//   (b) It does not, and the layout lives somewhere else entirely.
//
// It cannot be in the fetch constants: GPUVERTEX_FETCH_CONSTANT is 2 dwords of
// {Type, BaseAddress} {Endian, Size, ...} -- no stride, offset or format
// (d3d9gpu.h), and D3DDevice_SetVertexFetchConstant has nowhere to put one.
//
// The test: keep the container bytes at creation, re-read the same guest memory
// after the game has been drawing, and diff. Isolated dwords changing at vfetch
// positions proves (a). Nothing changing refutes it. Wholesale change means the
// allocation was reused and the test is inconclusive -- which is reported as
// such rather than being read either way.
//
// Gated behind DT_PATCHTEST.

#pragma once

#include <cstddef>
#include <cstdint>

namespace dt_shader_patchtest {

bool Enabled();

// Snapshot the container's microcode at CreateVertexShader/CreatePixelShader.
void Remember(uint64_t hash, bool is_vertex, uint32_t container_ea,
              uint32_t virtual_size, uint32_t physical_size, const uint8_t* raw);

// The object the create call returned. The container address is a reused scratch
// buffer (three shaders came back with the same one), so the live microcode has
// to be reached through the object, not through the container.
void RememberObject(uint64_t hash, uint32_t object_ea);

// Re-read and diff. `read` fetches one guest byte (the caller owns the base
// pointer and the load macros).
void Check(const char* reason, const uint8_t* (*resolve)(uint32_t ea, void* ctx), void* ctx);

}  // namespace dt_shader_patchtest

// Daytona USA -- shader microcode dump
//
// D3DDevice_CreateVertexShader / _CreatePixelShader are the ONLY places the
// Xbox 360 shader container is visible: both take a single argument in r3, a
// pointer to the container (there is no device parameter -- see XDK 20500
// d3d9.h:2268). Everything downstream holds an opaque D3DVertexShader*.
//
// Gated behind DT_DUMP_SHADERS so a normal run never writes anything.
// Blobs land in <cwd>/shaders/{vs,ps}_<hash>.bin and feed XenosRecomp.

#pragma once

#include <cstdint>

namespace dt_shader_dump {

// True when DT_DUMP_SHADERS is set. Cheap; safe to call per shader creation.
bool Enabled();

// virtual_size/physical_size come from the container header (+4 and +8);
// the blob is virtual_size + physical_size bytes starting at `raw`.
//
// Returns the FNV-1a of the container, or 0 if it was rejected or no consumer is
// enabled. The caller needs it to pair the hash with the guest shader object the
// create call returns (see dt_plume::OnShaderObject).
uint64_t Record(bool is_vertex,
                uint32_t function_ea,
                uint32_t virtual_size,
                uint32_t physical_size,
                const uint8_t* raw);

void Dump(const char* reason);

}  // namespace dt_shader_dump

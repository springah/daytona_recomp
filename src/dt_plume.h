// Daytona USA -- native renderer bring-up on plume (renderbag).
//
// The long-term goal is to replace rexglue's Xenos GPU emulation with a native
// renderer driven by the intercepted D3D9 calls (see
// _tools/daytona_d3d9_sigscan/RENDERER_INPUT_SPEC.md and SHADOW_STATE_TABLE.md).
//
// THIS STAGE (P0) DELIBERATELY DOES NOT DRAW. It stands up the plume device and
// resolves the game's shaders to their precompiled DXIL, and nothing else. The
// window and the present loop stay with rexglue until the draw path exists, so
// nothing here can affect a normal run.
//
// Gated behind DT_PLUME -- unset, every entry point is a no-op.

#pragma once

#include <cstdint>

namespace dt_plume {

// True when DT_PLUME is set in the environment.
bool Enabled();

// Create the render interface + device. Idempotent, lazily called on first
// shader creation. Returns false if plume could not initialise (logged).
bool Init();

// Called from the CreateVertexShader / CreatePixelShader hooks with the FNV-1a
// of the guest shader container -- the same hash dt_shader_dump computes, and
// the key into the generated dt_shader_blobs table. Looks the shader up and
// creates the plume shader object.
void OnShaderCreated(uint64_t container_hash, bool is_vertex);

// Log what resolved and what did not.
void Dump(const char* reason);

void Shutdown();

}  // namespace dt_plume

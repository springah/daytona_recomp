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

// The guest object address the create call RETURNED, paired with the container
// hash it was built from. This is the only place the two are visible together,
// and it is what lets SetVertexShader/SetPixelShader (which pass an opaque
// handle) be resolved back to a shader we have a blob for.
void OnShaderObject(uint32_t shader_object_ea, uint64_t container_hash);

// ── Draw-state shadow ────────────────────────────────────────────────────────
// Argument decoding lives in gen_trace.py's SEMANTIC map, from the XDK 20500
// d3d9.h prototypes. These run on the game's render thread at up to ~885
// draws/frame, so they must stay cheap.
void OnSetVertexShader(uint32_t shader_ea);
void OnSetPixelShader(uint32_t shader_ea);
void OnSetVertexDeclaration(uint32_t decl_ea);
void OnSetStreamSource(uint32_t stream, uint32_t buffer_ea, uint32_t offset, uint32_t stride);
void OnDrawVertices(uint32_t primitive_type, uint32_t start_vertex, uint32_t vertex_count);

// Log what resolved and what did not.
void Dump(const char* reason);

void Shutdown();

}  // namespace dt_plume

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
//
// `code`/`code_size` are the container's physical region -- the microcode as
// created. It is kept so the LIVE copy can be located inside the shader object
// later: the guest D3D9 runtime patches the vertex layout into the vfetch
// instructions at bind time (proven, commit 3755cbc), and that patched copy is
// the only place the vertex format, stride and offset ever appear.
void OnShaderObject(uint32_t shader_object_ea, uint64_t container_hash,
                    const uint8_t* code, uint32_t code_size);

// Resolve a guest address to a host pointer. The caller owns `base`, which the
// REX_RAW_ADDR macro needs in scope, so it is supplied as a callback.
//
// `size` is how many bytes the caller will read. It is not advisory: these
// addresses come from offsets that are still being reverse-engineered, so a
// wrong one is arbitrary, and most of the 4 GB guest reservation is
// uncommitted. The callback returns nullptr unless the WHOLE range is readable
// (see dt_guest_mem.h for why a crash here is the expensive failure mode).
using GuestResolve = const uint8_t* (*)(uint32_t ea, uint32_t size, void* ctx);

// Locate each shader's live microcode and decode the patched vfetch operands
// into a vertex input layout. Safe to call repeatedly; a layout is only derived
// once the runtime has actually patched that shader (i.e. once it has been
// bound), so early calls simply report nothing yet.
void DeriveVertexLayouts(const char* reason, GuestResolve resolve, void* ctx);

// D3D9 patches shaders two ways: DirectShaderPatch rewrites the stored microcode
// in place, and IncrementalShaderPatchAndLoad (its fallback) writes into a
// separate ucode buffer. The 94.6% path takes the second, which is why its
// stored microcode never changes. This is called after that fallback runs, with
// the ucode buffer it just filled -- the only place that layout is ever visible.
void OnIncrementalPatch(uint32_t shader_ea, uint32_t decl_ea, uint32_t ucode_ea,
                        GuestResolve resolve, void* ctx);

// THE patcher itself: sub_82231BF8, the leaf both patch paths converge on.
//
// Read out of the guest (daytona_recomp.8.cpp:52944 and the two call sites at
// sub_82233260+0xD0 / sub_822324F8+0x12C), its contract is:
//
//   sub_82231BF8(r3 = D3DVertexShader*, r4 = destination microcode,
//                r5 = D3DVertexDeclaration*, r6 = per-element stride bytes,
//                r7 = shader variant index)
//
// and its final loop is
//
//   for each patched element i:  memcpy(r4 + (index[i] & 0xFFF) * 12, triple, 12)
//
// i.e. it writes whole 3-dword vfetch instructions into `r4` indexed by
// INSTRUCTION NUMBER -- the same numbering dt_shader_blobs stores in
// element.address. That is the only moment a patched vertex layout exists in
// addressable memory for the 83%-of-draws shader: IncrementalShaderPatchAndLoad
// hands this function a scratch copy it has just memcpy'd INTO THE PM4 COMMAND
// BUFFER (sub_822324F8 builds an IM_LOAD_IMMEDIATE packet around it), so the
// result is DMA'd to the GPU and the shader object is never touched.
//
// `ucode_ea` must be latched before __imp__ runs: r4 is volatile.
void OnVfetchPatch(uint32_t shader_ea, uint32_t ucode_ea, uint32_t decl_ea,
                   uint32_t variant, GuestResolve resolve, void* ctx);

// Build one plume graphics pipeline per observed (VS, PS, topology) combination.
// Needs the vertex layouts, so it only builds a combination once that shader's
// layout is known. Safe to call repeatedly; already-built ones are skipped.
void BuildPipelines(const char* reason);

// Create the static expansion index buffers. QUADLIST and TRIANGLEFAN do not
// exist in D3D12/Vulkan, and neither depends on the vertex data, so one index
// buffer per topology serves every draw.
bool BuildIndexBuffers();

// Snapshot a draw's vertex data into a plume buffer, byte-swapping per
// component. Guest data is big-endian, and the swap width differs per attribute
// (a blanket dword swap would reverse the 8-8-8-8 COLOR channels), so this needs
// the derived layout. Bounded: only the first few draws per combination are
// captured, since the point is to validate the path, not to feed a renderer yet.
// Takes only `base`: start/count/shader are latched by OnDrawVertices before
// __imp__ runs. Argument registers are volatile, so r5/r6 read after the call
// are clobbered, and the draw's internal flush rewrites the bound shader.
void SnapshotDrawVertices(const uint8_t* base);

// ── Draw-state shadow ────────────────────────────────────────────────────────
// Argument decoding lives in gen_trace.py's SEMANTIC map, from the XDK 20500
// d3d9.h prototypes. These run on the game's render thread at up to ~885
// draws/frame, so they must stay cheap.
void OnSetVertexShader(uint32_t shader_ea);
void OnSetPixelShader(uint32_t shader_ea);
void OnSetVertexDeclaration(uint32_t decl_ea);
void OnSetStreamSource(uint32_t stream, uint32_t buffer_ea, uint32_t offset, uint32_t stride);
// `base` is the guest memory base, needed by the bounded vertex snapshot.
void OnDrawVertices(uint32_t device_ea, uint32_t primitive_type, uint32_t start_vertex,
                    uint32_t vertex_count, const uint8_t* base);

// D3DDevice_GpuBeginShaderConstantF4 -- the game's ONLY route for float shader
// constants. It hands back a pointer and the game writes the values into it, so
// there is no device-side shadow to read: the address must be captured here, at
// the call, and the values read later once the game has filled them.
//
// Measured: PixelShader=0 with StartRegister=72 Count=4 (the mvp matrix) and
// PixelShader=1 with StartRegister=40 Count=4. `dest_ea` is the RETURN value,
// so it can only be read after __imp__ runs.
void OnGpuBeginShaderConstantF4(bool pixel_shader, uint32_t start_register, uint32_t count,
                                uint32_t dest_ea, const uint8_t* base);

// D3DDevice_Resolve -- the handshake that closes the architecture. The GPU
// plugin discards draws, so EDRAM is empty, so the guest's resolve would write
// nothing -- and the guest re-submits its rendering forever waiting for output
// that never lands (measured: two descriptor stacks churning at loop rate;
// xenos converges in six rounds). But THIS renderer has the frame: so at
// resolve time, submit whatever has been recorded and write those pixels into
// the destination texture in guest memory, tiled and byte-ordered the way the
// guest expects to read them.
//
// `dest_texture_ea` is the guest D3DBaseTexture object (its GPU fetch constant
// is embedded in the header); resolution and scaling are derived from it.
void OnResolve(uint32_t dest_texture_ea, const uint8_t* base);

// D3DDevice_Swap -- the game's frame boundary, and so ours.
//
// Every draw between two Swaps is recorded into ONE command list against ONE
// colour+depth target, cleared once. This closes that list, copies the result
// into a swapchain backbuffer, presents it, and frees the frame's uploads.
//
// A frame in which the game drew nothing never opened a list, so this is a
// no-op rather than a present of a stale picture.
void OnFrameEnd();

// Log what resolved and what did not.
void Dump(const char* reason);

void Shutdown();

}  // namespace dt_plume

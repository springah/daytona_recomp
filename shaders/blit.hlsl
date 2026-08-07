// Present blit -- the ONE shader in this port that is not the guest's.
//
// The renderer draws a fixed 1280x720 frame and used to copy it 1:1 into the
// swapchain backbuffer. That required the window to be exactly 1280x720, which
// is why the renderer created its own window at exactly that size, and why
// daytona.ini's `resolution` never affected the picture. It is also why the
// picture FROZE when the window went fullscreen: the swapchain no longer
// matched, and EndFrame skipped presenting entirely rather than show a wrong
// picture.
//
// So present with a draw instead of a copy. Letterboxing is NOT done here --
// the backbuffer is cleared and an aspect-fit viewport is set on the CPU, so
// this shader is a straight sample and the bars are simply where it does not
// draw. That keeps the arithmetic somewhere it can be logged and asserted.
//
// Compiled by scripts/gen_blit_blobs.py with the dxc that XenosRecomp vendors.

// Fullscreen triangle from the vertex id -- no vertex buffer, no input layout.
// Three vertices covering the target: (0,0), (2,0), (0,2) in UV, which maps to
// clip-space (-1,1), (3,1), (-1,-3).
void main_vs(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD0) {
    uv = float2((id << 1) & 2, id & 2);
    pos = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
}

Texture2D<float4> g_source : register(t0, space0);
SamplerState g_sampler : register(s0, space1);

float4 main_ps(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    // No colour conversion, no gamma: the frame target and the backbuffer are
    // both B8G8R8A8_UNORM, so this is the same pixels at a different size.
    return g_source.SampleLevel(g_sampler, uv, 0.0f);
}

// The SDK overlay (settings, debug console, hint toast) under the native
// renderer.
//
// The overlay is Dear ImGui, drawn by the SDK's ImGuiDrawer in DETACHED mode:
// the SDK has no graphics backend here (the plume plugin declines
// presentation), so dt_plume renders the draw lists itself, straight onto the
// swapchain backbuffer after the present blit. This is that pass's shader
// pair: position in display pixels -> clip via a push-constant scale/translate,
// colour = vertex colour * texture (the font atlas, usually).
//
// The Vulkan Y flip is NOT here -- like the blit, it is a negative-height
// viewport at the bind site, so the shader is identical on both backends.
//
// Compiled by scripts/gen_imgui_blobs.py with the dxc that XenosRecomp vendors.

// b0 space2 for D3D12 root constants; the [[vk::push_constant]] block for
// SPIR-V (dxc wants the attribute on a struct-typed global, not cbuffer
// syntax). Same 16 bytes either way.
struct ImGuiTransform {
    float2 scale;
    float2 translate;
};
[[vk::push_constant]] ConstantBuffer<ImGuiTransform> g_xform : register(b0, space2);

struct VSIn {
    float2 pos : POSITION;
    float2 uv  : TEXCOORD0;
    float4 col : COLOR0;
};

void main_vs(VSIn v, out float4 pos : SV_Position, out float2 uv : TEXCOORD0,
             out float4 col : COLOR0) {
    pos = float4(v.pos * g_xform.scale + g_xform.translate, 0.0f, 1.0f);
    uv = v.uv;
    col = v.col;
}

Texture2D<float4> g_texture : register(t0, space0);
SamplerState g_sampler : register(s0, space1);

float4 main_ps(float4 pos : SV_Position, float2 uv : TEXCOORD0,
               float4 col : COLOR0) : SV_Target {
    return col * g_texture.Sample(g_sampler, uv);
}

#include <stereokit.hlsli>

///////////////////////////////////////////
// Generic depth-only shadow caster, also used as the rain occlusion variant.
//
// Both passes render into a single-layer depth target via Renderer.RenderTo.
// sk_renderer creates those subpasses with viewMask=0x1, and Vulkan forbids
// shaders with the Layer (SV_RenderTargetArrayIndex) decoration in any
// pipeline whose subpass viewMask is non-zero. So this shader deliberately
// avoids the multiview-style output the rest of the project uses, drawing
// only to view 0.
///////////////////////////////////////////

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos : SV_Position;
};

///////////////////////////////////////////

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn o;
	float4 world = mul(input.pos, sk_inst[id].world);
	o.pos        = mul(world,     sk_viewproj[0]);
	return o;
}

///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	return float4(1, 1, 1, 1);
}

#include <stereokit.hlsli>

///////////////////////////////////////////
// Alpha-tested depth-only caster, used as the shadow-caster variant for
// transparent surface materials (tree leaves, foliage, etc.). Samples the
// material's own diffuse and discards pixels with alpha below a threshold so
// shadow geometry follows the cutout silhouette instead of the quad edge.
//
// Like rain_caster.hlsl this draws into a single-layer depth target, so it
// must NOT use SV_RenderTargetArrayIndex — the subpass viewMask is non-zero
// and Vulkan rejects pipelines that combine the two.
///////////////////////////////////////////

//--diffuse = white
Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0);

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos : SV_Position;
	float2 uv  : TEXCOORD0;
};

///////////////////////////////////////////

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn o;
	float4 world = mul(input.pos, sk_inst[id].world);
	o.pos        = mul(world,     sk_viewproj[0]);
	o.uv         = input.uv;
	return o;
}

///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	float alpha = diffuse.Sample(diffuse_s, input.uv).a;
	if (alpha < 0.25) discard;
	return float4(1, 1, 1, 1);
}

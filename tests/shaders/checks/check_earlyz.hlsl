//--name = check/earlyz
// Early-Z toolkit runtime check, in spellings both compilers accept:
// conservative depth (SV_DepthGreaterEqual → DepthReplacing + DepthGreater),
// pixel-diffed against skshaderc. `invariant` and [early_depth_stencil] are
// SVSL dialect and live in ported/73_demote.hlsl instead.

#include "stereokit.hlsli"

//--color:color = 1, 1, 1, 1
float4 color;

Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0);

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos   : SV_POSITION;
	float2 uv    : TEXCOORD0;
	float3 norm  : NORMAL0;
};
struct psOut {
	float4 color : SV_Target;
	float  depth : SV_DepthGreaterEqual;
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;
	float4 world = mul(float4(input.pos.xyz, 1), sk_inst[ids.inst].world);
	o.pos        = mul(world, sk_viewproj[ids.view]);
	o.uv         = input.uv;
	o.norm       = normalize(mul(input.norm, (float3x3)sk_inst[ids.inst].world));
	return o;
}

psOut ps(psIn input) {
	psOut o;
	float4 c = diffuse.Sample(diffuse_s, input.uv) * color;
	o.color  = float4(c.rgb * (0.4 + 0.6 * saturate(input.norm.y)), 1);
	// nudge depth away from the camera only — the promise DepthGreater makes
	o.depth  = input.pos.z + (1 - input.pos.z) * 0.05 * c.r;
	return o;
}

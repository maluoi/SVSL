//--name = dissolve
//--tex = white

#include "common.hlsli"

// Dissolve effect using noise texture and discard

float4 color        = float4(1, 1, 1, 1);
float4 edge_color   = float4(1.0, 0.5, 0.0, 1);
float4 tex_trans    = float4(0, 0, 1, 1);
float  dissolve     = 0.5; // 0 = fully visible, 1 = fully dissolved
float  edge_width   = 0.1;

struct Inst {
	float4x4 world;
};
StructuredBuffer<Inst> inst : register(t2, space0);

struct vsIn {
	float3 pos   : SV_POSITION;
	float3 norm  : NORMAL;
	float2 uv    : TEXCOORD0;
	float4 color : COLOR0;
};
struct psIn {
	float4 pos       : SV_POSITION;
	float3 normal    : NORMAL;
	float2 uv        : TEXCOORD0;
	float4 color     : COLOR0;
	float3 world_pos : TEXCOORD1;
	uint   layer     : SV_RenderTargetArrayIndex;
};

Texture2D    tex         : register(t3);
SamplerState tex_sampler : register(s3);

// Simple hash function for noise
float hash(float2 p) {
	float3 p3 = frac(float3(p.x, p.y, p.x) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return frac((p3.x + p3.y) * p3.z);
}

// 2D noise
float noise(float2 p) {
	float2 i = floor(p);
	float2 f = frac(p);
	float2 u = f * f * (3.0 - 2.0 * f);

	float a = hash(i + float2(0.0, 0.0));
	float b = hash(i + float2(1.0, 0.0));
	float c = hash(i + float2(0.0, 1.0));
	float d = hash(i + float2(1.0, 1.0));

	return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

psIn vs(vsIn input, uint id : SV_InstanceID) {
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	float4 world_pos = mul(float4(input.pos, 1), inst[inst_idx].world);

	psIn output;
	output.pos       = mul(world_pos, viewproj[view_idx]);
	output.normal    = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;
	output.world_pos = world_pos.xyz;
	output.layer     = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	const float3 light_dir = normalize(float3(1, 3, 2));

	// Generate noise for dissolve pattern
	float n = noise(input.uv * 10.0);

	// Discard pixels below dissolve threshold
	if (n < dissolve) {
		discard;
	}

	float3 normal = normalize(input.normal);

	// Sample texture
	float4 albedo = tex.Sample(tex_sampler, input.uv) * input.color;

	// Simple diffuse lighting
	float ndotl = saturate(dot(normal, light_dir));
	float3 final_color = albedo.rgb * (0.2 + ndotl * 0.8);

	// Add glowing edge near dissolve boundary
	float edge = smoothstep(dissolve, dissolve + edge_width, n);
	final_color = lerp(edge_color.rgb, final_color, edge);

	return float4(final_color, albedo.a);
}

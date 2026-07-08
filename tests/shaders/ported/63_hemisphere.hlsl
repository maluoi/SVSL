//--name = hemisphere
//--tex = white

#include "common.hlsli"

// Hemisphere ambient lighting (sky/ground colors)

float4 color       = float4(1, 1, 1, 1);
float4 sky_color   = float4(0.5, 0.7, 1.0, 1);
float4 ground_color = float4(0.3, 0.2, 0.1, 1);
float4 tex_trans   = float4(0, 0, 1, 1);
float  direct_light = 0.5;

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
	const float3 up = float3(0, 1, 0);

	float3 normal = normalize(input.normal);

	// Hemisphere lighting - blend sky/ground based on normal Y
	float hemisphere = dot(normal, up) * 0.5 + 0.5;
	float3 ambient_color = lerp(ground_color.rgb, sky_color.rgb, hemisphere);

	// Direct light contribution
	float ndotl = saturate(dot(normal, light_dir));
	float3 direct = ndotl * direct_light;

	// Sample texture
	float4 albedo = tex.Sample(tex_sampler, input.uv) * input.color;

	// Combine hemisphere ambient with direct light
	float3 final_color = albedo.rgb * (ambient_color + direct);

	return float4(final_color, albedo.a);
}

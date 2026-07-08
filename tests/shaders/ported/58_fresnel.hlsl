//--name = fresnel
//--tex = white

#include "common.hlsli"

// Fresnel reflection with environment-like effect

float4 color       = float4(1, 1, 1, 1);
float4 tex_trans   = float4(0, 0, 1, 1);
float4 reflect_col = float4(0.4, 0.7, 1.0, 1);
float  fresnel_pow = 3.0;
float  fresnel_bias = 0.1;
float  ambient     = 0.2;

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

	float3 normal   = normalize(input.normal);
	float3 view_dir = normalize(cam_pos[0].xyz - input.world_pos);

	// Schlick's fresnel approximation
	float ndotv   = saturate(dot(normal, view_dir));
	float fresnel = fresnel_bias + (1.0 - fresnel_bias) * pow(1.0 - ndotv, fresnel_pow);

	// Diffuse lighting
	float ndotl   = saturate(dot(normal, light_dir));
	float diffuse = ambient + ndotl * (1.0 - ambient);

	// Sample base texture
	float4 albedo = tex.Sample(tex_sampler, input.uv) * input.color;

	// Blend between diffuse and reflection based on fresnel
	float3 diffuse_color = albedo.rgb * diffuse;
	float3 final_color   = lerp(diffuse_color, reflect_col.rgb, fresnel);

	return float4(final_color, albedo.a);
}

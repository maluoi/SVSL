//--name = triplanar
//--tex = white

#include "common.hlsli"

// Triplanar mapping - projects texture from 3 axes

float4 color      = float4(1, 0, 0, 1);
float4 tex_trans  = float4(0, 0, 1, 1);
float  tex_scale  = 1.0;
float  blend_sharpness = 2.0;

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
	float3 model_pos : TEXCOORD1;  // Model space for triplanar (object-relative)
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
	output.model_pos = input.pos;  // Pass model-space position for triplanar
	output.layer     = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	const float3 light_dir = normalize(float3(1, 3, 2));

	float3 normal = normalize(input.normal);
	float3 abs_normal = abs(normal);

	// Triplanar blend weights (normalized to sum to 1)
	float3 blend = pow(abs_normal, blend_sharpness);
	blend = blend / (blend.x + blend.y + blend.z);

	// Sample texture from all 3 projections (model space)
	float3 scaled = input.model_pos * tex_scale;
	float4 tex_x = tex.Sample(tex_sampler, scaled.yz);
	float4 tex_y = tex.Sample(tex_sampler, scaled.xz);
	float4 tex_z = tex.Sample(tex_sampler, scaled.xy);

	// Blend based on normal
	float4 albedo = tex_x * blend.x + tex_y * blend.y + tex_z * blend.z;
	albedo *= input.color;

	// Simple diffuse lighting
	float ndotl = saturate(dot(normal, light_dir));
	float3 final_color = albedo.rgb * (0.2 + ndotl * 0.8);

	return float4(final_color, albedo.a);
}

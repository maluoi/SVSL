//--name = envmap
//--tex = white

#include "common.hlsli"

// Environment map reflection shader

float4 color       = float4(1, 1, 1, 1);
float4 tex_trans   = float4(0, 0, 1, 1);
float  roughness   = 0.2;
float  metallic    = 0.8;

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

Texture2D    tex             : register(t3);
SamplerState tex_sampler     : register(s3);
TextureCube  environment_map : register(t4);
SamplerState env_sampler     : register(s4);

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

	float3 normal = normalize(input.normal);

	// Calculate view direction (from world pos to camera)
	float3 view_dir = normalize(cam_pos[input.layer].xyz - input.world_pos);

	// Reflection vector
	float3 reflect_dir = reflect(-view_dir, normal);

	// Sample environment map with mip based on roughness
	float mip_level = roughness * 8.0; // 8 mip levels typical for 256x256 cubemap
	float3 env_color = environment_map.SampleLevel(env_sampler, reflect_dir, mip_level).rgb;

	// Base color from texture
	float4 albedo = tex.Sample(tex_sampler, input.uv) * input.color;

	// Simple diffuse
	float ndotl = saturate(dot(normal, light_dir));
	float3 diffuse = albedo.rgb * ndotl * 0.5;

	// Fresnel approximation (Schlick)
	float fresnel = pow(1.0 - saturate(dot(view_dir, normal)), 5.0);
	fresnel = lerp(0.04, 1.0, fresnel);

	// Mix diffuse and reflection based on metallic/fresnel
	float3 specular = env_color * lerp(fresnel, 1.0, metallic);
	float3 final_color = lerp(diffuse + albedo.rgb * 0.1, specular, metallic);

	return float4(final_color, albedo.a);
}

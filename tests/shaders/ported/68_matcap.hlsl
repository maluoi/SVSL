//--name = matcap
//--tex = white

#include "common.hlsli"

// MatCap (Material Capture) shader - uses view-space normal to sample 2D matcap texture

float4 color       = float4(1, 1, 1, 1);
float4 tex_trans   = float4(0, 0, 1, 1);
float  brightness  = 1.0;

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
	float4 pos         : SV_POSITION;
	float3 view_normal : NORMAL; // Normal in view space
	float2 uv          : TEXCOORD0;
	float4 color       : COLOR0;
	float3 world_pos   : TEXCOORD1;
	uint   layer       : SV_RenderTargetArrayIndex;
};

Texture2D    tex         : register(t3);
SamplerState tex_sampler : register(s3);

psIn vs(vsIn input, uint id : SV_InstanceID) {
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	float4 world_pos = mul(float4(input.pos, 1), inst[inst_idx].world);

	// Transform normal to world space then to view space
	float3 world_normal = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	float3 view_normal = normalize(mul(float4(world_normal, 0), view[view_idx]).xyz);

	psIn output;
	output.pos         = mul(world_pos, viewproj[view_idx]);
	output.view_normal = view_normal;
	output.uv          = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color       = input.color * color;
	output.world_pos   = world_pos.xyz;
	output.layer       = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	// Convert view-space normal to matcap UV coordinates
	// Normal xy in [-1,1] maps to UV in [0,1]
	float3 normal = normalize(input.view_normal);
	float2 matcap_uv = normal.xy * 0.5 + 0.5;

	// Sample matcap texture
	float4 matcap = tex.Sample(tex_sampler, matcap_uv);

	// Apply color tint and brightness
	float4 final_color = matcap * input.color * brightness;

	return final_color;
}

//--name = texture3d
//--tex = white

#include "common.hlsli"

struct Inst {
	float4x4 world;
};
StructuredBuffer<Inst> inst : register(t2, space0);

struct vsIn {
	float3 pos  : SV_POSITION;
	float3 norm : NORMAL;
	float2 uv   : TEXCOORD0;
	float4 color: COLOR0;
};
struct psIn {
	float4 pos       : SV_POSITION;
	float3 world_pos : TEXCOORD0;
	float3 color     : COLOR0;
	uint   layer     : SV_RenderTargetArrayIndex;  // Output to array layer
};

Texture3D    tex         : register(t0);
SamplerState tex_sampler : register(s0);

psIn vs(vsIn input, uint id : SV_InstanceID) {
	const float3 light_dir = normalize(float3(1, 4, 2));

	// Multi-view instancing: extract instance index and view index
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	psIn output;
	float4 world_position = mul(float4(input.pos, 1), inst[inst_idx].world);
	output.pos = mul(world_position, viewproj[view_idx]);
	output.world_pos = world_position.xyz;
	float3 normal = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.color = input.color.rgb;
	output.layer = view_idx;  // Route each view to its corresponding array layer
	return output;
}
float4 ps(psIn input) : SV_TARGET {
	// Sample from the 3D texture using world space coordinates
	// Map world space (-2 to +2) to texture space (0 to 1)
	float3 uvw = (input.world_pos + 2.0) / 4.0;
	return float4(input.color, 1) * tex.Sample(tex_sampler, uvw);
}

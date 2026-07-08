//--name = cubemap_skybox

#include "common.hlsli"

struct vsIn {
	float3 pos    : SV_POSITION;
	float3 normal : NORMAL;
	float2 uv     : TEXCOORD0;
	float4 color  : COLOR;
};
struct psIn {
	float4 pos   : SV_POSITION;
	float3 dir   : TEXCOORD0;
	uint   layer : SV_RenderTargetArrayIndex;
};

TextureCube  cubemap         : register(t0);
SamplerState cubemap_sampler : register(s0);

psIn vs(vsIn input, uint id : SV_InstanceID) {
	// Multi-view instancing: extract view index
	uint view_idx = id % view_count;

	psIn output;
	output.pos   = float4(input.pos, 1.0);
	output.pos.z = 1; // Force Z to the back

	// Calculate view direction from inverse projection and view matrices
	float4 proj_inv = mul(output.pos, projection_inv[view_idx]);
	output.dir = mul(float4(proj_inv.xyz, 0), transpose(view[view_idx])).xyz;

	output.layer = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	// Sample cubemap using the direction vector
	float3 dir   = normalize(input.dir);
	float4 color = cubemap.Sample(cubemap_sampler, dir);
	return color;
}

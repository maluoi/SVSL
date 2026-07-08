//--name = shadow_caster

#include "common.hlsli"

// Shadow caster shader - renders depth to shadow map

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
	float4 pos : SV_POSITION;
	uint   layer : SV_RenderTargetArrayIndex;
};

psIn vs(vsIn input, uint id : SV_InstanceID) {
	// Multi-view instancing: extract instance index and view index
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	psIn output;
	output.pos = mul(float4(input.pos, 1), inst[inst_idx].world);
	output.pos = mul(output.pos, viewproj[view_idx]);
	output.layer = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	// Depth-only rendering - return value is ignored but shader must have output
	// The depth is written automatically from SV_POSITION
	return float4(1, 1, 1, 1);
}

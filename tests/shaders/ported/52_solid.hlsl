//--name = solid
//--tex = white

#include "common.hlsli"

// Simplest possible shader - just a solid color

float4 color = float4(1, 0.5, 0.2, 1);

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
	float4 pos   : SV_POSITION;
	uint   layer : SV_RenderTargetArrayIndex;
};

psIn vs(vsIn input, uint id : SV_InstanceID) {
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	psIn output;
	output.pos   = mul(float4(input.pos, 1), inst[inst_idx].world);
	output.pos   = mul(output.pos, viewproj[view_idx]);
	output.layer = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	return color;
}

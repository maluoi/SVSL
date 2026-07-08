//--name = unlit
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
	float4 pos   : SV_POSITION;
	float2 uv    : TEXCOORD0;
	float4 color : COLOR0;
	uint   layer : SV_RenderTargetArrayIndex;
};

Texture2D    tex         : register(t3);
SamplerState tex_sampler : register(s3);
float4       tex_trans = float4(0,0,1,1);

psIn vs(vsIn input, uint id : SV_InstanceID) {
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	psIn output;
	output.pos   = mul(float4(input.pos, 1), inst[inst_idx].world);
	output.pos   = mul(output.pos, viewproj[view_idx]);
	output.color = input.color;
	output.uv    = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.layer = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	return input.color * tex.Sample(tex_sampler, input.uv);
}

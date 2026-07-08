//--name = test
//--tex = white

#include "common.hlsli"

float4 emissive;

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
	float3 color : COLOR0;
	uint   layer : SV_RenderTargetArrayIndex;  // Output to array layer
};

Texture2D    tex         : register(t3);
SamplerState tex_sampler : register(s3);
float4       tex_trans   = float4(0,0,1,1);

psIn vs(vsIn input, uint id : SV_InstanceID) {
	const float3 light_dir = normalize(float3(1, 4, 2));

	// Multi-view instancing: extract instance index and view index
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	psIn output;
	output.pos = mul(float4(input.pos, 1), inst[inst_idx].world);
	output.pos = mul(output.pos, viewproj[view_idx]);
	float3 normal = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.color = (float3(.05,.05,.1) + saturate(dot(normal, light_dir)*0.8).xxx) * input.color.rgb;
	output.uv    = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.layer = view_idx;  // Route each view to its corresponding array layer
	return output;
}
float4 ps(psIn input) : SV_TARGET {
	return float4(input.color, 1) * tex.Sample(tex_sampler, input.uv) + emissive;
}

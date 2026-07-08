//--name = bloom_composite
// Final bloom composite - adds bloom to original scene

cbuffer BloomParams : register(b1, space0) {
	float bloom_strength;
	float _pad0;
	float _pad1;
	float _pad2;
};

Texture2D    scene_tex      : register(t0);
SamplerState scene_sampler  : register(s0);
Texture2D    bloom_tex      : register(t2);
SamplerState bloom_sampler  : register(s2);

struct vsIn {
	float3 pos    : SV_POSITION;
	float3 normal : NORMAL;
	float2 uv     : TEXCOORD0;
	float4 color  : COLOR;
};

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

psIn vs(vsIn input) {
	psIn output;
	output.pos = float4(input.pos.xy,0,1);
	output.uv  = input.uv;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	float4 scene = scene_tex.Sample(scene_sampler, input.uv);
	float4 bloom = bloom_tex.Sample(bloom_sampler, input.uv);

	// Additive blend with strength control
	return scene + bloom * bloom_strength;
}

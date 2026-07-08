//--name = stereo_display

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

Texture2DArray array_tex      : register(t0);
SamplerState   array_sampler  : register(s0);

psIn vs(vsIn input) {
	psIn output;
	output.pos = float4(input.pos, 1.0);  // Convert float3 to float4
	output.uv  = input.uv;
	// Note: normal and color are not used, but must be in vertex layout
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	// Sample both layers of the array texture
	float3 left  = array_tex.Sample(array_sampler, float3(input.uv, 0)).rgb;  // Layer 0 (left eye)
	float3 right = array_tex.Sample(array_sampler, float3(input.uv, 1)).rgb;  // Layer 1 (right eye)

	// Red/cyan anaglyph: left eye in red channel, right eye in cyan channels
	float3 stereo = float3(left.r, right.gb);

	return float4(stereo, 1);
}

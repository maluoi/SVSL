// video.hlsl - NV12 video display with YUV to RGB conversion
//--name = video

#include "common.hlsli"

// NV12 video planes
Texture2D<float>  tex_y    : register(t0);  // R8 luma (full resolution)
Texture2D<float2> tex_uv   : register(t1);  // RG8 chroma (half resolution, interleaved U,V)
SamplerState      tex_y_s  : register(s0);
SamplerState      tex_uv_s : register(s1);

// Instance data
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
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

psIn vs(vsIn input, uint id : SV_InstanceID) {
	psIn o;
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	// Apply world matrix for aspect ratio scaling (NDC space, no viewproj needed)
	float4 scaled_pos = mul(float4(input.pos, 1), inst[inst_idx].world);
	o.pos = scaled_pos;

	o.uv = input.uv;
	return o;
}

// sRGB to linear conversion (needed because render target is sRGB)
float3 srgb_to_linear(float3 srgb) {
	// Approximate: pow(srgb, 2.2)
	// Exact would be: c <= 0.04045 ? c/12.92 : pow((c+0.055)/1.055, 2.4)
	return pow(srgb, 2.2);
}

float4 ps(psIn input) : SV_TARGET {
	// Sample NV12 planes
	float  y  = tex_y.Sample(tex_y_s, input.uv);
	float2 uv = tex_uv.Sample(tex_uv_s, input.uv);

	// YUV uses full range (0-255 normalized to 0-1)
	// Center UV around 0
	float Y = y;
	float U = uv.x - 0.5;
	float V = uv.y - 0.5;

	// BT.709 YUV to RGB conversion matrix (HD video standard)
	float3 rgb;
	rgb.r = Y + 1.5748 * V;
	rgb.g = Y - 0.1873 * U - 0.4681 * V;
	rgb.b = Y + 1.8556 * U;

	// YUV->RGB produces sRGB (gamma-encoded) values
	// Convert to linear for sRGB render target (which will re-apply gamma)
	float3 linear_rgb = srgb_to_linear(saturate(rgb));

	return float4(linear_rgb, 1.0);
}

#include <stereokit.hlsli>

///////////////////////////////////////////

//--brightness   = {1, 0, 0, 0}
//--screen_size  = {1920, 1080, 0, 0}
float4 brightness;   // x = star brightness multiplier
float4 screen_size;  // xy = display resolution

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;  // x = vertex index (0,1,2), y = per-star brightness
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos   : SV_POSITION;
	float4 color : COLOR0;
};

///////////////////////////////////////////

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;

	// Rotate star sphere around celestial axis via world matrix
	float3 dir = normalize(mul(float4(input.pos.xyz, 0), sk_inst[ids.inst].world).xyz);

	// View rotation only (no translation) — locks stars around head
	float4x4 rot_view = sk_view[ids.view];
	rot_view._m30 = 0;
	rot_view._m31 = 0;
	rot_view._m32 = 0;
	float4 clip_pos = mul(mul(float4(dir, 1), rot_view), sk_proj[ids.view]);

	// Expand triangle to ~1 pixel
	int   vertex_idx = (int)input.uv.x;
	float2 pixel_size = (2.0 / screen_size.xy) * clip_pos.w;
	float2 offsets[3] = {
		float2( 0.0,    1.0),
		float2(-0.866, -0.5),
		float2( 0.866, -0.5),
	};
	clip_pos.xy += offsets[vertex_idx] * pixel_size * 0.75;

	// Same depth as sky dome — uses LessOrEq depth test
	clip_pos.z = clip_pos.w;

	o.pos   = clip_pos;
	o.color = input.col * brightness.x;
	return o;
}

///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	return input.color;
}

//--name = check/subpass
// SubpassInput runtime check: runs as a postfx subpass over the baseline
// sphere via sk_renderer's tile-local input attachment path, and both
// compilers' outputs are pixel-diffed. Vignette + channel remap so the result
// depends on both the input color and screen position. The attachment must be
// named `color` — sk_renderer's postfx pass auto-binds it by that name.

[[vk::input_attachment_index(0)]] SubpassInput<float4> color;

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

psIn vs(uint id : SV_VertexID) {
	psIn o;
	float2 uv = float2((id << 1) & 2, id & 2);
	o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
	o.uv  = uv;
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	float4 c        = color.SubpassLoad();
	float2 centered = input.uv - 0.5;
	float  vignette = 1.0 - dot(centered, centered) * 1.2;
	float3 remapped = float3(c.g, c.b, c.r) * vignette;
	return float4(lerp(c.rgb, remapped, 0.75), c.a);
}

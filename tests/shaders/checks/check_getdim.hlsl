//--name = check/getdim
// GetDimensions runtime check: queried sizes drive the output color, so any
// divergence from skshaderc (wrong opcode, component order, or binding) shows
// up in the pixel diff. Uses the HLSL out-parameter form both compilers accept;
// the SVSL value form (uint2 s = tex.GetDimensions()) is covered in ported/.

#include "stereokit.hlsli"

//--color:color = 1, 1, 1, 1
float4 color;

Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0);

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;
	float4 world = mul(float4(input.pos.xyz, 1), sk_inst[ids.inst].world);
	o.pos        = mul(world, sk_viewproj[ids.view]);
	o.uv         = input.uv;
	return o;
}

float4 ps(psIn input) : SV_Target {
	float fw, fh;
	diffuse.GetDimensions(fw, fh);
	uint uw, uh;
	diffuse.GetDimensions(uw, uh);
	uint mw, mh, levels;
	diffuse.GetDimensions(1, mw, mh, levels); // mip overload: sizes at mip 1
	float lod = saturate(diffuse.CalculateLevelOfDetail(diffuse_s, input.uv) * 0.25);

	// out-parameter and one-liner math, pixel-diffed against glslang
	float s, c;
	sincos(input.uv.x * 6.28318, s, c);
	float ip;
	float fr   = modf(input.uv.x * 4.0 + 1.5, ip);
	float ex;
	float mant = frexp(input.uv.y * 100.0 + 1.0, ex);
	// NOT ldexp: glslang produces invalid SPIR-V for HLSL's float exponent
	// (Ldexp with a float Exp operand); SVSL lowers it as x * exp2(e) instead
	float misc = rcp(input.uv.x + 1.5) + degrees(input.uv.y) * 0.001 +
	             radians(input.uv.x * 90.0);

	float4 samp  = diffuse.Sample(diffuse_s, input.uv);
	float2 texel = input.uv * float2(fw, fh); // uv scaled into texel space
	float  dims  = (float)(uw + uh + mw + mh + levels) / (fw + fh);
	return float4(samp.rgb * color.rgb, 1) *
	       float4(frac(texel * 0.03125), saturate(dims * 0.3), 1) *
	       saturate(float4(s * 0.5 + 0.5 + lod, c * 0.5 + 0.5, fr + ip * 0.1, 1) *
	                float4(1, 1, 1, mant * 0.5 + ex * 0.05 + misc * 0.1));
}

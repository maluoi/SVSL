//--name = check/qcom_fog_scatter
// The driving use case for VK_QCOM_image_processing in SVSL: depth-based fog
// with light scatter that widens deep in the fog. Fog density comes from the
// depth texture; the scene color is then filtered through a BoxFilterQCOM
// kernel whose size grows with density, so distant geometry blurs out into the
// fog instead of staying crisp behind a color fade.
// Declaration order matters for slot pairing: sceneColor(t0) fuses with
// scatterSmp(s0) — an image-processing sampler is exclusive, so the fused pair
// keeps BoxFilterQCOM and plain sampling from ever sharing a sampler.
// Compile + spirv-val only: no desktop runtime implements the extension.

Texture2D    sceneColor; // t0, only ever box-filtered
Texture2D    sceneDepth; // t1, plain-sampled
SamplerState scatterSmp; // s0, fuses with sceneColor (IMAGE_PROCESSING sampler)
SamplerState depthSmp;   // s1, fuses with sceneDepth

float4 fog_color   = float4(0.45, 0.55, 0.70, 1);
float  fog_falloff = 2.0;  // density = 1 - exp(-depth * falloff)
float  scatter_max = 15.0; // widest blur kernel, in texels

float4 ps(float2 uv : TEXCOORD0) : SV_Target {
	float depth   = sceneDepth.Sample(depthSmp, uv).r;
	float density = 1.0 - exp(-depth * fog_falloff);

	// scatter: the box kernel grows from 1 texel to scatter_max with density
	float  radius  = 1.0 + density * scatter_max;
	float4 scatter = sceneColor.BoxFilterQCOM(scatterSmp, uv, float2(radius, radius));

	return lerp(scatter, fog_color, density * 0.85);
}

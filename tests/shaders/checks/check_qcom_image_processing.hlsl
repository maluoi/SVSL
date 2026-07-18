//--name = check/qcom_image_processing
// VK_QCOM_image_processing + image_processing2: every method form. Verifies the
// emitted caps/extensions (TextureSampleWeightedQCOM, TextureBoxFilterQCOM,
// TextureBlockMatchQCOM, TextureBlockMatch2QCOM), the SPIR-V 1.4 version bump
// with its full-interface entry point, and the usage-inferred decorations:
// WeightTextureQCOM on `weights`, BlockMatchTextureQCOM on `target`/`reference`,
// BlockMatchSamplerQCOM on `winSmp` (Window forms decorate the sampler, so it
// uses a sampler distinct from the plain block-match one).
// Compile + spirv-val only: skshaderc cannot compile QCOM methods and desktop
// GPUs do not implement the extensions, so no runtime tier exists.

Texture2D      tex;
Texture2D      target;
Texture2D      winTarget;
Texture2D      reference;
Texture2D      winReference;
Texture2DArray weights;
SamplerState   smp;
SamplerState   bmSmp;
SamplerState   winSmp;

float4 ps(float2 uv : TEXCOORD0) : SV_Target {
	// weighted sampling + box filter share a sampler; their target texture stays
	// ordinary (it may also be sampled plainly)
	float4 w  = tex.SampleWeightedQCOM(smp, uv, weights);
	float4 b  = tex.BoxFilterQCOM(smp, uv, float2(3.0, 3.0));

	// plain + gather block match (image_processing / image_processing2)
	float4 m0 = target.BlockMatchSADQCOM      (bmSmp, uint2(16, 16), reference, uint2(8, 8), uint2(8, 8));
	float4 m1 = target.BlockMatchSSDQCOM      (bmSmp, uint2(16, 16), reference, uint2(8, 8), uint2(8, 8));
	float4 g0 = target.BlockMatchGatherSADQCOM(bmSmp, uint2(16, 16), reference, uint2(8, 8), uint2(8, 8));
	float4 g1 = target.BlockMatchGatherSSDQCOM(bmSmp, uint2(16, 16), reference, uint2(8, 8), uint2(8, 8));

	// window block match decorates its sampler → exclusive to Window forms
	float4 w0 = winTarget.BlockMatchWindowSADQCOM(winSmp, uint2(16, 16), winReference, uint2(8, 8), uint2(8, 8));
	float4 w1 = winTarget.BlockMatchWindowSSDQCOM(winSmp, uint2(16, 16), winReference, uint2(8, 8), uint2(8, 8));

	return w + b + m0 + m1 + g0 + g1 + w0 + w1;
}

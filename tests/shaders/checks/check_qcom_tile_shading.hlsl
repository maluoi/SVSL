//--name = check/qcom_tile_shading
//--apron = 4
// VK_QCOM_tile_shading, fragment side: a [tile_attachment] texture read while
// the tile sits in on-die memory, the [non_coherent_tile_reads_qcom] execution
// mode, the tile builtins, and the //--apron metadata (carried to the renderer,
// which applies it as VkRenderPassTileShadingCreateInfoQCOM::tileApronSize —
// the apron has no shader-side representation; shaders only read the active
// size via tile_apron_size_qcom()).
// Compile + spirv-val only: no desktop runtime implements the extension.

[tile_attachment] Texture2D<float4> sceneColor; // t0, fuses with smp(s0)
SamplerState smp;

struct psIn {
	float4 pos : SV_Position;
	float2 uv  : TEXCOORD0;
};

psIn vs(float3 pos : POSITION, float2 uv : TEXCOORD0) {
	psIn o;
	o.pos = float4(pos, 1);
	o.uv  = uv;
	return o;
}

[non_coherent_tile_reads_qcom]
float4 ps(psIn i) : SV_Target {
	uint2 off   = tile_offset_qcom();
	uint2 apron = tile_apron_size_qcom();
	uint3 dim   = tile_dimension_qcom();

	// blur within the tile: the apron guarantees neighbors exist at tile edges
	float2 texel = 1.0 / float2((float)dim.x, (float)dim.y);
	float4 c = sceneColor.Sample(smp, i.uv);
	c += sceneColor.Sample(smp, i.uv + texel * (float)apron.x);
	c += sceneColor.Sample(smp, i.uv - texel * (float)apron.y);
	return c / 3.0 + float4((float)off.x * 0.0001, 0, 0, 0);
}

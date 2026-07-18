//--name = check/qcom_tile_dispatch
//--apron = 2
// VK_QCOM_tile_shading, compute side: per-tile dispatch. The
// [tile_shading_rate_qcom(x, y, z)] execution mode REPLACES [numthreads] —
// the implementation derives the workgroup shape from the rate (threads map
// proportionally onto tile pixels), so declaring both is a compile error.
// Tile attachments keep normal set/binding descriptors but live in the
// TileAttachmentQCOM storage class.
// Compile + spirv-val only: no desktop runtime implements the extension, so
// this compute shader deliberately has no compute_cfgs[] entry in svsl_view.

[tile_attachment] Texture2D<float4>   lastFrame;
[tile_attachment] RWTexture2D<float4> colorOut;

RWStructuredBuffer<float4> debugOut;

[tile_shading_rate_qcom(2, 2, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint2 off   = tile_offset_qcom();
	uint3 dim   = tile_dimension_qcom();
	uint2 apron = tile_apron_size_qcom();

	float4 prev = lastFrame.Load(int3((int2)id.xy, 0));
	colorOut[(int2)id.xy] = prev * 0.9;
	debugOut[id.x] = float4((float)off.x, (float)dim.z, (float)apron.x, prev.x);
}

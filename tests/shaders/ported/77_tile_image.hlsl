// SPV_EXT_shader_tile_image coverage: color attachment reads in tile memory
// (OpColorAttachmentReadEXT via TileImage<T, N>), plus this fragment's depth
// and stencil (OpDepth/StencilAttachmentReadEXT). Compile + validate only —
// the runtime needs VK_EXT_shader_tile_image, which neither sk_renderer nor
// desktop RADV exposes yet. glslang has no HLSL spelling for any of this;
// the types and intrinsics below are native SVSL.

TileImage<float4, 0> tile_color;
TileImage<float4, 1> tile_normal;

struct psIn {
	float4 pos : SV_POSITION;
	float2 uv  : TEXCOORD0;
};

float4 ps(psIn input) : SV_Target {
	float4 color  = tile_color.Load();
	float4 normal = tile_normal.Load();
	float  depth  = tile_depth();
	uint   sten   = tile_stencil();

	// depth-aware tint using only tile memory — the tiler never leaves the tile
	float fog = saturate(1.0 - depth);
	return float4(color.rgb * fog + normal.rgb * 0.1, color.a) +
	       (float)(sten & 1u) * 0.01;
}

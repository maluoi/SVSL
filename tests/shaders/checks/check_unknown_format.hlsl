//--name = check/unknown_format
// Format-agnostic storage images (SPIR-V image format Unknown): the bound
// view's format drives the hardware's load/store conversion, so the same shader
// works against any compatible view. Omitting the format means this; the two
// explicit spellings below say it out loud.
//
// No reference tier: glslang has no `unknown` layout format, so skshaderc
// cannot compile either spelling. Goldens are derived from the thread id alone
// (not from the harness texture fill) so they hold whatever view format the
// runtime binds.

RWStructuredBuffer<uint>                             result     : register(u0);
[[vk::image_format("unknown")]] RWTexture2D<float4>  attr_tex   : register(u1);
Image2D<float4, unknown>                             native_tex : register(u2);

[numthreads(8, 8, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	// both images are write-only, so this asks for StorageImageWriteWithoutFormat
	// and NOT the read half — the feature with weaker old-mobile coverage
	attr_tex  [id.xy] = float4(1, 0, 0, 1);
	native_tex[id.xy] = float4(0, 1, 0, 1);

	// texture contents are deliberately not in expect[]: the goldens must not
	// depend on which format the harness picked for the view, which is the whole
	// point of an agnostic image. The buffer carries the checkable output.
	if (id.y == 0 && id.x < 4)
		result[id.x] = id.x * 7 + 3; // 3, 10, 17, 24
}

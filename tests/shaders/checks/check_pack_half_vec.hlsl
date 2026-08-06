//--name = check/pack_half_vec
// Vector f32tof16/f16tof32: per-lane GLSL450 packs on SPIR-V, paired
// pack2x16float lowering on WGSL. Every value is exactly representable in
// f16, so round-toward-zero (D3D) vs round-to-nearest (pack2x16float) cannot
// diverge — safe to compare bitwise against the skshaderc reference.
// `base`/`ubase` are 0 at runtime but opaque at compile time, keeping the
// conversions in the emitted code instead of the constant folder.
RWStructuredBuffer<uint> result : register(u0);

[numthreads(1, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	float base  = float(id.x); // 0.0
	uint  ubase = id.x;        // 0

	uint2  h2 = f32tof16(float2(base + 1.0, base - 2.0));
	uint3  h3 = f32tof16(float3(base + 0.5, base + 4096.0, base - 0.25));
	uint4  h4 = f32tof16(float4(base + 65504.0, base - 65504.0, base, base + 1.5));
	float2 f2 = f16tof32(uint2(ubase + 0x3C00, ubase + 0xC000)); //  1.0, -2.0
	float3 f3 = f16tof32(uint3(ubase + 0x3800, ubase + 0x6C00,
	                           ubase + 0xB400));                 //  0.5, 4096, -0.25
	float4 f4 = f16tof32(uint4(ubase + 0x7BFF, ubase + 0xFBFF,
	                           ubase,          ubase + 0x3E00)); //  65504, -65504, 0, 1.5

	result[0] = h2.x | (h2.y << 16);
	result[1] = h3.x | (h3.y << 16);
	result[2] = h3.z;
	result[3] = h4.x | (h4.y << 16);
	result[4] = h4.z | (h4.w << 16);
	result[5] = asuint(f2.x + f2.y);               // -1.0
	result[6] = asuint(f3.x * f3.y);               // 2048.0
	result[7] = asuint(f3.z);                      // -0.25
	result[8] = asuint(f4.x + f4.y + f4.z + f4.w); // 1.5
}

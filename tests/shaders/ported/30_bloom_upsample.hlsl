//--name = bloom_upsample
// Bloom upsample shader using 9-tap tent filter
// Based on technique from https://www.froyok.fr/blog/2021-12-ue4-custom-bloom/

cbuffer BloomParams : register(b0) {
	float2 texel_size;    // 1.0 / source_resolution (lower mip)
	float  radius;        // Bloom radius parameter
	float  intensity;     // Blend intensity (0-1)
	uint2  output_size;   // Destination texture dimensions
	float2 _pad;
};

RWTexture2D<float4> source_tex : register(u1);  // Lower resolution mip (to upsample)
Texture2D<float4>   blend_tex  : register(t2);  // Higher resolution mip (to blend with) - sampled texture
SamplerState        blend_sampler : register(s2);  // Linear sampler for blend texture
RWTexture2D<float4> dest_tex   : register(u3);

// Helper to read with bounds checking
float4 read_source(int2 pos, int2 size) {
	pos.x = clamp(pos.x, 0, size.x - 1);
	pos.y = clamp(pos.y, 0, size.y - 1);
	return source_tex[pos];
}

// 9-tap tent filter upsampling with blend
// Sample pattern:
//   a - b - c
//   d - e - f
//   g - h - i
[numthreads(8, 8, 1)]
void cs(uint3 thread_id : SV_DispatchThreadID) {
	uint src_width, src_height;
	source_tex.GetDimensions(src_width, src_height);
	int2 src_size = int2(src_width, src_height);

	// Source and dest are the same resolution
	// Blend texture is HALF the size (from previous smaller upsample pass)
	// Apply a 3x3 blur to the source before blending
	int2 pos = int2(thread_id.xy);

	// 3x3 box filter for smoothing the source
	float4 result = float4(0, 0, 0, 0);
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			result += read_source(pos + int2(x, y), src_size);
		}
	}
	result /= 9.0;

	// Blend texture is half the resolution - use bilinear sampling for smooth upscaling
	// Convert pixel coordinates to UV coordinates (0-1 range) for the current output
	float2 uv = (float2(thread_id.xy) + 0.5) / float2(output_size);
	float4 higher_mip = blend_tex.SampleLevel(blend_sampler, uv, 0);

	// Linear blend instead of additive (key difference from traditional bloom)
	dest_tex[thread_id.xy] = lerp(higher_mip, result, intensity);
}

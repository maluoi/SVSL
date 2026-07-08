//--name = bloom_downsample
// Bloom downsample shader using 13-tap tent filter
// Based on technique from https://www.froyok.fr/blog/2021-12-ue4-custom-bloom/

cbuffer BloomParams : register(b0) {
	float2 texel_size;  // 1.0 / source_resolution
	float  radius;      // Bloom radius parameter (0.7-0.85 recommended)
	uint   _pad;
};

RWTexture2D<float4> source_tex : register(u1);  // Read-only source (using storage image)
RWTexture2D<float4> dest_tex   : register(u2);  // Write destination

// Helper function to read texture with bounds checking
float4 read_tex(int2 pos, int2 size) {
	pos.x = clamp(pos.x, 0, size.x - 1);
	pos.y = clamp(pos.y, 0, size.y - 1);
	return source_tex[pos];
}

// 13-tap tent filter downsampling using direct texture loads
// Sample pattern (each number is a tap):
//   a - b - c
//   - d - e -
//   f - g - h
//   - i - j -
//   k - l - m
[numthreads(8, 8, 1)]
void cs(uint3 thread_id : SV_DispatchThreadID) {
	// Source position at center of 2x2 block for proper downsampling
	int2 src_pos = int2(thread_id.xy) * 2 + int2(1, 1);
	uint src_width, src_height;
	source_tex.GetDimensions(src_width, src_height);
	int2 src_size = int2(src_width, src_height);

	// 13 taps arranged in a tent pattern, 1 pixel offset for proper downsampling
	float4 a = read_tex(src_pos + int2(-2, -2), src_size);
	float4 b = read_tex(src_pos + int2( 0, -2), src_size);
	float4 c = read_tex(src_pos + int2( 2, -2), src_size);

	float4 d = read_tex(src_pos + int2(-1, -1), src_size);
	float4 e = read_tex(src_pos + int2( 1, -1), src_size);

	float4 f = read_tex(src_pos + int2(-2,  0), src_size);
	float4 g = read_tex(src_pos + int2( 0,  0), src_size);
	float4 h = read_tex(src_pos + int2( 2,  0), src_size);

	float4 i = read_tex(src_pos + int2(-1,  1), src_size);
	float4 j = read_tex(src_pos + int2( 1,  1), src_size);

	float4 k = read_tex(src_pos + int2(-2,  2), src_size);
	float4 l = read_tex(src_pos + int2( 0,  2), src_size);
	float4 m = read_tex(src_pos + int2( 2,  2), src_size);

	// Weighted average (tent filter)
	// Center gets highest weight, edges get lowest
	float4 result = (d + e + i + j + g) * 0.5; // = 2.5
	result += (a + b + c + f + h + k + l + m) * 0.125; // = 1
	result /= 3.5;

	dest_tex[thread_id.xy] = result;
}

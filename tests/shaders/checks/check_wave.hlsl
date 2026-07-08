//--name = check/wave
//--wave_size = 32
// Wave/subgroup intrinsic runtime check: every op writes its result to the
// output buffer per thread, and both compilers must produce identical bits on
// the same GPU. wave_size pins the subgroup size so the wave partitioning
// can't differ between the two pipelines.

RWStructuredBuffer<uint> result : register(u0);

[numthreads(64, 1, 1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	uint i    = tid.x;
	uint base = i * 14;

	result[base + 0]  = WaveGetLaneCount();
	result[base + 1]  = WaveGetLaneIndex();
	result[base + 2]  = WaveActiveSum(i);
	result[base + 3]  = WaveActiveMin(i ^ 21u);
	result[base + 4]  = WaveActiveMax(i * 3u);
	// NOT WavePrefixSum: HLSL defines it as an exclusive scan (DXC agrees) but
	// glslang emits InclusiveScan, so the reference output is wrong — SVSL
	// deliberately diverges there. The ballot-count prefix agrees on both.
	result[base + 5]  = WaveActiveCountBits(i % 3u == 0u) + 1000u * WavePrefixCountBits(i % 2u == 0u);
	result[base + 6]  = WaveActiveBitOr(1u << (i % 32u));
	result[base + 7]  = WaveActiveBitXor(i * 2654435761u);
	result[base + 8]  = WaveActiveAnyTrue(i > 60u) ? 1u : 0u;
	result[base + 9]  = WaveActiveAllTrue(i < 128u) ? 1u : 0u;
	result[base + 10] = WaveReadLaneFirst(i + 7u);
	result[base + 11] = WaveReadLaneAt(i * 5u, 7u);
	result[base + 12] = WaveIsFirstLane() ? 1u : 0u;
	result[base + 13] = countbits(WaveActiveBallot((i % 3u) == 0u).x);
}

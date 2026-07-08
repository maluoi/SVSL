//--name = check/atomics
// Atomic runtime check: buffer + groupshared atomics with barriers. Only
// order-independent quantities are written — final values of commutative
// reductions, and the TOTAL of returned originals (which is invariant under
// thread scheduling even though each individual original is not).

RWStructuredBuffer<uint> result : register(u0);

groupshared uint gs_sum;
groupshared uint gs_or;
groupshared uint gs_max;

[numthreads(64, 1, 1)]
void cs(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
	uint i = tid.x;

	if (gi == 0) {
		for (uint k = 0; k < 16; k++) result[k] = 0;
		result[1] = 0xFFFFFFFF; // min seed
		result[4] = 0xFFFFFFFF; // and seed
		gs_sum = 0;
		gs_or  = 0;
		gs_max = 0;
	}
	AllMemoryBarrierWithGroupSync();

	InterlockedAdd(result[0], i * 3u);
	InterlockedMin(result[1], i ^ 37u);
	InterlockedMax(result[2], i * 2654435761u);
	InterlockedOr (result[3], 1u << (i % 32u));
	InterlockedAnd(result[4], ~(1u << (i % 24u)));
	InterlockedXor(result[5], i * 40503u);

	uint orig;
	InterlockedAdd(result[6], 1u, orig);
	InterlockedAdd(result[7], orig); // totals 0+1+...+63 regardless of order

	InterlockedAdd(gs_sum, i);
	InterlockedOr (gs_or,  1u << (i % 32u));
	InterlockedMax(gs_max, (i * 31u) % 97u);
	AllMemoryBarrierWithGroupSync();

	if (gi == 0) {
		result[8]  = gs_sum;
		result[9]  = gs_or;
		result[10] = gs_max;
		uint prev;
		InterlockedExchange(result[11], 4242u, prev);
		result[12] = prev;
		InterlockedExchange(result[13], 42u, prev);  // known starting value
		uint prev_hit, prev_miss;
		InterlockedCompareExchange(result[13], 42u, 777u, prev_hit);  // succeeds → 777
		InterlockedCompareExchange(result[13], 5u,  999u, prev_miss); // fails; stays 777
		// NOT InterlockedCompareStore: glslang silently compiles it to nothing,
		// so the reference output is wrong — SVSL implements it (unit-tested)
		result[14] = prev_hit + prev_miss; // 42 + 777
		result[15] = 0xC0DEC0DE;
	}
}

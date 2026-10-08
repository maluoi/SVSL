//--name = check_unroll_sroa
// Loop unrolling and array splitting (docs/PLAN_optimizer_llvm.md items 8-9).
// An [unroll] loop whose counter indexes a function-local array of 32+ elements
// unrolls, and the array splits into one variable per element - registers, not
// the scratch memory an indexed array lives in on Adreno. This exercises every
// counter shape the trip simulation handles (signed, unsigned, stepped,
// descending, nested), arrays that must stay whole (a dynamic index after the
// loops), vector and nested-array elements, and a hoisted local constant table.
// Compared bitwise against skshaderc: integer math, and floats only in exact
// small-integer arithmetic, one thread per 8-word output group, no races.

StructuredBuffer  <uint> source  : register(t0);
RWStructuredBuffer<uint> results : register(u1);

[numthreads(8, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint t    = id.x;
	uint base = t * 8;

	// signed counter: fill, then reduce
	uint a[32];
	[unroll] for (int i = 0; i < 32; i++) a[i] = source[(t * 32 + (uint)i) & 255] ^ ((uint)i * 0x9E3779B9u);
	uint sum = 0;
	[unroll] for (int i2 = 0; i2 < 32; i2++) sum += a[i2] * (uint)(i2 + 1);
	results[base + 0] = sum;

	// descending signed counter, reading the element the previous trip wrote
	uint b[33];
	b[32] = t;
	[unroll] for (int j = 31; j >= 0; j--) b[j] = b[j + 1] * 3u + a[j];
	results[base + 1] = b[0] ^ b[16];

	// unsigned counter stepping by 2, two elements per trip
	uint c[32];
	[unroll] for (uint k = 0; k < 32; k += 2) {
		c[k]     = a[k] + a[k + 1];
		c[k + 1] = a[k] - a[k + 1];
	}
	uint mix = 0;
	[unroll] for (uint k2 = 0; k2 < 32; k2++) mix = (mix << 1 | mix >> 31) ^ c[k2];
	results[base + 2] = mix;

	// nested loops over a nested array (splits one level per round)
	float grid[8][8];
	[unroll] for (int r = 0; r < 8; r++)
		[unroll] for (int q = 0; q < 8; q++) grid[r][q] = (float)((a[r * 4] >> (q * 4)) & 15u);
	float diag = 0, edge = 0;
	[unroll] for (int r2 = 0; r2 < 8; r2++) { diag += grid[r2][r2]; edge += grid[r2][7] - grid[0][r2]; }
	results[base + 3] = (uint)diag | ((uint)(edge + 128.0) << 16);

	// vector elements, reduced with dot (exact: small integers)
	float4 v[32];
	[unroll] for (int e = 0; e < 32; e++) v[e] = float4(a[e] & 7u, (a[e] >> 3) & 7u, (a[e] >> 6) & 7u, e);
	float4 acc = 0;
	[unroll] for (int e2 = 0; e2 < 32; e2++) acc += v[e2] * dot(v[e2].xyz, float3(1, 2, 4));
	results[base + 4] = (uint)acc.x ^ ((uint)acc.y << 8) ^ ((uint)acc.z << 16) ^ ((uint)acc.w << 4);

	// a local constant table: hoisted to a constant global, each read folded
	const uint K[32] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3,
	                     2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5 };
	uint dotk = 0;
	[unroll] for (int m = 0; m < 32; m++) dotk += a[m] * K[m];
	results[base + 5] = dotk ^ K[t & 31];

	// a dynamic index after the loops: d stays one indexable array
	uint d[32];
	[unroll] for (int n = 0; n < 32; n++) d[n] = a[31 - n] + (uint)n;
	results[base + 6] = d[t & 31] + d[source[t] & 31];

	// an [unroll] loop serving no array keeps its plain counted shape
	uint p = t;
	[unroll] for (int s = 0; s < 32; s++) p = p * 1664525u + 1013904223u;
	results[base + 7] = p;
}

//--name = check_static_globals
// Writable module-scope `static` globals: per-invocation Private storage. Each
// entry initializes them in declaration order before its body (runtime
// initializers included), helpers read and write them through full inlining,
// and conditional/looped writes must survive the optimizer's forwarding. One
// thread per output group, integer math only - compared bitwise against
// skshaderc. Never-initialized statics are not read: glslang leaves them
// undefined (SVSL zero-initializes, as DXC does).

cbuffer Params : register(b0) { uint4 knobs; }; // harness fill: nonzero float bits
RWStructuredBuffer<uint> results : register(u0);

struct acc_t { uint sum; uint count; };

static uint  seed    = 7;
static uint  derived = seed * 3 + 1;               // reads an earlier static
static uint  from_cb = knobs.x >> 20;              // runtime initializer (0.25f bits -> 1000)
static uint  hist[4] = { 1, 2, 3, 4 };
static acc_t acc     = { 0, 0 };

void accumulate(uint v) {
	acc.sum   += v;
	acc.count += 1;
	seed       = seed * 1103515245u + 12345u;     // helper writes a static
}

uint peek_seed() { return seed; }                  // helper reads the helper-written value

[numthreads(8, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint t    = id.x;
	uint base = t * 8;

	accumulate(t);
	accumulate(t * 2);
	results[base + 0] = acc.sum;
	results[base + 1] = acc.count;
	results[base + 2] = peek_seed();

	hist[t & 3] += 10;                              // dynamic index write
	if (t > 3) hist[0] = 100;                       // conditional write
	results[base + 3] = hist[0] + hist[1] * 2 + hist[2] * 3 + hist[3] * 4;

	for (uint i = 0; i < t; i++) derived += i;      // written in a loop
	results[base + 4] = derived;

	results[base + 5] = from_cb;
	from_cb = t;                                    // overwrite a runtime-initialized static
	results[base + 6] = from_cb + seed;
	results[base + 7] = acc.sum ^ derived;
}

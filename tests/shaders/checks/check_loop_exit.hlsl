//--name = check_loop_exit
// Loop exit tests. `for`/`while` conditions emit as glslang's exit shape (a
// conditional branch straight to the loop merge, no selection construct) -
// Adreno's compiler never finished a large [unroll] encoder in the old if/break
// shape. User-written `if (x) break;` statements keep their selection, as in
// glslang. This exercises every loop form that path touches or must leave
// alone, and is compared bitwise against skshaderc: all integer math, one
// thread per output slot group, no races.

RWStructuredBuffer<uint> results : register(u0);

// early return inside a loop: inlines as a single-trip wrapper loop whose
// body opens with a guarded break
uint first_over(uint seed, uint limit) {
	for (uint k = 0; k < 16; k++) {
		if (seed * k > limit) return k;
	}
	return 99;
}

// conditions that inline a call with an early return: the loop's exit test sits
// after the call's wrapper loop, not right after the header
bool keep_going(uint v, uint lim) { if (v > 1000) return false; return v < lim; }
uint limit(uint v)                { if (v > 5) return 5; return v + 2; }
uint first_nonzero(uint v)        { if (v == 0) return 7; return v * 2; }

[numthreads(8, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint t    = id.x;
	uint base = t * 13;

	// uint counter, literal bound: the negated test folds into the exit branch
	uint a = 0;
	[unroll] for (uint i = 0; i < 4; i++) a += i * t + 1;
	results[base + 0] = a;

	// signed counter counting down, runtime-dependent body
	int b = 0;
	for (int j = 7; j >= 0; j--) b += (j & 1) != 0 ? j : -(int)t;
	results[base + 1] = (uint)b;

	// while with a runtime condition
	uint c = t + 1;
	while (c < 100) c *= 3;
	results[base + 2] = c;

	// user-written top break: stays a selection (only loop conditions convert)
	uint d = t;
	for (;;) {
		if (d > 20) break;
		d += 7;
	}
	results[base + 3] = d;

	// a [branch] hint on the top if keeps its selection construct
	uint e = t;
	while (true) {
		[branch] if (e >= 12) break;
		e += 5;
	}
	results[base + 4] = e;

	// user-written negated break whose condition has a second user
	uint f = 0;
	for (uint m = 0; ; m++) {
		bool stop = !(m < t + 2);
		if (stop) break;
		f += stop ? 100 : m;
	}
	results[base + 5] = f;

	// continue plus a mid-body break (stays a selection)
	uint g = 0;
	for (uint n = 0; n < 10; n++) {
		if ((n & 1) == 0) continue;
		g += n;
		if (g > t * 3) break;
	}
	results[base + 6] = g;

	// nested loops, inner bound from the outer counter
	uint h = 0;
	for (uint p = 0; p < 3; p++)
		for (uint q = 0; q <= p; q++) h += p * 4 + q + t;
	results[base + 7] = h;

	// do-while with a top break (the condition is the back edge)
	uint k2 = t;
	do {
		if (k2 > 30) break;
		k2 = k2 * 2 + 1;
	} while (k2 < 50);
	results[base + 8] = k2;

	results[base + 9] = first_over(t + 1, 20);

	uint w = t;
	while (keep_going(w, 90)) w = w * 3 + 1;                    // condition inlines an early return
	results[base + 10] = w;

	uint acc = 0;
	for (uint j2 = 0; j2 < 4; j2++) acc += first_nonzero(j2 * t); // early return inside the body
	results[base + 11] = acc;

	uint s2 = 0;
	for (uint j3 = 0; j3 < limit(t); j3++) s2 += j3;              // for bound from an early-return call
	results[base + 12] = s2;
}

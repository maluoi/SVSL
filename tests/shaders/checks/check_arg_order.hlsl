//--name = check_arg_order
// Function-argument evaluation order. Arguments evaluate left to right, but a
// plain lvalue argument (`x`, `a[i]`, `s.m`, `v.x`, `v.yx`) is only addressed at
// its position; its value is read once every argument has run, as the call
// starts - so `f(x, x++)` passes the incremented x. Converted or computed
// arguments are values at their own position, index side effects run exactly
// once, `inout` copies in at call time too, and `out` writes back after the
// call. This is glslang's order (and what the author means), so the buffer is
// compared bitwise against skshaderc; the goldens are thread 0's values.

RWStructuredBuffer<uint> results : register(u0);

struct s_t { uint m; uint arr[2]; };

uint  p2(uint v, uint w)               { return v * 100 + w; }
uint  pv(uint2 v, uint w)              { return v.x * 1000 + v.y * 10 + w; }
float pf(float v, uint w)              { return v * 100 + w; }
uint  pio(inout uint v, uint w)        { uint r = v * 100 + w; v = 9; return r; }
uint  pio2(inout uint v, inout uint w) { v += 1; w += 10; return v * 100 + w; }
uint  pout(out uint v, uint w)         { v = 5; return w; }
uint  bump(inout uint v)               { v += 1; return 0; }
uint  bumpa(inout uint v[2])           { v[0] += 1; return 0; }
uint  pa(uint v[2], uint w)            { return v[0] * 100 + v[1] * 10 + w; }
uint  ps_(s_t v, uint w)               { return v.m * 100 + v.arr[1] * 10 + w; }
uint  pinout_arr(uint v[2], out uint w[2]) { w[0] = 50; w[1] = 60; return v[0] * 100 + v[1]; }

[numthreads(4, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint t    = id.x;
	uint base = t * 34;
	uint x, i;
	uint a[3] = { 1 + t, 2, 3 };

	x = 7 + t; results[base + 0] = p2(x, x++);            // lvalue read after the side effect
	x = 7 + t; results[base + 1] = p2(x + 0, x++);        // computed: value at its position
	x = 7 + t; results[base + 2] = p2(x++, x);            // side effect first, then the read
	i = 0;     results[base + 3] = p2(a[i], i++);         // index taken before i++
	x = 7 + t; results[base + 4] = p2(x, bump(x));        // later inout call
	x = 7 + t; results[base + 5] = (uint)pf(x, x++);      // converted: value at its position
	x = 7 + t; results[base + 6] = pio(x, x++); results[base + 7] = x; // inout copy-in at call time
	uint2 y = uint2(7 + t, 8);
	results[base + 8] = p2(y.x, y.x++);                   // single-component swizzle
	y = uint2(7 + t, 8); results[base + 9]  = pv(y.xy, y.x++); // multi-component swizzle
	y = uint2(7 + t, 8); results[base + 10] = pv(y, y.x++);    // whole vector
	x = 7 + t; uint r = pout(x, x++); results[base + 11] = x; results[base + 12] = r; // out wins
	x = 7 + t; results[base + 13] = pio2(x, x); results[base + 14] = x; // same lvalue twice as inout
	uint b[2] = { 1 + t, 2 };
	results[base + 15] = pa(b, b[0]++);                   // array (bound by reference)
	b[0] = 1 + t; b[1] = 2; results[base + 16] = pa(b, bumpa(b));
	s_t s; s.m = 3 + t; s.arr[0] = 4; s.arr[1] = 5;
	results[base + 17] = ps_(s, s.m++);                   // struct (bound by reference)
	s.m = 3 + t; results[base + 18] = p2(s.arr[1], s.arr[1]++);
	x = 7 + t; results[base + 19] = p2(x, p2(x, x++));    // nested call argument
	i = 0; a[0] = 1 + t; a[1] = 2; results[base + 20] = p2(a[i++], a[i]);
	x = 7 + t; results[base + 21] = pio(x, bump(x)); results[base + 22] = x;
	i = 0; a[0] = 1 + t; a[1] = 2; results[base + 23] = (uint)pf(a[i++], i); results[base + 24] = i;
	i = 0; a[0] = 1 + t; a[1] = 2; results[base + 25] = pio(a[i++], i); results[base + 26] = a[0]; results[base + 27] = i;
	uint2 z = uint2(3 + t, 4); results[base + 28] = pv(z.yx, z.y++);
	float2 fz = float2(3 + t, 4); results[base + 29] = (uint)pf(fz.y, (uint)fz.y++);
	x = 7 + t; results[base + 30] = p2(x, p2(x++, x)); results[base + 31] = x;
	b[0] = 1 + t; b[1] = 2;
	results[base + 32] = pinout_arr(b, b);                // same array as by-reference in and as out:
	results[base + 33] = b[0];                            // `in` sees call-time b, out writes back after
}

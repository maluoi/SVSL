//--name = check_array_param
// Aggregate `in` parameters. HLSL passes them by value; SVSL binds a parameter
// the callee never writes straight to the caller's storage when nothing can
// change that storage during the call, and keeps the copy otherwise. Each case
// below is observable if the wrong choice is made - notably the ones where the
// callee writes the argument's storage through another path (a global, an inout
// alias) and must still read the value from call time. Integer math, one thread
// per output group, compared bitwise against skshaderc.

struct pair_t { uint a; uint b; };

cbuffer Params  : register(b0) { uint4 knobs[2]; }; // harness fill: 0.25..1.0 float bits
cbuffer Weights : register(b2) { float wts[4]; };   // std140: 16-byte element stride
RWStructuredBuffer<uint>   results : register(u0);
RWStructuredBuffer<pair_t> pairs   : register(u1);

groupshared uint shared_vals[4];
static uint      priv_vals[4] = { 5, 6, 7, 8 };

uint sum4(uint v[4]) { return v[0] + v[1] * 2 + v[2] * 3 + v[3] * 4; }
uint sum4_nested(uint v[4]) { return sum4(v) + v[3]; }          // pass-through, still read-only

uint scramble(uint v[4]) {                                     // writes its own copy
	v[0] = v[3] * 7;
	return v[0] + v[1];
}

uint read_after_global_write(pair_t p, uint slot) {            // RW buffer argument
	pairs[slot].a = 999;                                       // clobbers the argument's storage
	return p.a + p.b;                                          // must still see the call-time value
}

uint shared_after_write(uint v[4], bool writer) {              // groupshared argument
	if (writer) shared_vals[0] = 1000;
	return v[0];
}

uint private_after_write(uint v[4]) {                          // private (static) argument
	priv_vals[1] = 2000;
	return v[1];
}

uint alias_inout(uint v[4], inout uint w[4]) {                 // same array as in and inout
	w[2] = 3000;                                               // writes back only after return
	return v[2];
}

uint from_cbuffer(uint4 k[2]) { return (k[0].x >> 20) + (k[1].y & 0xFu); }

float copy_whole(float w[4]) { float l[4] = w; l[0] += 1; return l[0] + l[2]; } // whole copy in the callee
float dyn_index(float w[4], uint i) { return w[i & 3]; }

uint lvl3(uint v[4], uint i) { return v[i & 3]; }                // three read-only levels
uint lvl2(uint v[4], uint i) { return lvl3(v, i) + v[0]; }
uint lvl1(uint v[4], uint i) { return lvl2(v, i) * 2; }

void bump4(inout uint v[4]) { v[0] += 100; }
uint written_nested(uint v[4]) { bump4(v); return v[0]; }       // written via inout: must copy

[numthreads(4, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint t    = id.x;
	uint base = t * 16;

	uint local_vals[4] = { t, t + 1, t * 3, 9 };
	results[base + 0] = sum4(local_vals);
	results[base + 1] = sum4_nested(local_vals);
	results[base + 2] = scramble(local_vals);
	results[base + 3] = local_vals[0];                         // caller's array untouched by scramble

	pairs[t].a = t + 10; pairs[t].b = t + 20;
	results[base + 4] = read_after_global_write(pairs[t], t);
	results[base + 5] = pairs[t].a;

	shared_vals[t] = t + 40;
	GroupMemoryBarrierWithGroupSync();
	uint s = shared_after_write(shared_vals, t == 0);          // thread 0 is the only writer,
	GroupMemoryBarrierWithGroupSync();                         // and its read is race-free
	results[base + 6] = t == 0 ? s : 0;

	results[base + 7] = private_after_write(priv_vals) + priv_vals[1];
	results[base + 8] = alias_inout(local_vals, local_vals) + local_vals[2];
	results[base + 9] = from_cbuffer(knobs);

	float fl[4] = { t * 0.5, 1.5, t + 0.25, 4 };
	results[base + 10] = asuint(copy_whole(fl));                 // (t*0.5 + 1) + (t + 0.25)
	results[base + 11] = asuint(dyn_index(wts, t));
	results[base + 12] = lvl1(local_vals, t);                    // local_vals[2] is 3000 by now
	results[base + 13] = written_nested(local_vals);
	results[base + 14] = local_vals[0];                          // untouched by the callee's copy
	uint grid[2][4];
	for (uint k = 0; k < 4; k++) { grid[0][k] = 0; grid[1][k] = k * t + 1; }
	results[base + 15] = sum4(grid[1]);                          // a 2D array's row
}

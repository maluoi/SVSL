//--name = check_pack_stride
// Object-form buffer whose element straddles only from element 1 onward: under
// pack1 the struct stride is 24 and float2 b sits at offset 4, so element 0's b
// (bytes 4..11) never crosses a 16-byte boundary — but element 1's b (bytes
// 28..35) crosses the 32-byte boundary. Detecting this requires walking element
// stride phases, not just element 0, so it must raise scalarBlockLayout exactly
// like the equivalent block-form 'storagebuffer { spread_t data[]; }' does.
//
// Compile+spirv-val only (no compute config): the straddle needs the
// scalarBlockLayout device feature, which svsl_view does not enable. test_sks
// asserts feature-mask bit 16 on this file.

struct spread_t {
	float  a;
	float2 b; // offset 4 — fine at element 0, straddles from element 1 (stride 24)
	float  c;
	float  d;
	float  e;
};
pack1 RWStructuredBuffer<spread_t> data : register(u0);

[numthreads(1, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	data[id.x].b = data[id.x].b.yx + data[id.x].a;
}

//--name = check_pack_scalar
// Explicit pack1 with a vector that crosses a 16-byte boundary: float4 b sits at
// offset 4, which core relaxed block layout forbids — this layout is only legal
// with the scalarBlockLayout device feature. The default (no keyword) refuses
// this struct with a compile error; the pack1 keyword is the opt-in.
//
// Deliberately compile+spirv-val only (no compute config): svsl_view's device
// does not enable scalarBlockLayout, so there is nothing to dispatch. The corpus
// test passes --scalar-block-layout to spirv-val when the program flags it, and
// test_sks asserts feature-mask bit 16 on this file.

struct wide_t {
	float  a;
	float4 b; // offset 4 under pack1 — improper straddle
};
pack1 RWStructuredBuffer<wide_t> data : register(u0);

[numthreads(1, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	data[id.x].b = data[id.x].b.yxwz + data[id.x].a;
}

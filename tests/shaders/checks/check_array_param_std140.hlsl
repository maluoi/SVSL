//--name = check_array_param_std140
// A cbuffer array passed to an array parameter by reference, where the callee
// copies the whole parameter into a local. The cbuffer array has a 16-byte
// std140 element stride; the local has 4-byte elements, so the whole-array load
// must convert layouts. Split from check_array_param because WGSL can't copy a
// whole std140-wrapped scalar array (that backend skips this shader by design).

cbuffer Weights : register(b2) { float wts[4]; }; // harness fill: 0.25..1.0
RWStructuredBuffer<uint> results : register(u0);

float copy_whole(float w[4]) { float l[4] = w; l[0] += 1; return l[0] + l[2]; }
float copy_twice(float w[4]) { return copy_whole(w) + w[3]; }   // reference passed on again

[numthreads(4, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	results[id.x * 2 + 0] = asuint(copy_whole(wts) * (id.x + 1));
	results[id.x * 2 + 1] = asuint(copy_twice(wts) + id.x);
}

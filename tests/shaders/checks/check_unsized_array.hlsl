//--name = check_unsized_array
// Regression test for unsized-array size inference. `static const float3 g[] =
// {...}` used to resolve as float3[0]: the initializer never folded, the
// Private variable was never emitted, and every use errored "global constant
// initializer is not foldable" at the use site instead of the declaration.
// The outermost dimension now takes its count from the initializer list — for
// const globals, locals, and multi-dim arrays. This shader looks a gradient up
// by a uv-derived index, so a wrong count, element order, or broken fold shows
// up as a pixel difference against skshaderc.

static const int    gradient_count = 5;
static const float3 gradient[] = { float3(0.9, 0.1, 0.1), float3(0.1, 0.9, 0.1),
                                   float3(0.065, 0.018, 0.429), float3(0.9, 0.9, 0.1),
                                   float3(0.013, 0.117, 0.597), float3(0.1, 0.9, 0.9) };
static const float  weights[][2] = { { 0.25, 0.75 }, { 0.5, 0.5 } }; // outer dim inferred

struct vsIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
struct psIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };

psIn vs(vsIn i) {
	psIn o;
	o.pos = i.pos;
	o.uv  = i.uv;
	return o;
}

float4 ps(psIn i) : SV_TARGET {
	float  pct  = saturate(i.uv.x) * gradient_count;
	int    g    = min((int)pct, gradient_count - 1); // keep g+1 in bounds at uv == 1
	float3 curr = gradient[g];
	float3 next = gradient[g + 1];
	float3 col  = lerp(curr, next, frac(pct));

	float shade[] = { 0.2, 0.4, 0.6, 0.8 };          // local size inference
	int   s       = min((int)(saturate(i.uv.y) * 4.0), 3);
	float w       = weights[s & 1][0] + weights[s & 1][1] * shade[s];
	return float4(col * w, 1);
}

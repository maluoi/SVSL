//--name = check_dynamic_index
// Regression test for dynamic vector indexing (#17). Indexing a non-addressable
// vector value with a runtime index used to spill the vector to a temporary
// variable and access-chain into it; it now lowers to a single
// OpVectorExtractDynamic on the value. This shader picks a palette channel by a
// uv-derived index, so a wrong component (or a broken spill removal) shows up as
// a pixel difference against skshaderc.

struct vsIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
struct psIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };

psIn vs(vsIn i) {
	psIn o;
	o.pos = i.pos;
	o.uv  = i.uv;
	return o;
}

float4 ps(psIn i) : SV_TARGET {
	float4 palette = float4(0.15, 0.4, 0.65, 0.9);
	int    idx     = (int)(i.uv.x * 3.0); // runtime index 0..3
	float  c       = palette[idx];        // dynamic component → OpVectorExtractDynamic
	return float4(c, c * 0.5, 1.0 - c, 1);
}

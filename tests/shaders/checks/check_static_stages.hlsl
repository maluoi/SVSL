//--name = check_static_stages
// Writable statics across a vertex + pixel pair. Each entry point initializes
// the statics it uses, in declaration order, before its body (a runtime
// initializer included); a static only one stage touches is
// declared in that stage alone; helpers inlined into both stages read and
// write them. Pixels are compared against skshaderc. No static is read before
// being written or initialized - glslang leaves those undefined. (The runtime
// initializer reads a $Global parameter: the render harness binds only $Global.)

float4 tint_in = { 0.8, 0.6, 0.4, 0.9 };            // $Global material parameter
struct vsIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
struct psIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; float4 col : COLOR0; };

static float4 tint     = saturate(tint_in * 0.5 + 0.25); // runtime initializer, both stages
static float  scale    = 0.75;
static float  wobble   = scale * 0.1;                    // reads an earlier static
static float  vs_accum = 0;                              // vertex stage only
static float2 ps_ring[3] = { float2(0.2, 0.8), float2(0.5, 0.5), float2(0.8, 0.2) }; // pixel only

float shade(float v) { scale *= 0.9; return v * scale; } // helper writes a static

psIn vs(vsIn input) {
	vs_accum += input.uv.x;
	vs_accum += input.uv.y * 0.5;
	psIn o;
	o.pos = input.pos;
	o.uv  = input.uv;
	o.col = float4(input.uv * tint.rg * shade(1.0), vs_accum * wobble * 4, 1); // gradient, per-vertex
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	float ring = 0;
	for (int i = 0; i < 3; i++) {
		float d = distance(input.uv, ps_ring[i]);
		if (d < 0.15) { ring += shade(1.0); ps_ring[i] = input.uv; } // conditional static write
	}
	float3 c = input.col.rgb * tint.a + ring * 0.5;
	return float4(c.r, c.g * scale, c.b + wobble, 1);
}

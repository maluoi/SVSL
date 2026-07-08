//--name = check_bool_param
// Regression test for bool uniforms. OpTypeBool is not allowed in externally
// visible storage (Uniform/StorageBuffer/PushConstant); bool members are laid
// out as uint 0/1 and converted at load (!= 0), matching glslang. This used to
// emit a real OpTypeBool into the $Global block, which spirv-val rejects and
// which reflected as var_int where the runtime expects var_uint. Wrong lowering
// or a bad conversion flips the branches below and shows up as a pixel diff.

struct flags_t {
	bool  invert;
	float scale;
};

bool    on_off = true;
bool2   mask;
flags_t flags;

struct vsIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
struct psIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };

psIn vs(vsIn i) {
	psIn o;
	o.pos = i.pos;
	o.uv  = i.uv;
	return o;
}

float4 ps(psIn i) : SV_TARGET {
	float3 col = float3(i.uv.x, i.uv.y, 0.5);
	if (on_off)     col  = col.bgr;
	if (mask.x)     col.r = 1 - col.r;
	if (mask.y)     col.g = 1 - col.g;
	if (flags.invert) col = 1 - col;
	return float4(col * (0.25 + flags.scale), 1); // visible even with zero defaults
}

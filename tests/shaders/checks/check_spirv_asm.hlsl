//--name = check/spirv_asm
// Inline SPIR-V escape hatch (spirv_asm). Exercises raw opcodes, a GLSL.std.450
// ext instruction, an intermediate %local reused across two instructions, and a
// helper function whose whole body is an asm block (so it must inline like any
// other call). Operands are written in binary order: $$type, %id (%result carries
// the value), $value, integer literals, and glsl450.

// a function body that is pure inline SPIR-V — inlines like any other call
float fadd(float a, float b) {
	return spirv_asm(float) {
		OpFAdd $$float %result $a $b;
	};
}

float4 ps(float2 uv : TEXCOORD0) : SV_Target {
	float s = fadd(uv.x, uv.y);

	// GLSL.std.450 Sin (13) through an ext instruction, on an SVSL value
	float sn = spirv_asm(float) {
		OpExtInst $$float %result glsl450 13 $(uv.x);
	};

	// two instructions with an intermediate %local, then a vector-scalar product
	float2 v = spirv_asm(float2) {
		OpFNegate           $$float2 %neg    $uv;
		OpVectorTimesScalar $$float2 %result %neg $s;
	};

	return float4(v.x, v.y, sn, s);
}

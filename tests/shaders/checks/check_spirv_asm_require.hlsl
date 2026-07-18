//--name = check/spirv_asm_require
// spirv_asm blocks can state their own prerequisites: OpCapability / OpExtension
// instructions are routed to the module's declaration streams instead of the
// function body. Uses SPV_KHR_shader_clock — an extension the language never
// emits itself — consumed by a real instruction, so spirv-val fails if either
// declaration goes missing. Two blocks declare the same pair to check dedup.
// Compile + spirv-val only: no desktop runtime tier for this feature.

float4 ps(float2 uv : TEXCOORD0) : SV_Target {
	// OpReadClockKHR, Scope Subgroup (constant 3)
	uint2 t0 = spirv_asm(uint2) {
		OpCapability 5055;                    // ShaderClockKHR
		OpExtension "SPV_KHR_shader_clock";
		OpReadClockKHR $$uint2 %result $(3u);
	};
	uint2 t1 = spirv_asm(uint2) {
		OpCapability 5055;                    // deduped with the block above
		OpExtension "SPV_KHR_shader_clock";
		OpReadClockKHR $$uint2 %result $(3u);
	};
	uint2 d = t1 - t0;
	return float4((float)d.x, (float)d.y, (float)uv.x, 1);
}

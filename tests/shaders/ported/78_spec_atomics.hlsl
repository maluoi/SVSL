// Specialization-constant expressions (OpSpecConstantOp keeps integer/bool
// math over spec constants specializable at pipeline creation) and float
// atomics (SPV_EXT_shader_atomic_float: add/min/max, plus core float
// exchange). glslang has no HLSL spelling for either; native SVSL.

specialization const uint32 TILE      = 16;
specialization const bool   HALF_RATE = false;

RWStructuredBuffer<float> accum;
groupshared float gs_peak;

[numthreads(64,1,1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint area   = TILE * TILE;              // OpSpecConstantOp IMul
	uint stride = HALF_RATE ? TILE / 2u : TILE; // Select over UDiv
	uint slot   = (id.x & (TILE - 1u)) % stride;

	float v = (float)id.x * 0.25;
	atomic_add(accum[slot], v);
	atomic_min(accum[area % 64u], -v);
	atomic_max(gs_peak, v);
	float prior = atomic_exchange(accum[63], v);
	atomic_add(accum[62], prior);
}

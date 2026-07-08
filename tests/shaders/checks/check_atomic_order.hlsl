//--name = check/atomic_order
// Native atomic_* with an explicit memory order (the optional trailing argument),
// and storage-inferred scope. Order names: relaxed (default), acquire, release,
// acq_rel (seq_cst is rejected — Vulkan has no sequential consistency). Relaxed
// stays bit-identical to glslang; an order adds the SPIR-V memory-semantics bits,
// and the scope follows the destination's storage (Workgroup for groupshared,
// Device for buffers).

RWStructuredBuffer<uint> buf  : register(u0);
groupshared uint         gate;

[numthreads(64, 1, 1)]
void cs(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
	uint i = tid.x;

	// relaxed (default, no order arg) — unchanged
	atomic_add(buf[0], i);

	// acquire / release on a device-visible flag → Device scope + UniformMemory
	uint taken = atomic_exchange(buf[1], 1u, acquire);
	atomic_exchange(buf[1], 0u, release);

	// acq_rel compare-exchange (native form returns the prior value)
	uint prior = atomic_compare_exchange(buf[2], 0u, i, acq_rel);

	// groupshared with acq_rel → Workgroup scope + WorkgroupMemory semantics
	atomic_or(gate, 1u << (gi % 32u), acq_rel);

	buf[3] = taken + prior + gate;
}

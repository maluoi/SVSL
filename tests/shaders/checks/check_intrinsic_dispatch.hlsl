// Broad intrinsic coverage: exercises the signed/unsigned/float ext450 variants,
// core ops, matrix ops, custom lowerings, subgroup ops + scans, Wave aliases,
// the builtin-var load, and barriers — one shader per emit-dispatch category.
// Regression for the table-driven emit dispatch (svsl_emit_ on the intrinsic row).
RWStructuredBuffer<uint> buf;

float4 ps(float4 p : SV_Position) : SV_Target {
	int   i = (int)p.x,  j = (int)p.y;
	uint  u = (uint)p.x, v = (uint)p.y;
	float a = p.x, b = p.y;

	int   si = min(i,j) + max(i,j) + clamp(i,0,j) + abs(i) + sign(i) + firstbithigh(i); // S* variants
	uint  su = min(u,v) + max(u,v) + clamp(u,1u,v) + firstbithigh(u);                   // U* variants
	float sf = min(a,b) + saturate(a) + rcp(a) + log10(a) + ldexp(a,b) + fmod(a,b)
	         + dot(p.xyz,p.yzw) + ddx(a) + ddy(a) + fwidth(a);                          // F*/core/custom
	uint  bits = asuint(a) + countbits(u) + reversebits(u) + f32tof16(a);
	float un   = f16tof32(bits);

	float3x3 m = float3x3(p.xyz, p.yzw, p.wzx);
	float3   mr = transpose(m)[0] + inverse(m)[1];                                       // matrix ext/core
	float3   sel = select(p.xyz > 0.5, p.xyz, -p.xyz) + select(a > b, mr, m[0]);          // OpSelect
	return (float)(si + (int)su + (int)bits) + sf + un + mr.x + sel.y + determinant(m);
}

[numthreads(64,1,1)]
void cs(uint3 id : SV_DispatchThreadID) {
	uint s = WaveActiveSum(id.x) + WaveActiveMin(id.x) + WaveActiveBitOr(id.x)          // Wave → subgroup
	       + WaveGetLaneCount()                                                          // builtin-var load
	       + subgroup_add(id.x) + subgroup_max(id.x) + subgroup_inclusive_add(id.x)      // native subgroup
	       + WavePrefixSum(id.x) + WaveReadLaneFirst(id.x);
	GroupMemoryBarrierWithGroupSync();                                                   // control barrier
	AllMemoryBarrier();                                                                  // memory barriers
	DeviceMemoryBarrier();
	buf[id.x] = s;
}

//--name = check/vertex_locations
// Regression test for SKS v10 vertex-input location metadata. The container
// records each input's SPIR-V location so the runtime can wire attributes by
// semantic instead of assuming array index == location. The metadata must
// mirror the module's input interface exactly:
//  - `norm` is never read: its OpVariable is stripped (matching glslang) but
//    its location is still consumed — uv lands at 2 with a hole at 1.
//  - `col` jumps to location 6 via [[vk::location]]; the recorded value must
//    be the explicit one, not the running counter's. Every semantic here is
//    one the test mesh supplies, so the visual pass renders for real.
// test_sks.c asserts the emitted records byte-for-byte against skshaderc.

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;    // unused: stripped, location 1 still consumed
	float2 uv   : TEXCOORD0;
	[[vk::location(6)]]
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos : SV_POSITION;
	float4 col : COLOR0;
};

psIn vs(vsIn input) {
	psIn o;
	o.pos = input.pos;
	o.col = input.col * float4(input.uv, 1, 1);
	return o;
}

float4 ps(psIn input) : SV_Target {
	return input.col;
}

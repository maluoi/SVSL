//--name = check_pack_layout
// The C-packed default for StructuredBuffer elements, against a pack16 buffer of
// the same struct. particle_t's C layout (id@0, pos@4, scale@12, stride 16)
// differs from std430 (pos@8, scale@16, stride 24) and from glslang's hybrid
// (DX offsets with a std430-rounded stride), so there is no reference compiler
// to compare with — goldens in compare.c are derived from the C layout by hand.
// A wrong member offset or element stride scrambles the written words.
//
// The final line rewrites one member through an access chain into the packed
// layout, so both whole-element stores and member chains are covered.

struct particle_t {
	uint   id;
	float2 pos;
	float  scale;
};
RWStructuredBuffer<particle_t>        parts   : register(u0); // C-packed default
pack16 RWStructuredBuffer<particle_t> parts16 : register(u1); // std140 elements

[numthreads(8, 1, 1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	particle_t p;
	p.id    = tid.x;
	p.pos   = float2(tid.x * 0.25, tid.x * 0.5);
	p.scale = 1.0 + tid.x;
	parts[tid.x]   = p;
	parts16[tid.x] = p;
	parts[tid.x].scale = parts[tid.x].scale * 2; // member chain through the packed layout
}

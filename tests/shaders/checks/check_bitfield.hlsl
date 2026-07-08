//--name = check/bitfield
// Explicit bit-packing intrinsics (Step 1 toward packed structs):
//   bitfield_extract(value, offset, bits)        → the [offset, offset+bits) field
//   bitfield_insert(base, value, offset, bits)   → base with that field replaced
// They map to SPIR-V OpBitFieldUExtract / OpBitFieldSExtract / OpBitFieldInsert.
// A signed value extracts with sign extension; offset/bits are uint scalars.

RWStructuredBuffer<uint> buf : register(u0);

[numthreads(1, 1, 1)]
void cs(uint3 tid : SV_DispatchThreadID) {
	// pack the 32-bit example layout: a=4@0, b=4@4, c=10@8, d=6@18, e=8@24
	uint packed = 0u;
	packed = bitfield_insert(packed, 3u,   0u,  4u);
	packed = bitfield_insert(packed, 9u,   4u,  4u);
	packed = bitfield_insert(packed, 500u, 8u,  10u);
	packed = bitfield_insert(packed, 40u,  18u, 6u);
	packed = bitfield_insert(packed, 200u, 24u, 8u);

	// unpack (unsigned)
	uint a = bitfield_extract(packed, 0u,  4u);
	uint c = bitfield_extract(packed, 8u,  10u);

	// signed extract sign-extends the field
	int  d = bitfield_extract((int)packed, 18, 6);

	// vector form is component-wise
	uint2 v = bitfield_extract(uint2(packed, packed), 24u, 8u);

	buf[tid.x] = packed + a + c + (uint)d + v.x + v.y;
}

//--name = check_bool_store
// Regression test for bool members in storage-buffer memory. OpTypeBool cannot
// live in externally visible storage, so bool/boolN struct members lay out as
// uint 0/1: stores select 1/0, loads compare != 0 (see check_bool_param for the
// cbuffer/load side). Each thread writes its own element — deterministic, so
// the buffer must match skshaderc bitwise; the final line round-trips a bool
// through a member load+store to exercise both conversions on one pointer.
//
// Member order keeps every offset alignment-neutral, so SVSL's C-packed element
// default, std430, and glslang's hybrid layout all agree byte-for-byte — this
// test stays bitwise-comparable against the reference (see check_pack_layout
// for the layouts where no reference comparison is possible).

struct rec_t {
	bool2 pair; // offset 0 under both layouts
	bool  flag; // offset 8
	float v;    // offset 12, stride 16
};
RWStructuredBuffer<rec_t> results : register(u0);

[numthreads(8, 1, 1)]
void cs(uint3 id : SV_DispatchThreadID) {
	rec_t r;
	r.flag = id.x > 3;
	r.pair = bool2(id.x > 1, id.x > 5);
	r.v    = id.x * 0.5;
	results[id.x] = r;
	results[id.x].flag = !results[id.x].flag;
}

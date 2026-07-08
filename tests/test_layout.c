// Layout engine tests: every value hand-computed, plus the stereokit_buffer
// offsets verified against skshaderc's reflection output (2026-07-04).

#include "test.h"

#include "sema/layout.h"
#include "sema/types.h"
#include "util/arena.h"

typedef struct lt_t {
	svsl_types_t   types;
	svsl_type_id_t f, f2, f3, f4, m44, m33, u, h, h4, f16, b;
} lt_t;

static lt_t lt_init(svsl_arena_t *arena) {
	lt_t lt = { .types = { .arena = arena } };
	lt.f   = svsl_type_scalar_id(&lt.types, svsl_scalar_float32);
	lt.f2  = svsl_type_vector_id(&lt.types, svsl_scalar_float32, 2);
	lt.f3  = svsl_type_vector_id(&lt.types, svsl_scalar_float32, 3);
	lt.f4  = svsl_type_vector_id(&lt.types, svsl_scalar_float32, 4);
	lt.m44 = svsl_type_matrix_id(&lt.types, svsl_scalar_float32, 4, 4);
	lt.m33 = svsl_type_matrix_id(&lt.types, svsl_scalar_float32, 3, 3);
	lt.u   = svsl_type_scalar_id(&lt.types, svsl_scalar_uint32);
	lt.h   = svsl_type_scalar_id(&lt.types, svsl_scalar_half);
	lt.h4  = svsl_type_vector_id(&lt.types, svsl_scalar_half, 4);
	lt.f16 = svsl_type_scalar_id(&lt.types, svsl_scalar_float16);
	lt.b   = svsl_type_scalar_id(&lt.types, svsl_scalar_bool);
	return lt;
}

static void test_layout_interning(void) {
	svsl_arena_t arena = {0};
	lt_t         lt    = lt_init(&arena);

	// interning dedups: same request → same id
	TEST_CHECK(svsl_type_vector_id(&lt.types, svsl_scalar_float32, 4) == lt.f4);
	TEST_CHECK(svsl_type_scalar_id(&lt.types, svsl_scalar_half) == lt.h);
	TEST_CHECK(lt.f4 != lt.h4);

	svsl_type_id_t arr_a = svsl_type_array_id(&lt.types, lt.f4, 6);
	svsl_type_id_t arr_b = svsl_type_array_id(&lt.types, lt.f4, 6);
	svsl_type_id_t arr_c = svsl_type_array_id(&lt.types, lt.f4, 7);
	TEST_CHECK(arr_a == arr_b && arr_a != arr_c);

	// name parsing
	svsl_scalar_ s; int32_t rows, cols;
	TEST_CHECK(svsl_scalar_name_parse(svsl_str("float4x4"), &s, &rows, &cols) &&
	           s == svsl_scalar_float32 && rows == 4 && cols == 4);
	TEST_CHECK(svsl_scalar_name_parse(svsl_str("min16float3"), &s, &rows, &cols) &&
	           s == svsl_scalar_half && rows == 3 && cols == 0);
	TEST_CHECK(svsl_scalar_name_parse(svsl_str("int8"), &s, &rows, &cols) &&
	           s == svsl_scalar_int8 && rows == 0);
	TEST_CHECK(svsl_scalar_name_parse(svsl_str("uint"), &s, &rows, &cols) && s == svsl_scalar_uint32);
	TEST_CHECK(!svsl_scalar_name_parse(svsl_str("float5"), &s, &rows, &cols));
	TEST_CHECK(!svsl_scalar_name_parse(svsl_str("floaty"), &s, &rows, &cols));

	svsl_arena_free(&arena);
}

static void test_layout_scalars_vectors(void) {
	svsl_arena_t arena = {0};
	lt_t         lt    = lt_init(&arena);
	svsl_types_t *t    = &lt.types;

	// scalars: same in every layout
	for (svsl_layout_ l = svsl_layout_pack1; l <= svsl_layout_pack16; l++) {
		TEST_CHECK(svsl_layout_size (t, lt.f, l) == 4 && svsl_layout_align(t, lt.f, l) == 4);
		TEST_CHECK(svsl_layout_size (t, lt.h, l) == 4);  // half is 4 bytes, always
		TEST_CHECK(svsl_layout_size (t, lt.f16, l) == 2); // float16 is really 2
		TEST_CHECK(svsl_layout_size (t, lt.b, l) == 4);   // bool is 32-bit in buffers
	}

	// vectors: pack16 = std140
	TEST_CHECK(svsl_layout_size(t, lt.f2, svsl_layout_pack16) == 8  && svsl_layout_align(t, lt.f2, svsl_layout_pack16) == 8);
	TEST_CHECK(svsl_layout_size(t, lt.f3, svsl_layout_pack16) == 12 && svsl_layout_align(t, lt.f3, svsl_layout_pack16) == 16);
	TEST_CHECK(svsl_layout_size(t, lt.f4, svsl_layout_pack16) == 16 && svsl_layout_align(t, lt.f4, svsl_layout_pack16) == 16);
	TEST_CHECK(svsl_layout_size(t, lt.h4, svsl_layout_pack16) == 16); // half4 = float4 size

	// pack1: scalar alignment
	TEST_CHECK(svsl_layout_align(t, lt.f3, svsl_layout_pack1) == 4 && svsl_layout_size(t, lt.f3, svsl_layout_pack1) == 12);
	TEST_CHECK(svsl_layout_align(t, lt.f4, svsl_layout_pack1) == 4);

	// pack8: min(natural, 8)
	TEST_CHECK(svsl_layout_align(t, lt.f2, svsl_layout_pack8) == 8);
	TEST_CHECK(svsl_layout_align(t, lt.f3, svsl_layout_pack8) == 8);
	TEST_CHECK(svsl_layout_align(t, lt.f4, svsl_layout_pack8) == 8);

	svsl_arena_free(&arena);
}

static void test_layout_matrices_arrays(void) {
	svsl_arena_t arena = {0};
	lt_t         lt    = lt_init(&arena);
	svsl_types_t *t    = &lt.types;

	// matrices, column-major
	TEST_CHECK(svsl_layout_size(t, lt.m44, svsl_layout_pack16) == 64 && svsl_layout_align(t, lt.m44, svsl_layout_pack16) == 16);
	TEST_CHECK(svsl_layout_size(t, lt.m33, svsl_layout_pack16) == 48); // 3 columns, stride 16
	TEST_CHECK(svsl_layout_size(t, lt.m33, svsl_layout_pack1)  == 36); // 3 columns, stride 12
	TEST_CHECK(svsl_layout_align(t, lt.m33, svsl_layout_pack1) == 4);

	// float3x4 (3 rows, 4 cols): 4 columns of float3
	svsl_type_id_t m34 = svsl_type_matrix_id(t, svsl_scalar_float32, 3, 4);
	TEST_CHECK(svsl_layout_size(t, m34, svsl_layout_pack16) == 64); // 4 * 16
	TEST_CHECK(svsl_layout_size(t, m34, svsl_layout_pack1)  == 48); // 4 * 12

	// arrays
	svsl_type_id_t f_arr3   = svsl_type_array_id(t, lt.f, 3);
	svsl_type_id_t f4_arr7  = svsl_type_array_id(t, lt.f4, 7);
	svsl_type_id_t m44_arr6 = svsl_type_array_id(t, lt.m44, 6);
	TEST_CHECK(svsl_layout_array_stride(t, lt.f, svsl_layout_pack16) == 16); // std140 rounds to 16
	TEST_CHECK(svsl_layout_size(t, f_arr3,   svsl_layout_pack16) == 48);
	TEST_CHECK(svsl_layout_size(t, f_arr3,   svsl_layout_pack1)  == 12); // tight
	TEST_CHECK(svsl_layout_size(t, f4_arr7,  svsl_layout_pack16) == 112);
	TEST_CHECK(svsl_layout_size(t, m44_arr6, svsl_layout_pack16) == 384);
	TEST_CHECK(svsl_layout_size(t, svsl_type_array_id(t, lt.f, 0), svsl_layout_pack16) == 0); // runtime-sized

	svsl_arena_free(&arena);
}

static void test_layout_stereokit_buffer(void) {
	svsl_arena_t arena = {0};
	lt_t         lt    = lt_init(&arena);
	svsl_types_t *t    = &lt.types;

	svsl_type_id_t m44x6 = svsl_type_array_id(t, lt.m44, 6);
	svsl_type_id_t f4x7  = svsl_type_array_id(t, lt.f4, 7);
	svsl_type_id_t f4x6  = svsl_type_array_id(t, lt.f4, 6);
	svsl_type_id_t f4x2  = svsl_type_array_id(t, lt.f4, 2);

	// the actual stereokit_buffer, verified against skshaderc reflection output
	svsl_member_t members[] = {
		{ .name = svsl_str("sk_view"),        .type = m44x6, .explicit_offset = -1 },
		{ .name = svsl_str("sk_proj"),        .type = m44x6, .explicit_offset = -1 },
		{ .name = svsl_str("sk_proj_inv"),    .type = m44x6, .explicit_offset = -1 },
		{ .name = svsl_str("sk_viewproj"),    .type = m44x6, .explicit_offset = -1 },
		{ .name = svsl_str("sk_lighting_sh"), .type = f4x7,  .explicit_offset = -1 },
		{ .name = svsl_str("sk_camera_pos"),  .type = f4x6,  .explicit_offset = -1 },
		{ .name = svsl_str("sk_camera_dir"),  .type = f4x6,  .explicit_offset = -1 },
		{ .name = svsl_str("sk_fingertip"),   .type = f4x2,  .explicit_offset = -1 },
		{ .name = svsl_str("sk_cubemap_i"),   .type = lt.f4, .explicit_offset = -1 },
		{ .name = svsl_str("sk_screen_size"), .type = lt.f4, .explicit_offset = -1 },
		{ .name = svsl_str("sk_time"),        .type = lt.f,  .explicit_offset = -1 },
		{ .name = svsl_str("sk_view_count"),  .type = lt.u,  .explicit_offset = -1 },
		{ .name = svsl_str("sk_eye_offset"),  .type = lt.u,  .explicit_offset = -1 },
	};
	uint32_t offsets[13];
	int32_t  bad  = -1;
	uint32_t size = svsl_layout_members(t, members, 13, svsl_layout_pack16, offsets, &bad);

	TEST_CHECK(bad == -1);
	static const uint32_t expected[13] = {
		0, 384, 768, 1152, 1536, 1648, 1744, 1840, 1872, 1888, 1904, 1908, 1912 };
	for (int32_t i = 0; i < 13; i++)
		TEST_CHECK(offsets[i] == expected[i]);
	TEST_CHECK(size == 1920); // skshaderc reports 1920 bytes

	svsl_arena_free(&arena);
}

static void test_layout_structs_and_offsets(void) {
	svsl_arena_t arena = {0};
	lt_t         lt    = lt_init(&arena);
	svsl_types_t *t    = &lt.types;

	// inst_t { float4x4 world; float4 color; } — the instancing struct
	svsl_struct_info_t inst = { .name = svsl_str("inst_t") };
	svsl_array_push(&arena, &inst.members, (svsl_member_t){ .name = svsl_str("world"), .type = lt.m44, .explicit_offset = -1 });
	svsl_array_push(&arena, &inst.members, (svsl_member_t){ .name = svsl_str("color"), .type = lt.f4,  .explicit_offset = -1 });
	svsl_array_push(&arena, &t->structs, inst);
	svsl_type_id_t inst_id = svsl_type_intern(t, (svsl_type_t){ .kind = svsl_type_struct, .struct_index = 0 });

	TEST_CHECK(svsl_layout_size(t, inst_id, svsl_layout_pack16) == 80); // structured-buffer stride
	TEST_CHECK(svsl_layout_size(t, inst_id, svsl_layout_pack1)  == 80); // same here: all 16-multiples

	// std140: struct with a trailing float still pads to 16
	svsl_struct_info_t padded = { .name = svsl_str("padded_t") };
	svsl_array_push(&arena, &padded.members, (svsl_member_t){ .name = svsl_str("v"), .type = lt.f3, .explicit_offset = -1 });
	svsl_array_push(&arena, &padded.members, (svsl_member_t){ .name = svsl_str("s"), .type = lt.f,  .explicit_offset = -1 });
	svsl_array_push(&arena, &padded.members, (svsl_member_t){ .name = svsl_str("w"), .type = lt.f,  .explicit_offset = -1 });
	svsl_array_push(&arena, &t->structs, padded);
	svsl_type_id_t padded_id = svsl_type_intern(t, (svsl_type_t){ .kind = svsl_type_struct, .struct_index = 1 });

	uint32_t offs[3];
	int32_t  bad;
	svsl_layout_members(t, padded.members.items, 3, svsl_layout_pack16, offs, &bad);
	TEST_CHECK(offs[0] == 0 && offs[1] == 12 && offs[2] == 16); // float packs into vec3's tail
	TEST_CHECK(svsl_layout_size(t, padded_id, svsl_layout_pack16) == 32);
	TEST_CHECK(svsl_layout_size(t, padded_id, svsl_layout_pack1)  == 20); // scalar layout is tight

	// explicit [offset(N)]: forward moves work, backward moves flag the member
	svsl_member_t explicit_members[] = {
		{ .name = svsl_str("a"), .type = lt.f, .explicit_offset = -1 },
		{ .name = svsl_str("b"), .type = lt.f, .explicit_offset = 16 },
		{ .name = svsl_str("c"), .type = lt.f, .explicit_offset = -1 },
	};
	uint32_t eoffs[3];
	svsl_layout_members(t, explicit_members, 3, svsl_layout_pack1, eoffs, &bad);
	TEST_CHECK(bad == -1);
	TEST_CHECK(eoffs[0] == 0 && eoffs[1] == 16 && eoffs[2] == 20);

	svsl_member_t bad_members[] = {
		{ .name = svsl_str("a"), .type = lt.f4, .explicit_offset = -1 },
		{ .name = svsl_str("b"), .type = lt.f,  .explicit_offset = 8 }, // backwards!
	};
	svsl_layout_members(t, bad_members, 2, svsl_layout_pack1, eoffs, &bad);
	TEST_CHECK(bad == 1);

	svsl_arena_free(&arena);
}

void test_layout(void) {
	test_layout_interning();
	test_layout_scalars_vectors();
	test_layout_matrices_arrays();
	test_layout_stereokit_buffer();
	test_layout_structs_and_offsets();
}

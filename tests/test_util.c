#include "test.h"

#include "util/arena.h"
#include "util/array.h"
#include "util/str.h"
#include "sema/types.h"

#include <stdint.h>
#include <string.h>
#include <math.h>

static void test_arena(void) {
	svsl_arena_t arena = {0};

	// Allocations are zeroed and 16-aligned
	uint8_t *a = svsl_arena_alloc(&arena, 100);
	TEST_CHECK(a != NULL);
	TEST_CHECK(((uintptr_t)a & 15) == 0);
	bool zeroed = true;
	for (int32_t i = 0; i < 100; i++)
		if (a[i] != 0) zeroed = false;
	TEST_CHECK(zeroed);

	// Consecutive allocations don't overlap
	uint8_t *b = svsl_arena_alloc(&arena, 40);
	memset(a, 0xAA, 100);
	memset(b, 0xBB, 40);
	TEST_CHECK(a[99] == 0xAA && b[0] == 0xBB);
	TEST_CHECK(((uintptr_t)b & 15) == 0);

	// Allocation larger than the default block size
	uint8_t *big = svsl_arena_alloc(&arena, 1024 * 1024);
	TEST_CHECK(big != NULL);
	big[1024 * 1024 - 1] = 7;
	TEST_CHECK(big[1024 * 1024 - 1] == 7);

	// Small allocation still works after a big one
	uint8_t *c = svsl_arena_alloc(&arena, 8);
	TEST_CHECK(c != NULL);

	// strndup copies exactly len bytes and terminates
	char *dup = svsl_arena_strndup(&arena, "hello world", 5);
	TEST_CHECK(strcmp(dup, "hello") == 0);

	svsl_arena_free(&arena);
	TEST_CHECK(arena.first == NULL && arena.last == NULL);
}

static void test_array(void) {
	svsl_arena_t arena = {0};

	svsl_array_t(int32_t) values = {0};
	for (int32_t i = 0; i < 1000; i++)
		svsl_array_push(&arena, &values, i * 3);

	TEST_CHECK(values.count == 1000);
	TEST_CHECK(values.capacity >= 1000);
	bool ordered = true;
	for (int32_t i = 0; i < 1000; i++)
		if (values.items[i] != i * 3) ordered = false;
	TEST_CHECK(ordered);

	// Struct elements with compound-literal push
	typedef struct { int32_t x; float y; } pair_t;
	svsl_array_t(pair_t) pairs = {0};
	svsl_array_push(&arena, &pairs, (pair_t){ .x = 1, .y = 2.0f });
	svsl_array_push(&arena, &pairs, (pair_t){ .x = 3, .y = 4.0f });
	TEST_CHECK(pairs.count == 2);
	TEST_CHECK(pairs.items[1].x == 3 && pairs.items[1].y == 4.0f);

	svsl_arena_free(&arena);
}

static void test_str(void) {
	svsl_str_t s = svsl_str("hello world");
	TEST_CHECK(s.len == 11);
	TEST_CHECK(svsl_str_eq_cstr(s, "hello world"));
	TEST_CHECK(!svsl_str_eq_cstr(s, "hello"));
	TEST_CHECK(svsl_str(NULL).len == 0);

	TEST_CHECK(svsl_str_eq(svsl_str(""), svsl_str(NULL)));
	TEST_CHECK(svsl_str_starts_with(s, "hello"));
	TEST_CHECK(!svsl_str_starts_with(s, "world"));
	TEST_CHECK(!svsl_str_starts_with(svsl_str("hi"), "hello"));

	svsl_str_t slice = svsl_str_slice(s, 6, 11);
	TEST_CHECK(svsl_str_eq_cstr(slice, "world"));
	TEST_CHECK(svsl_str_eq_cstr(svsl_str_slice(s, -5, 100), "hello world")); // clamped
	TEST_CHECK(svsl_str_slice(s, 8, 3).len == 0);                            // inverted → empty

	TEST_CHECK(svsl_str_eq_cstr(svsl_str_trim(svsl_str("  \t hi \r\n ")), "hi"));
	TEST_CHECK(svsl_str_trim(svsl_str("   ")).len == 0);

	TEST_CHECK(svsl_str_find_char(s, 'w') == 6);
	TEST_CHECK(svsl_str_find_char(s, 'z') == -1);
}

static void test_f16(void) {
	// basic encodings
	TEST_CHECK(svsl_f32_to_f16_bits(0.0f)  == 0x0000);
	TEST_CHECK(svsl_f32_to_f16_bits(1.0f)  == 0x3C00);
	TEST_CHECK(svsl_f32_to_f16_bits(2.0f)  == 0x4000);
	TEST_CHECK(svsl_f32_to_f16_bits(-2.0f) == 0xC000);
	TEST_CHECK(svsl_f32_to_f16_bits(INFINITY) == 0x7C00); // overflow path too: 70000 → inf
	TEST_CHECK(svsl_f32_to_f16_bits(70000.0f)  == 0x7C00);

	// round-to-nearest, non-ties: the 11th mantissa bit decides the direction
	TEST_CHECK(svsl_f32_to_f16_bits(1.0001220703125f) == 0x3C00); // 1 + 2^-13, below half → down
	TEST_CHECK(svsl_f32_to_f16_bits(1.000732421875f)  == 0x3C01); // 1 + 3·2^-12, above half → up

	// exact ties resolve to even (this is what "round-to-nearest-even" means;
	// round-half-up would give 0x3C01 / 0x3C01 here)
	TEST_CHECK(svsl_f32_to_f16_bits(1.00048828125f) == 0x3C00); // tie, LSB even → stays 0x3C00
	TEST_CHECK(svsl_f32_to_f16_bits(1.00146484375f) == 0x3C02); // tie, LSB odd  → up to 0x3C02
}

void test_util(void) {
	test_arena();
	test_array();
	test_str();
	test_f16();
}

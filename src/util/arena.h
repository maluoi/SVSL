// Arena allocator: one arena per compilation, freed all at once.
// All allocations are zeroed and 16-byte aligned. The alignment guarantee
// assumes malloc returns at least 16-byte-aligned blocks (true for every
// 64-bit target we ship on, where max_align_t is 16).

#pragma once

#include <stddef.h>

typedef struct svsl_arena_block_t svsl_arena_block_t;

typedef struct svsl_arena_t {
	svsl_arena_block_t *first;
	svsl_arena_block_t *last;
} svsl_arena_t;

// Arena starts zero-initialized: svsl_arena_t arena = {0};
void *svsl_arena_alloc  (svsl_arena_t *arena, size_t size);
char *svsl_arena_strndup(svsl_arena_t *arena, const char *str, size_t len); // copies len bytes + NUL
void  svsl_arena_free   (svsl_arena_t *arena);

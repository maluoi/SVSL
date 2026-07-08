// Typed dynamic array backed by an arena. Data-oriented: flat storage, index access.
//
//   svsl_array_t(svsl_token_t) tokens = {0};
//   svsl_array_push(&arena, &tokens, (svsl_token_t){ .kind = tok_ident });
//   tokens.items[0], tokens.count

#pragma once

#include "arena.h"

#include <stdint.h>

#define svsl_array_t(T) struct { T *items; int32_t count; int32_t capacity; }

// Internal: allocate grown storage and copy count items of item_size into it.
void *svsl_array_grow_(svsl_arena_t *arena, void *items, int32_t count, int32_t *ref_capacity, int32_t item_size);

#define svsl_array_push(arena, a, ...) ( \
	((a)->count >= (a)->capacity \
		? (void)((a)->items = svsl_array_grow_((arena), (a)->items, (a)->count, &(a)->capacity, (int32_t)sizeof(*(a)->items))) \
		: (void)0), \
	(a)->items[(a)->count++] = (__VA_ARGS__))

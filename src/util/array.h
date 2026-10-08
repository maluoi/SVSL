// Typed dynamic array backed by an arena. Data-oriented: flat storage, index access.
//
//   svsl_array_t(svsl_token_t) tokens = {0};
//   svsl_array_push(&arena, &tokens, (svsl_token_t){ .kind = tok_ident });
//   tokens.items[0], tokens.count

#pragma once

#include "arena.h"

#include <stdint.h>
#include <string.h>

#define svsl_array_t(T) struct { T *items; int32_t count; int32_t capacity; }

// Internal: allocate grown storage (at least `need` items) and copy count items of item_size into it.
void *svsl_array_grow_(svsl_arena_t *arena, void *items, int32_t count, int32_t *ref_capacity, int32_t need, int32_t item_size);

#define svsl_array_push(arena, a, ...) ( \
	((a)->count >= (a)->capacity \
		? (void)((a)->items = svsl_array_grow_((arena), (a)->items, (a)->count, &(a)->capacity, (a)->count + 1, (int32_t)sizeof(*(a)->items))) \
		: (void)0), \
	(a)->items[(a)->count++] = (__VA_ARGS__))

// Grows capacity to at least n items up front (no-op when it already has them).
#define svsl_array_reserve(arena, a, n) ( \
	(a)->capacity < (int32_t)(n) \
		? (void)((a)->items = svsl_array_grow_((arena), (a)->items, (a)->count, &(a)->capacity, (int32_t)(n), (int32_t)sizeof(*(a)->items))) \
		: (void)0)

// Appends n items copied from src: one capacity check, one memcpy.
#define svsl_array_append(arena, a, src, n) do { \
	int32_t n_ = (n); \
	if (n_ <= 0) break; \
	if ((a)->count + n_ > (a)->capacity) \
		(a)->items = svsl_array_grow_((arena), (a)->items, (a)->count, &(a)->capacity, (a)->count + n_, (int32_t)sizeof(*(a)->items)); \
	memcpy((a)->items + (a)->count, (src), (size_t)n_ * sizeof(*(a)->items)); \
	(a)->count += n_; \
} while (0)

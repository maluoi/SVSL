#include "array.h"

#include <string.h>

void *svsl_array_grow_(svsl_arena_t *arena, void *items, int32_t count, int32_t *ref_capacity, int32_t need, int32_t item_size) {
	int32_t capacity = *ref_capacity * 2;
	if (capacity < 16)   capacity = 16;
	if (capacity < need) capacity = need;

	// Fatal-OOM stance: a NULL from the arena is not handled - the memcpy below
	// dereferences it and crashes, matching how the rest of the compiler treats
	// allocation failure as unrecoverable.
	void *result = svsl_arena_alloc_raw(arena, (size_t)capacity * (size_t)item_size); // items past count are written before they are read
	if (items && count > 0)
		memcpy(result, items, (size_t)count * (size_t)item_size);
	*ref_capacity = capacity;
	return result;
}

#include "arena.h"

#include <stdlib.h>
#include <string.h>

struct svsl_arena_block_t {
	svsl_arena_block_t *next;
	size_t              used;
	size_t              capacity;
};

#define ARENA_ALIGN       16
#define ARENA_BLOCK_SIZE  (64 * 1024)
#define ARENA_HEADER_SIZE ((sizeof(svsl_arena_block_t) + ARENA_ALIGN - 1) & ~(size_t)(ARENA_ALIGN - 1))

static svsl_arena_block_t *arena_block_new(size_t min_capacity) {
	size_t              capacity = min_capacity > ARENA_BLOCK_SIZE ? min_capacity : ARENA_BLOCK_SIZE;
	svsl_arena_block_t *block    = malloc(ARENA_HEADER_SIZE + capacity);
	if (!block) return NULL;
	block->next     = NULL;
	block->used     = 0;
	block->capacity = capacity;
	return block;
}

void *svsl_arena_alloc(svsl_arena_t *arena, size_t size) {
	size = (size + ARENA_ALIGN - 1) & ~(size_t)(ARENA_ALIGN - 1);
	if (size == 0) size = ARENA_ALIGN;

	svsl_arena_block_t *block = arena->last;
	if (!block || block->used + size > block->capacity) {
		block = arena_block_new(size);
		if (!block) return NULL;
		if (arena->last) arena->last->next = block;
		else             arena->first      = block;
		arena->last = block;
	}

	void *result = (char *)block + ARENA_HEADER_SIZE + block->used;
	block->used += size;
	memset(result, 0, size);
	return result;
}

char *svsl_arena_strndup(svsl_arena_t *arena, const char *str, size_t len) {
	char *result = svsl_arena_alloc(arena, len + 1);
	if (!result) return NULL;
	if (len) memcpy(result, str, len); // str may be NULL when len is 0
	result[len] = '\0';
	return result;
}

void svsl_arena_free(svsl_arena_t *arena) {
	svsl_arena_block_t *block = arena->first;
	while (block) {
		svsl_arena_block_t *next = block->next;
		free(block);
		block = next;
	}
	arena->first = NULL;
	arena->last  = NULL;
}

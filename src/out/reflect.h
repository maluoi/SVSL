// Reflection output: human-readable table for `svslc -r` (no JSON, anywhere).

#pragma once

#include "../sema/sema.h"
#include "../util/arena.h"

// Renders the program's reflection as text, NUL-terminated, arena-owned.
const char *svsl_reflect_print(svsl_arena_t *arena, const svsl_program_t *prog);

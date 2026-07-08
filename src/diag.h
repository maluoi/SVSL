// Diagnostics: every diagnostic carries a source location that survives preprocessing.
// svsl_severity_, svsl_loc_t and svsl_diag_t are public — see <svsl/svsl.h>.

#pragma once

#include "util/arena.h"

#include <svsl/svsl.h>

#include <stdbool.h>
#include <stdint.h>

typedef struct svsl_diag_list_t {
	svsl_diag_t *items;
	int32_t      count;
	int32_t      capacity;
	int32_t      error_count;
} svsl_diag_list_t;

void svsl_diag_add(svsl_arena_t *arena, svsl_diag_list_t *ref_diags, svsl_severity_ severity,
                   svsl_loc_t loc, const char *fmt, ...);

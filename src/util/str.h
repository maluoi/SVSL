// String view: pointer + length into memory owned elsewhere (usually the arena).
// Never assumes NUL termination.

#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef struct svsl_str_t {
	const char *ptr;
	int32_t     len;
} svsl_str_t;

svsl_str_t svsl_str            (const char *opt_cstr);                   // NULL → empty
bool       svsl_str_eq         (svsl_str_t a, svsl_str_t b);
bool       svsl_str_eq_cstr    (svsl_str_t s, const char *cstr);
bool       svsl_str_starts_with(svsl_str_t s, const char *prefix);
svsl_str_t svsl_str_slice      (svsl_str_t s, int32_t start, int32_t end); // clamped, end exclusive
svsl_str_t svsl_str_trim       (svsl_str_t s);                            // strip ascii whitespace both ends
int32_t    svsl_str_find_char  (svsl_str_t s, char c);                    // index or -1

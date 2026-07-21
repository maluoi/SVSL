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

// The equality pair lives here as inlines: they run per table row in the
// keyword/intrinsic/macro scans, where the call overhead itself was the cost
// and the first-character guard rejects almost every row in one compare.
static inline bool svsl_str_eq(svsl_str_t a, svsl_str_t b) {
	if (a.len != b.len) return false;
	if (a.len == 0)     return true;
	if (a.ptr[0] != b.ptr[0]) return false;
	for (int32_t i = 1; i < a.len; i++)
		if (a.ptr[i] != b.ptr[i]) return false;
	return true;
}

// compares against s.len and confirms cstr terminates there — never strlen()s
static inline bool svsl_str_eq_cstr(svsl_str_t s, const char *cstr) {
	if (s.len == 0) return cstr[0] == '\0';
	if (s.ptr[0] != cstr[0]) return false;
	for (int32_t i = 1; i < s.len; i++)
		if (cstr[i] == '\0' || cstr[i] != s.ptr[i]) return false;
	return cstr[s.len] == '\0';
}
bool       svsl_str_starts_with(svsl_str_t s, const char *prefix);
svsl_str_t svsl_str_slice      (svsl_str_t s, int32_t start, int32_t end); // clamped, end exclusive
svsl_str_t svsl_str_trim       (svsl_str_t s);                            // strip ascii whitespace both ends
int32_t    svsl_str_find_char  (svsl_str_t s, char c);                    // index or -1

#include "str.h"

#include <string.h>

svsl_str_t svsl_str(const char *opt_cstr) {
	if (!opt_cstr) return (svsl_str_t){0};
	return (svsl_str_t){ .ptr = opt_cstr, .len = (int32_t)strlen(opt_cstr) };
}

bool svsl_str_starts_with(svsl_str_t s, const char *prefix) {
	int32_t prefix_len = (int32_t)strlen(prefix);
	if (s.len < prefix_len) return false;
	return memcmp(s.ptr, prefix, (size_t)prefix_len) == 0;
}

svsl_str_t svsl_str_slice(svsl_str_t s, int32_t start, int32_t end) {
	if (start < 0)     start = 0;
	if (end   > s.len) end   = s.len;
	if (start > end)   start = end;
	return (svsl_str_t){ .ptr = s.ptr + start, .len = end - start };
}

static bool is_space(char c) {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}

svsl_str_t svsl_str_trim(svsl_str_t s) {
	while (s.len > 0 && is_space(s.ptr[0]))         { s.ptr++; s.len--; }
	while (s.len > 0 && is_space(s.ptr[s.len - 1])) { s.len--; }
	return s;
}

int32_t svsl_str_find_char(svsl_str_t s, char c) {
	for (int32_t i = 0; i < s.len; i++)
		if (s.ptr[i] == c) return i;
	return -1;
}

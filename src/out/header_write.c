#include "header_write.h"

#include "../util/array.h"

#include <stdio.h>
#include <string.h>

typedef svsl_array_t(char) text_buf_t;

const char *svsl_header_write(svsl_arena_t *arena, svsl_str_t name, svsl_sks_blob_t blob) {
	text_buf_t out = {0};
	char       line[192]; // fits "#pragma once…const unsigned char <ident up to 95>[<size>] = {"

	// sanitized identifier: sks_<name>
	char ident[96] = "sks_";
	int32_t len = 4;
	for (int32_t i = 0; i < name.len && len < (int32_t)sizeof(ident) - 1; i++) {
		char c = name.ptr[i];
		bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
		          (c >= '0' && c <= '9') || c == '_';
		ident[len++] = ok ? c : '_';
	}
	ident[len] = '\0';

	int32_t n = snprintf(line, sizeof(line), "#pragma once\n\nconst unsigned char %s[%d] = {", ident, blob.size);
	if (n > (int32_t)sizeof(line) - 1) n = (int32_t)sizeof(line) - 1; // snprintf returns the untruncated length
	for (int32_t i = 0; i < n; i++) svsl_array_push(arena, &out, line[i]);

	for (int32_t i = 0; i < blob.size; i++) {
		if ((i & 15) == 0) {
			svsl_array_push(arena, &out, '\n');
			svsl_array_push(arena, &out, '\t');
		}
		n = snprintf(line, sizeof(line), "%u,", blob.bytes[i]);
		for (int32_t k = 0; k < n; k++) svsl_array_push(arena, &out, line[k]);
	}
	const char *tail = "\n};\n";
	for (const char *c = tail; *c; c++) svsl_array_push(arena, &out, *c);

	svsl_array_push(arena, &out, '\0');
	return out.items;
}

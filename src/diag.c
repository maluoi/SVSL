#include "diag.h"
#include "util/array.h"

#include <stdarg.h>
#include <stdio.h>

// degenerate input can cascade indefinitely; past this point further errors
// carry no information and only cost time and memory
#define DIAG_MAX_ERRORS 100

void svsl_diag_add(svsl_arena_t *arena, svsl_diag_list_t *ref_diags, svsl_severity_ severity,
                   svsl_loc_t loc, const char *fmt, ...) {
	if (severity == svsl_severity_error && ref_diags->error_count >= DIAG_MAX_ERRORS) {
		if (ref_diags->error_count == DIAG_MAX_ERRORS) {
			ref_diags->error_count++;
			svsl_array_push(arena, ref_diags, (svsl_diag_t){
				.severity = svsl_severity_error,
				.loc      = loc,
				.message  = "too many errors, further diagnostics suppressed" });
		}
		ref_diags->error_count++;
		return;
	}

	va_list args;
	va_start(args, fmt);
	int32_t len = vsnprintf(NULL, 0, fmt, args);
	va_end(args);
	if (len < 0) len = 0;

	char *message = svsl_arena_alloc(arena, (size_t)len + 1);
	va_start(args, fmt);
	vsnprintf(message, (size_t)len + 1, fmt, args);
	va_end(args);

	svsl_array_push(arena, ref_diags, (svsl_diag_t){
		.severity = severity,
		.loc      = loc,
		.message  = message });
	if (severity == svsl_severity_error)
		ref_diags->error_count++;
}

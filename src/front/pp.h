// Preprocessor: text → text with a line map (every output line knows its source
// file + line), macro expansion, conditionals, includes, and //-- metadata capture.

#pragma once

#include "../diag.h"
#include "../util/arena.h"
#include "../util/str.h"

#include <svsl/svsl.h> // svsl_include_src_t, svsl_include_cb_t, svsl_define_t

#include <stdbool.h>
#include <stdint.h>

typedef struct svsl_pp_options_t {
	svsl_include_cb_t    include_cb;
	void                *include_user;
	const svsl_define_t *defines;
	int32_t              define_count;
} svsl_pp_options_t;

// One entry per output line: where it came from.
typedef struct svsl_pp_line_t {
	const char *file; // arena-owned
	int32_t     line; // 1-based
} svsl_pp_line_t;

// A //--name[:tag] = value annotation (main file only, matching skshaderc).
typedef struct svsl_pp_meta_t {
	svsl_str_t name;
	svsl_str_t tag;   // empty if none
	svsl_str_t value; // empty if none
	svsl_loc_t loc;
} svsl_pp_meta_t;

typedef struct svsl_pp_result_t {
	const char     *text;     // preprocessed source, NUL-terminated
	int32_t         text_len;
	svsl_pp_line_t *lines;    // line map, one entry per output line
	int32_t         line_count;
	svsl_pp_meta_t *metas;
	int32_t         meta_count;
} svsl_pp_result_t;

// Returns false if any error diagnostic was emitted. The result is still
// populated as far as processing got, so callers can report all diagnostics.
bool svsl_pp_run(svsl_arena_t *arena, const char *source, const char *opt_filename,
                 const svsl_pp_options_t *opt_options, svsl_pp_result_t *out_result,
                 svsl_diag_list_t *ref_diags);

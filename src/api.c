// Public API implementation (<svsl/svsl.h>): orchestrates the internal pipeline
// pp → lex → parse → sema → IR → SPIR-V and owns all of a compile's memory in one
// arena. The SKS reader lives in out/sks_read.c. This is the only translation unit
// that stitches the modules together for consumers; the CLI goes through here too.

#include <svsl/svsl.h>

#include "front/pp.h"
#include "front/lexer.h"
#include "front/parser.h"
#include "sema/sema.h"
#include "ir/ir.h"
#include "back/emit_spirv.h"
#include "out/sks_write.h"
#include "out/header_write.h"
#include "out/reflect.h"
#include "util/arena.h"
#include "util/str.h"

#include <stdlib.h>
#include <string.h>

typedef struct impl_t {
	svsl_arena_t       arena;
	svsl_diag_list_t   diags;
	svsl_program_t     program;
	svsl_ir_module_t   ir;
	svsl_spirv_blob_t *blobs; // one per IR entry point (arena-sized after ir_build)
	svsl_sks_blob_t    sks;   // memoized container; .bytes stays NULL until first serialized
	bool               have_program;
	bool               have_ir;
	svsl_result_t      result; // returned to the caller; _impl points back here
} impl_t;

static const char *arena_cstr(svsl_arena_t *arena, svsl_str_t s) {
	return svsl_arena_strndup(arena, s.ptr, (size_t)(s.len < 0 ? 0 : s.len));
}

svsl_result_t *svsl_compile(const svsl_source_t *source, const svsl_options_t *opt_options) {
	impl_t *impl = calloc(1, sizeof(impl_t));
	if (!impl) return NULL;
	svsl_arena_t *arena = &impl->arena;

	svsl_options_t opt = opt_options ? *opt_options : (svsl_options_t){0};

	// the preprocessor takes a NUL-terminated buffer; copy when the caller gave an
	// explicit length (its buffer need not be terminated)
	const char *text = source && source->text ? source->text : "";
	if (source && source->length > 0)
		text = svsl_arena_strndup(arena, source->text, (size_t)source->length);
	const char *filename = source ? source->filename : NULL;

	svsl_pp_options_t popt = { .include_cb = opt.include_cb, .include_user = opt.include_user,
	                           .defines = opt.defines, .define_count = opt.define_count };
	svsl_pp_result_t pp;
	svsl_pp_run(arena, text, filename, &popt, &pp, &impl->diags);

	svsl_token_list_t tokens = {0};
	svsl_lex(arena, &pp, &tokens, &impl->diags);
	svsl_ast_t *ast = svsl_parse(arena, &tokens, &impl->diags);

	svsl_sema_options_t sopt = { .entry_vs = opt.entry_vertex, .entry_ps = opt.entry_pixel,
	                             .entry_cs = opt.entry_compute, .half_strict16 = opt.half_strict16,
	                             .porting_hints = opt.porting_hints };
	svsl_sema_run(arena, ast, &pp, filename, &sopt, &impl->program, &impl->diags);
	impl->have_program = true;

	if (impl->diags.error_count == 0) {
		svsl_ir_build(arena, &impl->program, opt.opt_level, &impl->ir, &impl->diags);
		impl->have_ir = true;
		impl->blobs = svsl_arena_alloc(arena, sizeof(svsl_spirv_blob_t) * (impl->ir.func_count > 0 ? impl->ir.func_count : 1));
		if (impl->diags.error_count == 0)
			for (int32_t i = 0; i < impl->ir.func_count; i++)
				svsl_spirv_emit(arena, &impl->program, &impl->ir.funcs[i], &impl->blobs[i], &impl->diags);
	}

	bool                 ok          = impl->diags.error_count == 0;
	int32_t              stage_count = 0;
	svsl_stage_output_t *stages      = NULL;
	if (ok && impl->have_ir) {
		stage_count = impl->ir.func_count;
		stages = svsl_arena_alloc(arena, sizeof(svsl_stage_output_t) * (stage_count > 0 ? stage_count : 1));
		for (int32_t i = 0; i < stage_count; i++) {
			const svsl_entry_t *e = impl->ir.funcs[i].entry;
			stages[i] = (svsl_stage_output_t){
				.stage            = e->stage,
				.entry            = arena_cstr(arena, e->name),
				.spirv            = impl->blobs[i].words,
				.spirv_word_count = impl->blobs[i].word_count };
		}
	}

	int32_t warnings = 0;
	for (int32_t i = 0; i < impl->diags.count; i++)
		if (impl->diags.items[i].severity == svsl_severity_warning) warnings++;

	// svsl_diag_list_t.items is already an array of the public svsl_diag_t
	impl->result = (svsl_result_t){
		.ok               = ok,
		.error_count      = impl->diags.error_count,
		.warning_count    = warnings,
		.diagnostic_count = impl->diags.count,
		.diagnostics      = impl->diags.items,
		.stage_count      = stage_count,
		.stages           = stages,
		._impl            = impl };
	return &impl->result;
}

void svsl_result_free(svsl_result_t *result) {
	if (!result) return;
	impl_t *impl = result->_impl;
	svsl_arena_free(&impl->arena);
	free(impl);
}

svsl_bytes_t svsl_result_sks(svsl_result_t *result) {
	impl_t *impl = result->_impl;
	if (!result->ok || !impl->have_ir) return (svsl_bytes_t){0};
	if (!impl->sks.bytes) // serialize once; -sks, -h, and reflection share the blob
		svsl_sks_write(&impl->arena, &impl->program, &impl->ir, impl->blobs, &impl->sks);
	return (svsl_bytes_t){ .data = impl->sks.bytes, .size = impl->sks.size };
}

const char *svsl_result_header(svsl_result_t *result, const char *name) {
	impl_t      *impl = result->_impl;
	svsl_bytes_t sks  = svsl_result_sks(result);
	if (!sks.data) return NULL;
	return svsl_header_write(&impl->arena, svsl_str(name ? name : "shader"),
	                         (svsl_sks_blob_t){ .bytes = sks.data, .size = sks.size });
}

const char *svsl_result_reflection(svsl_result_t *result) {
	impl_t *impl = result->_impl;
	if (!result->ok || !impl->have_program) return NULL;
	return svsl_reflect_print(&impl->arena, &impl->program);
}

const char *svsl_result_ir(svsl_result_t *result) {
	impl_t *impl = result->_impl;
	if (!result->ok || !impl->have_ir) return NULL;
	return svsl_ir_dump(&impl->arena, &impl->ir, &impl->program);
}

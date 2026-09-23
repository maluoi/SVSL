// Shared helpers for emit-level tests: compile a source string to SPIR-V words
// through the full pipeline, and count instructions in the result.

#pragma once

#include "back/emit_spirv.h"
#include "front/pp.h"
#include "ir/ir.h"
#include "sema/sema.h"
#include "util/arena.h"
#include "../vendor/spirv.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct test_spv_t {
	svsl_program_t    prog;
	svsl_ir_module_t  ir;
	svsl_spirv_blob_t blobs[4]; // one per entry point, in program order
	int32_t           blob_count;
	svsl_diag_list_t  diags;
	int32_t           warnings;
	bool              ok;       // no errors, and every entry point emitted
} test_spv_t;

// Runs pp -> lex -> parse -> sema -> IR (at `level`) -> SPIR-V. Allocations come
// from `arena`; free it to release everything. `opt_path` names the source for
// diagnostics and relative includes; `opt_pp` supplies an include callback.
void test_spv_compile(svsl_arena_t *arena, const char *src, const char *opt_path,
                      const svsl_pp_options_t *opt_pp, svsl_opt_level_ level, test_spv_t *out);

// Instructions with opcode `op` whose first/second operand words (after the
// opcode word) equal op_a/op_b; a negative filter matches anything.
int32_t test_spv_count(const svsl_spirv_blob_t *b, SpvOp op, int64_t op_a, int64_t op_b);

// Word offset of the instruction whose result is `id`, or -1.
int32_t test_spv_find_def(const svsl_spirv_blob_t *b, uint32_t id);

// Loop exits in glslang's shape: conditional branches straight to a loop's merge
// from a block with no merge instruction of its own. `opt_true_to_merge` counts
// the ones whose *true* target is the merge.
int32_t test_spv_loop_exits(const svsl_spirv_blob_t *b, int32_t *opt_true_to_merge);

// OpLoads whose result is an array: whole-array copies.
int32_t test_spv_array_loads(const svsl_spirv_blob_t *b);

// OpBitcasts of a constant (a literal the front end should have typed).
int32_t test_spv_const_bitcasts(const svsl_spirv_blob_t *b);

// spirv-val on every emitted module of a compile (Vulkan 1.1, or 1.1 + SPIR-V 1.4
// when the module asks for it). True when valid - or when spirv-val isn't on PATH,
// like the corpus test.
bool test_spv_validate(const test_spv_t *spv);

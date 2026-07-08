// SPIR-V emission: one IR entry function → one SPIR-V module.

#pragma once

#include "../ir/ir.h"
#include "../sema/sema.h"
#include "../diag.h"

#include <stdint.h>

typedef struct svsl_spirv_blob_t {
	const uint32_t *words;
	int32_t         word_count;
} svsl_spirv_blob_t;

bool svsl_spirv_emit(svsl_arena_t *arena, const svsl_program_t *prog,
                     const svsl_ir_func_t *fn, svsl_spirv_blob_t *out_blob,
                     svsl_diag_list_t *ref_diags);

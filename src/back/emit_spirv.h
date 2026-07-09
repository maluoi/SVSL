// SPIR-V emission: one IR entry function → one SPIR-V module.

#pragma once

#include "../ir/ir.h"
#include "../sema/sema.h"
#include "../diag.h"

#include <stdint.h>

typedef struct svsl_spirv_blob_t {
	const uint32_t *words;
	int32_t         word_count;
	// Vertex stage only (NULL otherwise): the SPIR-V input location assigned to
	// each prog->vertex_inputs entry, or -1 if the input's OpVariable was
	// stripped (unreferenced SROA member). Recorded at decoration time so the
	// SKS metadata can mirror the module's interface exactly.
	const int32_t  *vs_input_locations;
} svsl_spirv_blob_t;

bool svsl_spirv_emit(svsl_arena_t *arena, const svsl_program_t *prog,
                     const svsl_ir_func_t *fn, svsl_spirv_blob_t *out_blob,
                     svsl_diag_list_t *ref_diags);

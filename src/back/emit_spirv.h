// SPIR-V emission: one IR entry function → one SPIR-V module.

#pragma once

#include "../ir/ir.h"
#include "../sema/sema.h"
#include "../diag.h"

#include <stdint.h>

// QCOM image-processing use classification, recorded per program resource by
// the emitter (which derives decorations and enforces exclusivity from it) and
// consumed by the SKS writer (register forms + sampler create-flag bit)
typedef enum svsl_qcom_use_ {
	svsl_qcom_use_none = 0,
	svsl_qcom_use_plain,             // ordinary sample/load/gather/query
	svsl_qcom_use_weight,            // SampleWeightedQCOM weights → WeightTextureQCOM
	svsl_qcom_use_block_match,       // block-match target/reference → BlockMatchTextureQCOM
	svsl_qcom_use_ip_sampler,        // weighted/box/block-match sampler (no decoration)
	svsl_qcom_use_bm_window_sampler, // Window-op sampler → BlockMatchSamplerQCOM
} svsl_qcom_use_;

typedef struct svsl_spirv_blob_t {
	const uint32_t *words;
	int32_t         word_count;
	// Vertex stage only (NULL otherwise): the SPIR-V input location assigned to
	// each prog->vertex_inputs entry, or -1 if the input's OpVariable was
	// stripped (unreferenced SROA member). Recorded at decoration time so the
	// SKS metadata can mirror the module's interface exactly.
	const int32_t  *vs_input_locations;
	// Per prog->resources entry: this stage's svsl_qcom_use_ classification
	const uint8_t  *qcom_res_use;
} svsl_spirv_blob_t;

bool svsl_spirv_emit(svsl_arena_t *arena, const svsl_program_t *prog,
                     const svsl_ir_func_t *fn, svsl_spirv_blob_t *out_blob,
                     svsl_diag_list_t *ref_diags);

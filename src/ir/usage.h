// Usage analysis over lowered IR: which buffers/resources/spec constants each
// stage touches, and which vertex-input members the vertex entry actually reads.
// Feeds SKS stage_bits and the used-only vertex-input reflection (matching
// skshaderc, which reflects post-optimization SPIR-V).

#pragma once

#include "ir.h"

typedef struct svsl_usage_t {
	uint8_t *buffer_stages;   // per program buffer: svsl_stage_ bits
	uint8_t *resource_stages; // per program resource
	uint8_t *spec_stages;     // per spec constant
	uint8_t *vs_input_used;   // per program vertex_input entry (declaration order)
} svsl_usage_t;

void svsl_ir_analyze_usage(svsl_arena_t *arena, const svsl_program_t *prog,
                           const svsl_ir_module_t *module, svsl_usage_t *out_usage);

// Marks which buffers/resources a single stage function references, so emit can
// skip globals the stage never touches (a fragment shader that only reads its
// color input should not declare the whole system cbuffer). Caller-allocated
// arrays sized to prog->buffers.count / prog->resources.count; set to 1 when
// used. Mirrors the buffer/resource half of the usage scan exactly, so it is a
// precise superset of what the emitted body can reference.
void svsl_ir_func_globals(const svsl_program_t *prog, const svsl_ir_func_t *fn,
                          uint8_t *out_buffer_used, uint8_t *out_resource_used);

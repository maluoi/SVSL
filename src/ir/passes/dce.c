// Dead-code elimination: liveness from side-effecting roots backward through
// args and aux operands. Everything else is killed.

#include "passes.h"
#include "../ir_operands.h"

void svsl_ir_dce(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	(void)level;
	svsl_ir_func_t *fn    = ed->fn;
	int32_t         count = fn->insts.count;
	uint8_t        *live  = svsl_arena_alloc(ed->scratch, (size_t)(count > 0 ? count : 1)); // zeroed

	for (int32_t i = 0; i < count; i++)
		if (svsl_ir_has_side_effects(&fn->insts.items[i], &prog->types)) live[i] = 1;

	// instructions only reference earlier ids, so one reverse sweep is exact
	for (int32_t i = count - 1; i >= 0; i--) {
		if (!live[i]) continue;
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		uint32_t              mask = svsl_ir_value_arg_mask(inst);
		for (int32_t a = 0; a < 4; a++)
			if ((mask & (1u << a)) && inst->args[a] < (uint32_t)count) live[inst->args[a]] = 1;
		if (svsl_ir_aux_holds_values(inst))
			for (uint32_t k = 0; k < inst->aux_count; k++) {
				uint32_t arg = fn->aux.items[inst->aux + k];
				if (arg < (uint32_t)count) live[arg] = 1;
			}
	}

	for (int32_t i = 0; i < count; i++)
		if (!live[i]) svsl_ir_edit_kill(ed, (uint32_t)i);
}

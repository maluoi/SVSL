// Dead-code elimination: liveness from side-effecting roots backward through
// args and aux operands. Dead instructions become nops so indices stay stable.

#include "../ir.h"
#include "../ir_operands.h"

#include <string.h>

bool svsl_ir_dce(svsl_arena_t *arena, svsl_ir_func_t *fn, const svsl_types_t *types) {
	int32_t  count = fn->insts.count;
	uint8_t *live  = svsl_arena_alloc(arena, (size_t)(count > 0 ? count : 1)); // zeroed

	for (int32_t i = 0; i < count; i++)
		if (svsl_ir_has_side_effects(&fn->insts.items[i], types)) live[i] = 1;

	// instructions only reference earlier ids, so reverse sweeps converge fast
	bool changed = true;
	while (changed) {
		changed = false;
		for (int32_t i = count - 1; i >= 0; i--) {
			if (!live[i]) continue;
			const svsl_ir_inst_t *inst = &fn->insts.items[i];
			uint32_t              mask = svsl_ir_value_arg_mask(inst);
			for (int32_t a = 0; a < 4; a++) {
				if (!(mask & (1u << a))) continue;
				uint32_t arg = inst->args[a];
				if (arg < (uint32_t)count && !live[arg]) { live[arg] = 1; changed = true; }
			}
			if (svsl_ir_aux_holds_values(inst)) {
				for (uint32_t k = 0; k < inst->aux_count; k++) {
					uint32_t arg = fn->aux.items[inst->aux + k];
					if (arg < (uint32_t)count && !live[arg]) { live[arg] = 1; changed = true; }
				}
			}
		}
	}

	bool nopped = false;
	for (int32_t i = 0; i < count; i++) {
		if (!live[i] && fn->insts.items[i].op != svsl_ir_nop) {
			fn->insts.items[i].op        = svsl_ir_nop;
			fn->insts.items[i].type      = SVSL_TYPE_NONE;
			fn->insts.items[i].aux_count = 0;
			nopped = true;
		}
	}
	return nopped;
}

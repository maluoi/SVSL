// See ir_operands.h. Extracted from the DCE pass so every pass agrees on the
// per-op operand layout.

#include "ir_operands.h"

uint32_t svsl_ir_value_arg_mask(const svsl_ir_inst_t *inst) {
	switch ((svsl_ir_op_)inst->op) {
	case svsl_ir_chain:
	case svsl_ir_load:
	case svsl_ir_extract:
	case svsl_ir_shuffle:
	case svsl_ir_neg: case svsl_ir_bit_not: case svsl_ir_log_not:
	case svsl_ir_convert:
	case svsl_ir_if:
	case svsl_ir_switch:
		return 0x1;
	case svsl_ir_store:
	case svsl_ir_extract_dynamic: // args[0] = vector value, args[1] = index value
	case svsl_ir_add: case svsl_ir_sub: case svsl_ir_mul:
	case svsl_ir_div: case svsl_ir_rem:
	case svsl_ir_bit_and: case svsl_ir_bit_or: case svsl_ir_bit_xor:
	case svsl_ir_shl: case svsl_ir_shr:
	case svsl_ir_eq: case svsl_ir_ne: case svsl_ir_lt: case svsl_ir_le:
	case svsl_ir_gt: case svsl_ir_ge:
	case svsl_ir_log_and: case svsl_ir_log_or:
	case svsl_ir_mat_mul:
		return 0x3;
	case svsl_ir_insert:
		return 0x5;
	case svsl_ir_bitfield_extract: // value, offset id, width id
	case svsl_ir_select:
		return 0x7;
	case svsl_ir_bitfield_insert: // base, insert, offset id, width id
		return 0xF;
	case svsl_ir_image_load:
		return 0x2;
	case svsl_ir_image_store:
	case svsl_ir_image_atomic:
		return 0x6;
	case svsl_ir_atomic:
		// cmpxchg (op 8 in args[3]'s low byte) also uses args[2]; the high byte
		// carries the memory order, so mask it off before comparing
		return (inst->args[3] & 0xFF) == 8 ? 0x7 : 0x3;
	case svsl_ir_return:
	case svsl_ir_end_loop:
		return inst->args[0] != SVSL_IR_NONE ? 0x1 : 0x0;
	default:
		return 0x0; // constants, vars, params, pointers, markers, aux-operand ops
	}
}

bool svsl_ir_aux_holds_values(const svsl_ir_inst_t *inst) {
	switch ((svsl_ir_op_)inst->op) {
	case svsl_ir_chain:
	case svsl_ir_construct:
	case svsl_ir_intrinsic:
	case svsl_ir_tex:
	case svsl_ir_spirv_asm: // aux = the $value operand ids, in binary order
		return true;
	default:
		return false; // switch case literals, shuffle indices, ...
	}
}

bool svsl_ir_intrinsic_is_pure(const svsl_ir_inst_t *inst, const svsl_types_t *types) {
	if (inst->type == SVSL_TYPE_NONE) return false;              // defensive
	return svsl_type_get(types, inst->type)->kind != svsl_type_void; // void == barrier
}

bool svsl_ir_has_side_effects(const svsl_ir_inst_t *inst, const svsl_types_t *types) {
	switch ((svsl_ir_op_)inst->op) {
	case svsl_ir_store:
	case svsl_ir_image_store:
	case svsl_ir_image_atomic:
	case svsl_ir_atomic:
	case svsl_ir_if: case svsl_ir_else: case svsl_ir_end_if:
	case svsl_ir_loop: case svsl_ir_loop_continue: case svsl_ir_end_loop:
	case svsl_ir_break: case svsl_ir_continue:
	case svsl_ir_switch: case svsl_ir_case: case svsl_ir_end_switch:
	case svsl_ir_return: case svsl_ir_discard: case svsl_ir_demote:
		return true;
	case svsl_ir_spirv_asm:
		return true; // opaque: assume side effects, never dead-code away
	case svsl_ir_intrinsic:
		return !svsl_ir_intrinsic_is_pure(inst, types); // only void barriers are roots
	default:
		return false;
	}
}

bool svsl_ir_ends_run(svsl_ir_op_ op) {
	switch (op) {
	case svsl_ir_if: case svsl_ir_else: case svsl_ir_end_if:
	case svsl_ir_loop: case svsl_ir_loop_continue: case svsl_ir_end_loop:
	case svsl_ir_break: case svsl_ir_continue:
	case svsl_ir_switch: case svsl_ir_case: case svsl_ir_end_switch:
	case svsl_ir_return: case svsl_ir_discard: case svsl_ir_demote:
		return true;
	default:
		return false;
	}
}

static uint32_t resolve(const uint32_t *remap, uint32_t id) {
	while (remap[id] != id) id = remap[id];
	return id;
}

void svsl_ir_remap_operands(svsl_ir_func_t *fn, uint32_t idx,
                            const uint32_t *remap, uint32_t count) {
	svsl_ir_inst_t *inst = &fn->insts.items[idx];
	uint32_t        mask = svsl_ir_value_arg_mask(inst);
	for (int32_t a = 0; a < 4; a++)
		if ((mask & (1u << a)) && inst->args[a] < count)
			inst->args[a] = resolve(remap, inst->args[a]);
	if (svsl_ir_aux_holds_values(inst))
		for (uint32_t k = 0; k < inst->aux_count; k++) {
			uint32_t *slot = &fn->aux.items[inst->aux + k];
			if (*slot < count) *slot = resolve(remap, *slot);
		}
}

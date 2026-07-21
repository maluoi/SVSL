// See ir_operands.h. Extracted from the DCE pass so every pass agrees on the
// per-op operand layout.

#include "ir_operands.h"

// Per-op arg mask as data; only three ops need a dynamic answer. Runs in
// every optimizer pass loop, so it stays a table load, not a branch chain.
static const uint8_t arg_mask_table[] = {
	[svsl_ir_chain]   = 0x1, [svsl_ir_load]    = 0x1, [svsl_ir_extract] = 0x1,
	[svsl_ir_shuffle] = 0x1, [svsl_ir_neg]     = 0x1, [svsl_ir_bit_not] = 0x1,
	[svsl_ir_log_not] = 0x1, [svsl_ir_convert] = 0x1, [svsl_ir_if]      = 0x1,
	[svsl_ir_switch]  = 0x1,
	[svsl_ir_store]   = 0x3, [svsl_ir_extract_dynamic] = 0x3,
	[svsl_ir_add] = 0x3, [svsl_ir_sub] = 0x3, [svsl_ir_mul] = 0x3,
	[svsl_ir_div] = 0x3, [svsl_ir_rem] = 0x3,
	[svsl_ir_bit_and] = 0x3, [svsl_ir_bit_or] = 0x3, [svsl_ir_bit_xor] = 0x3,
	[svsl_ir_shl] = 0x3, [svsl_ir_shr] = 0x3,
	[svsl_ir_eq] = 0x3, [svsl_ir_ne] = 0x3, [svsl_ir_lt] = 0x3, [svsl_ir_le] = 0x3,
	[svsl_ir_gt] = 0x3, [svsl_ir_ge] = 0x3,
	[svsl_ir_log_and] = 0x3, [svsl_ir_log_or] = 0x3, [svsl_ir_mat_mul] = 0x3,
	[svsl_ir_insert]           = 0x5,
	[svsl_ir_bitfield_extract] = 0x7, [svsl_ir_select] = 0x7,
	[svsl_ir_bitfield_insert]  = 0xF,
	[svsl_ir_image_load]  = 0x2,
	[svsl_ir_image_store] = 0x6, [svsl_ir_image_atomic] = 0x6,
	[svsl_ir_demote] = 0x0, // highest op: sizes the table over the whole enum
};

uint32_t svsl_ir_value_arg_mask(const svsl_ir_inst_t *inst) {
	switch ((svsl_ir_op_)inst->op) {
	case svsl_ir_atomic:
		// cmpxchg (op 8 in args[3]'s low byte) also uses args[2]; the high byte
		// carries the memory order, so mask it off before comparing
		return (inst->args[3] & 0xFF) == 8 ? 0x7 : 0x3;
	case svsl_ir_return:
	case svsl_ir_end_loop:
		return inst->args[0] != SVSL_IR_NONE ? 0x1 : 0x0;
	default:
		return arg_mask_table[inst->op];
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

// See ir_operands.h. Extracted from the DCE pass so every pass agrees on the
// per-op operand layout.

#include "ir_operands.h"

// Per-op arg mask as data; only three ops need a dynamic answer (inline in
// ir_operands.h - it runs in every optimizer pass loop).
const uint8_t svsl_ir_arg_mask_table_[] = {
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

#define P svsl_ir_trait_pure
#define C svsl_ir_trait_commutative
#define R svsl_ir_trait_compare
#define M svsl_ir_trait_marker
#define E svsl_ir_trait_effect
#define A svsl_ir_trait_aux_values
const uint8_t svsl_ir_traits_table_[] = {
	[svsl_ir_const]   = P, [svsl_ir_ptr] = P, [svsl_ir_chain] = P | A,
	[svsl_ir_store]   = E,
	[svsl_ir_construct] = P | A, [svsl_ir_extract] = P, [svsl_ir_insert] = P, [svsl_ir_shuffle] = P,
	[svsl_ir_extract_dynamic] = P,
	[svsl_ir_add] = P | C, [svsl_ir_sub] = P, [svsl_ir_mul] = P | C, [svsl_ir_div] = P, [svsl_ir_rem] = P,
	[svsl_ir_neg] = P, [svsl_ir_bit_not] = P, [svsl_ir_log_not] = P,
	[svsl_ir_bit_and] = P | C, [svsl_ir_bit_or] = P | C, [svsl_ir_bit_xor] = P | C,
	[svsl_ir_shl] = P, [svsl_ir_shr] = P,
	[svsl_ir_eq] = P | C | R, [svsl_ir_ne] = P | C | R,
	[svsl_ir_lt] = P | R, [svsl_ir_le] = P | R, [svsl_ir_gt] = P | R, [svsl_ir_ge] = P | R,
	[svsl_ir_log_and] = P | C, [svsl_ir_log_or] = P | C,
	[svsl_ir_select] = P, [svsl_ir_convert] = P, [svsl_ir_mat_mul] = P,
	[svsl_ir_intrinsic] = P | A, // a barrier (void result) is the exception: see svsl_ir_is_pure
	[svsl_ir_image_store] = E, [svsl_ir_image_atomic] = E, [svsl_ir_atomic] = E,
	[svsl_ir_tex] = A,
	[svsl_ir_spirv_asm] = E | A, // opaque: assume side effects; aux = its $value ids
	[svsl_ir_bitfield_extract] = P, [svsl_ir_bitfield_insert] = P,
	[svsl_ir_if] = M, [svsl_ir_else] = M, [svsl_ir_end_if] = M,
	[svsl_ir_loop] = M, [svsl_ir_loop_continue] = M, [svsl_ir_end_loop] = M,
	[svsl_ir_break] = M, [svsl_ir_continue] = M,
	[svsl_ir_switch] = M, [svsl_ir_case] = M, [svsl_ir_end_switch] = M,
	[svsl_ir_return] = M, [svsl_ir_discard] = M, [svsl_ir_demote] = M, // highest op
};
#undef P
#undef C
#undef R
#undef M
#undef E
#undef A

bool svsl_ir_intrinsic_is_pure(const svsl_ir_inst_t *inst, const svsl_types_t *types) {
	if (inst->type == SVSL_TYPE_NONE) return false;              // defensive
	return svsl_type_get(types, inst->type)->kind != svsl_type_void; // void == barrier
}

bool svsl_ir_is_pure(const svsl_ir_inst_t *inst, const svsl_types_t *types) {
	if (!(svsl_ir_traits_table_[inst->op] & svsl_ir_trait_pure)) return false;
	return inst->op != svsl_ir_intrinsic || svsl_ir_intrinsic_is_pure(inst, types);
}

bool svsl_ir_has_side_effects(const svsl_ir_inst_t *inst, const svsl_types_t *types) {
	if (svsl_ir_traits_table_[inst->op] & (svsl_ir_trait_marker | svsl_ir_trait_effect)) return true;
	return inst->op == svsl_ir_intrinsic && !svsl_ir_intrinsic_is_pure(inst, types); // barriers
}

uint32_t svsl_ir_root_ptr(const svsl_ir_func_t *fn, uint32_t p) {
	while (p < (uint32_t)fn->insts.count && fn->insts.items[p].op == svsl_ir_chain)
		p = fn->insts.items[p].args[0];
	return p;
}

bool svsl_ir_is_invocation_local(const svsl_ir_func_t *fn, uint32_t p) {
	if (p >= (uint32_t)fn->insts.count) return false;
	const svsl_ir_inst_t *in = &fn->insts.items[p];
	return in->op == svsl_ir_var ||
	       (in->op == svsl_ir_ptr && (svsl_ref_)in->args[0] == svsl_ref_private_global);
}

bool svsl_ir_forwardable_root(const svsl_ir_func_t *fn, const svsl_program_t *prog, uint32_t r) {
	const svsl_ir_inst_t *in = &fn->insts.items[r];
	if (in->op == svsl_ir_var) return true;
	if (in->op != svsl_ir_ptr) return false;
	switch ((svsl_ref_)in->args[0]) {
	case svsl_ref_buffer_member: {
		const svsl_buffer_t *b = &prog->buffers.items[in->args[1]];
		return b->kind == svsl_block_uniform || b->kind == svsl_block_pushconstant;
	}
	case svsl_ref_const_global:
	case svsl_ref_private_global: // written only by this invocation, through one canonical pointer
		return true;
	case svsl_ref_resource:
		return prog->resources.items[in->args[1]].kind == svsl_res_structured; // read-only
	case svsl_ref_stage_io: // inputs are read-only; outputs are only written, and observed at return
		return !fn->entry->io.items[in->args[1]].output;
	default:
		return false; // storage buffers, images, groupshared, builtins, ...
	}
}

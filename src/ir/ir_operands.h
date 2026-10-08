// Per-op operand layout and traits: which args[]/aux entries of an instruction
// are value references (vs. literals like an extract index or a resource id),
// and what each op is allowed to have done to it. Single source of truth
// shared by every IR pass that walks the dataflow graph. Encodes the same op
// semantics as ir.h.

#pragma once

#include "ir.h"

// Per-op traits, as data. Every pass that asks "may I merge / drop / reorder
// this?" reads the same table.
enum {
	svsl_ir_trait_pure        = 1 << 0, // a function of its operands alone: no memory read, no
	                                    // effect (CSE/DCE-able; intrinsics also need a result)
	svsl_ir_trait_commutative = 1 << 1, // args[0], args[1] swap freely (int and IEEE float alike)
	svsl_ir_trait_compare     = 1 << 2, // bool result relating args[0] and args[1]
	svsl_ir_trait_marker      = 1 << 3, // structured control flow: ends a straight-line run
	svsl_ir_trait_effect      = 1 << 4, // observable whether or not its result is used
	svsl_ir_trait_aux_values  = 1 << 5, // aux holds value ids (chain indices, construct
	                                    // components, intrinsic/tex/asm operands), not literals
};

// The tables behind the inline queries below (ir_operands.c), indexed by op.
// They run in every optimizer pass loop, so they stay table loads.
extern const uint8_t svsl_ir_traits_table_[];
extern const uint8_t svsl_ir_arg_mask_table_[];

static inline uint8_t svsl_ir_op_traits(svsl_ir_op_ op) {
	return svsl_ir_traits_table_[op];
}

// Bitmask over args[0..3]: bit a set -> args[a] is a value/pointer reference.
static inline uint32_t svsl_ir_value_arg_mask(const svsl_ir_inst_t *inst) {
	switch ((svsl_ir_op_)inst->op) {
	case svsl_ir_atomic:
		// cmpxchg (op 8 in args[3]'s low byte) also uses args[2]; the high byte
		// carries the memory order, so mask it off before comparing
		return (inst->args[3] & 0xFF) == 8 ? 0x7 : 0x3;
	case svsl_ir_end_loop: // a do-while's back-edge condition, when it has one
		return inst->args[0] != SVSL_IR_NONE ? 0x1 : 0x0;
	default:
		return svsl_ir_arg_mask_table_[inst->op];
	}
}

// True when the instruction's aux pool holds value references rather than
// literals (switch case values).
static inline bool svsl_ir_aux_holds_values(const svsl_ir_inst_t *inst) {
	return (svsl_ir_traits_table_[inst->op] & svsl_ir_trait_aux_values) != 0;
}

// True for control-flow ops that end a straight-line run.
static inline bool svsl_ir_ends_run(svsl_ir_op_ op) {
	return (svsl_ir_traits_table_[op] & svsl_ir_trait_marker) != 0;
}

// The compare relating (b, a) the way `op` relates (a, b): lt <-> gt, le <-> ge;
// eq/ne (and any non-compare) map to themselves.
static inline svsl_ir_op_ svsl_ir_mirror_compare(svsl_ir_op_ op) {
	switch (op) {
	case svsl_ir_lt: return svsl_ir_gt;
	case svsl_ir_gt: return svsl_ir_lt;
	case svsl_ir_le: return svsl_ir_ge;
	case svsl_ir_ge: return svsl_ir_le;
	default:         return op;
	}
}

// True for ops whose execution is observable regardless of whether their
// result is used (stores to observable memory, control flow, barriers, ...).
// Intrinsics are impure only when they return void - the barriers; every
// value-producing intrinsic (math, derivatives, subgroup, pack) is a pure
// function of its operands and may be eliminated when its result is dead.
bool svsl_ir_has_side_effects(const svsl_ir_inst_t *inst, const svsl_types_t *types);

// True when CSE may merge two identical instances: a pure op, and for an
// intrinsic one that yields a value (never a barrier).
bool svsl_ir_is_pure(const svsl_ir_inst_t *inst, const svsl_types_t *types);

// True when this intrinsic instruction is a pure value (safe to DCE/CSE): it
// produces a non-void result. Barriers (void) are the only impure intrinsics.
bool svsl_ir_intrinsic_is_pure(const svsl_ir_inst_t *inst, const svsl_types_t *types);

// --- pointers ------------------------------------------------------------------

// Walk chain bases down to the underlying var or ptr (global or stage-IO slot).
uint32_t svsl_ir_root_ptr(const svsl_ir_func_t *fn, uint32_t p);

// Storage only this invocation can see, and only until the entry returns: local
// vars, and private globals (one canonical pointer each - see ir_build's
// lower_private_globals). Unread stores to it are dead.
bool svsl_ir_is_invocation_local(const svsl_ir_func_t *fn, uint32_t root);

// A root whose loads/stores can be tracked soundly: distinct vars (entry
// parameters are vars too) and private globals never alias (per-invocation
// storage, no escaping pointers, one pointer instruction per private global),
// read-only globals (uniform/pushconstant buffers, const globals, read-only
// structured buffers) are never written, and stage inputs are read-only.
bool svsl_ir_forwardable_root(const svsl_ir_func_t *fn, const svsl_program_t *prog, uint32_t root);

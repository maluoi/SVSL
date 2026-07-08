// Local common-subexpression elimination (value numbering). Within a
// straight-line run, a pure value op whose (op, type, operands) exactly matches
// an earlier one reuses that earlier result. The forward-only-reference
// invariant means the earlier definition always dominates, and resetting the
// table at every control-flow boundary keeps reuse within a single region — so
// no dominance analysis is needed. Value-preserving: the reused value is
// bit-identical because the operand SSA ids are identical. See
// docs/OPTIMIZATION_PLAN.md §4 (#4, #10).
//
// Only side-effect-free, memory-independent ops participate. Loads and samples
// are memory-dependent (handled by forwarding); intrinsics are skipped until
// the table carries a purity flag (barriers/subgroup ops must never be merged).

#include "../ir.h"
#include "../ir_operands.h"

static bool cse_pure(svsl_ir_op_ op) {
	switch (op) {
	case svsl_ir_const:
	case svsl_ir_construct: case svsl_ir_extract: case svsl_ir_insert: case svsl_ir_shuffle:
	case svsl_ir_extract_dynamic:
	case svsl_ir_add: case svsl_ir_sub: case svsl_ir_mul: case svsl_ir_div: case svsl_ir_rem:
	case svsl_ir_neg: case svsl_ir_bit_not: case svsl_ir_log_not:
	case svsl_ir_bit_and: case svsl_ir_bit_or: case svsl_ir_bit_xor:
	case svsl_ir_shl: case svsl_ir_shr:
	case svsl_ir_eq: case svsl_ir_ne: case svsl_ir_lt: case svsl_ir_le:
	case svsl_ir_gt: case svsl_ir_ge:
	case svsl_ir_log_and: case svsl_ir_log_or:
	case svsl_ir_select: case svsl_ir_convert: case svsl_ir_mat_mul:
	case svsl_ir_bitfield_extract: case svsl_ir_bitfield_insert:
	case svsl_ir_ptr:       // global-address producer (buffer member, resource, …)
	case svsl_ir_chain:     // pure address computation — safe to share
	case svsl_ir_intrinsic: // only value-producing (non-void) ones; guarded below
		return true;
	default:
		return false; // load/store/tex/image/atomic/var/param/spec/undef/markers
	}
}

static uint32_t cse_hash(const svsl_ir_func_t *fn, uint32_t id) {
	const svsl_ir_inst_t *in = &fn->insts.items[id];
	uint32_t h = 2166136261u;
#define MIX(v) do { h = (h ^ (uint32_t)(v)) * 16777619u; } while (0)
	MIX(in->op);
	MIX(in->flags); // a precise op must not merge with a contractable one
	MIX(in->type);
	for (int32_t a = 0; a < 4; a++) MIX(in->args[a]);
	if (svsl_ir_aux_holds_values(in))
		for (uint32_t k = 0; k < in->aux_count; k++) MIX(fn->aux.items[in->aux + k]);
#undef MIX
	return h;
}

static bool cse_equal(const svsl_ir_func_t *fn, uint32_t a, uint32_t b) {
	const svsl_ir_inst_t *x = &fn->insts.items[a], *y = &fn->insts.items[b];
	if (x->op != y->op || x->type != y->type || x->flags != y->flags) return false;
	for (int32_t i = 0; i < 4; i++)
		if (x->args[i] != y->args[i]) return false;
	if (svsl_ir_aux_holds_values(x)) {
		if (x->aux_count != y->aux_count) return false;
		for (uint32_t k = 0; k < x->aux_count; k++)
			if (fn->aux.items[x->aux + k] != fn->aux.items[y->aux + k]) return false;
	}
	return true;
}

bool svsl_ir_cse(svsl_arena_t *arena, svsl_ir_func_t *fn, const svsl_types_t *types) {
	int32_t count = fn->insts.count;
	if (count == 0) return false;

	uint32_t cap = 16;
	while (cap < (uint32_t)count * 2) cap <<= 1;
	uint32_t  mask   = cap - 1;
	uint32_t *bucket = svsl_arena_alloc(arena, (size_t)cap * sizeof(uint32_t));
	uint32_t *stamp  = svsl_arena_alloc(arena, (size_t)cap * sizeof(uint32_t)); // zeroed → empty
	uint32_t *remap  = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t));
	for (int32_t i = 0; i < count; i++) remap[i] = (uint32_t)i;

	uint32_t gen     = 1;
	bool     changed = false;

	for (int32_t i = 0; i < count; i++) {
		svsl_ir_op_ op = (svsl_ir_op_)fn->insts.items[i].op;

		// every instruction (side-effecting ones too) must have its operands
		// redirected to surviving canonical ids before we nop the duplicates
		svsl_ir_remap_operands(fn, (uint32_t)i, remap, (uint32_t)count);

		if (svsl_ir_ends_run(op)) { gen++; continue; }   // new region → forget prior values
		if (!cse_pure(op)) continue;
		// barriers (void intrinsics) must never be merged — they are side effects
		if (op == svsl_ir_intrinsic &&
		    !svsl_ir_intrinsic_is_pure(&fn->insts.items[i], types)) continue;

		uint32_t h = cse_hash(fn, (uint32_t)i) & mask;
		for (uint32_t p = 0;; p = (p + 1) & mask) {
			uint32_t s = (h + p) & mask;
			if (stamp[s] != gen) {                       // empty slot → first of its kind
				stamp[s]  = gen;
				bucket[s] = (uint32_t)i;
				break;
			}
			if (cse_equal(fn, bucket[s], (uint32_t)i)) {  // redundant → reuse earlier value
				remap[i] = bucket[s];
				changed  = true;
				break;
			}
		}
	}

	if (!changed) return false;
	for (int32_t i = 0; i < count; i++) {
		if (remap[i] != (uint32_t)i) {
			svsl_ir_inst_t *inst = &fn->insts.items[i];
			inst->op        = svsl_ir_nop;
			inst->type      = SVSL_TYPE_NONE;
			inst->aux_count = 0;
		}
	}
	return true;
}

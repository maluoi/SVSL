// Common-subexpression elimination (value numbering, scoped like LLVM's
// EarlyCSE). A pure value op whose (op, type, operands) matches an earlier one
// reuses that earlier result - wherever the earlier one dominates. In this
// structured IR the dominator tree is region nesting, so each table entry is
// stamped with the arm it was made in (svsl_ir_scope_t): values from enclosing
// arms are visible inside nested ones and after they close, while a sibling arm
// (else, continue section, next case) never sees them. No dominance analysis,
// no deletion: a stale slot is simply overwritten. Value-preserving: the reused
// value is bit-identical because the operand ids are identical. See
// docs/OPTIMIZATION_PLAN.md section 4 (#4, #10) and docs/PLAN_optimizer_llvm.md item 3.
//
// Only side-effect-free, memory-independent ops participate (svsl_ir_is_pure).
// Loads and samples are memory-dependent (handled by forwarding); barriers are
// effects and never merge.

#include "passes.h"
#include "../ir_cf.h"
#include "../ir_operands.h"

// The instruction's key: op and first two args, with the operands of a
// commutative op or compare in a fixed order (a compare mirrors its relation
// when they swap), so `a+b` meets `b+a` and `x<y` meets `y>x` (EarlyCSE).
typedef struct cse_key_t { uint8_t op; uint32_t a0, a1; } cse_key_t;

static cse_key_t cse_key(const svsl_ir_inst_t *in) {
	cse_key_t k = { in->op, in->args[0], in->args[1] };
	if ((svsl_ir_op_traits((svsl_ir_op_)in->op) & (svsl_ir_trait_commutative | svsl_ir_trait_compare)) &&
	    k.a0 > k.a1)
		k = (cse_key_t){ (uint8_t)svsl_ir_mirror_compare((svsl_ir_op_)in->op), in->args[1], in->args[0] };
	return k;
}

static uint32_t cse_hash(const svsl_ir_func_t *fn, uint32_t id) {
	const svsl_ir_inst_t *in = &fn->insts.items[id];
	cse_key_t             k  = cse_key(in);
	uint32_t h = 2166136261u;
#define MIX(v) do { h = (h ^ (uint32_t)(v)) * 16777619u; } while (0)
	MIX(k.op);
	MIX(in->flags); // a precise op must not merge with a contractable one
	MIX(in->type);
	MIX(k.a0); MIX(k.a1); MIX(in->args[2]); MIX(in->args[3]);
	if (svsl_ir_aux_holds_values(in))
		for (uint32_t i = 0; i < in->aux_count; i++) MIX(fn->aux.items[in->aux + i]);
#undef MIX
	return h;
}

static bool cse_equal(const svsl_ir_func_t *fn, uint32_t a, uint32_t b) {
	const svsl_ir_inst_t *x = &fn->insts.items[a], *y = &fn->insts.items[b];
	cse_key_t             kx = cse_key(x), ky = cse_key(y);
	if (kx.op != ky.op || kx.a0 != ky.a0 || kx.a1 != ky.a1) return false;
	if (x->type != y->type || x->flags != y->flags) return false;
	if (x->args[2] != y->args[2] || x->args[3] != y->args[3]) return false;
	if (svsl_ir_aux_holds_values(x)) {
		if (x->aux_count != y->aux_count) return false;
		for (uint32_t k = 0; k < x->aux_count; k++)
			if (fn->aux.items[x->aux + k] != fn->aux.items[y->aux + k]) return false;
	}
	return true;
}

void svsl_ir_cse(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	(void)level;
	svsl_ir_func_t *fn    = ed->fn;
	int32_t         count = fn->insts.count;
	if (count == 0) return;

	uint32_t cap = 16;
	while (cap < (uint32_t)count * 2) cap <<= 1;
	uint32_t  mask   = cap - 1;
	uint32_t *bucket = svsl_arena_alloc(ed->scratch, (size_t)cap * sizeof(uint32_t)); // id + 1; 0 = empty
	uint32_t *serial = svsl_arena_alloc(ed->scratch, (size_t)cap * sizeof(uint32_t)); // the entry's arm
	int32_t  *depth  = svsl_arena_alloc(ed->scratch, (size_t)cap * sizeof(int32_t));
	svsl_ir_scope_t scope;
	svsl_ir_scope_begin(&scope, ed->scratch, fn);
	ed->resolved = true; // every instruction's operands resolve below, in order

	for (int32_t i = 0; i < count; i++) {
		// hash on canonical operands: an earlier duplicate's users now name the survivor
		if (ed->replaced) svsl_ir_edit_resolve_operands(ed, (uint32_t)i);

		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		svsl_ir_scope_step(&scope, (svsl_ir_op_)inst->op);
		if (!svsl_ir_is_pure(inst, &prog->types)) continue; // barriers are never merged

		uint32_t h     = cse_hash(fn, (uint32_t)i) & mask;
		uint32_t reuse = SVSL_IR_NONE; // first stale slot on the probe path
		for (uint32_t p = 0;; p = (p + 1) & mask) {
			uint32_t s = (h + p) & mask;
			if (!bucket[s]) {                                   // end of the chain: first of its kind
				if (reuse == SVSL_IR_NONE) reuse = s;
				bucket[reuse] = (uint32_t)i + 1;
				serial[reuse] = scope.serial[scope.depth];
				depth[reuse]  = scope.depth;
				break;
			}
			if (!svsl_ir_scope_live(&scope, depth[s], serial[s])) { // out of scope: reusable
				if (reuse == SVSL_IR_NONE) reuse = s;
				continue;
			}
			if (cse_equal(fn, bucket[s] - 1, (uint32_t)i)) {    // dominating duplicate -> reuse it
				svsl_ir_edit_replace(ed, (uint32_t)i, bucket[s] - 1);
				break;
			}
		}
	}
}

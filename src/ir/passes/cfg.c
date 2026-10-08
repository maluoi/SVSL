// Structured CFG simplification: LLVM's SimplifyCFG (constant branches, empty
// arms, unreachable code, single-iteration loops) and SCCP's dead edges, on
// SVSL's marker form. Every rule only removes markers or whole arms, so the
// structure stays balanced and every surviving value still dominates its uses
// (values never escape the arm that computes them; see ir.h).
//
//   * if on a constant: the live arm stays, its markers and the dead arm go.
//     A for/while exit test (svsl_ir_flag_loop_exit) on false never exits there
//     and goes; on true, a loop with no effect ahead of the test runs zero
//     times and goes whole - otherwise its break becomes unconditional.
//   * empty arms: `if c {}` and `if c {} else {}` vanish, and an empty else
//     loses its marker.
//   * code after break/continue/return/discard, up to the end of its arm.
//   * a loop whose body always ends in break/return/discard and that nothing
//     else breaks out of or continues runs once: its markers go (the wrapper
//     loops inlining makes for early returns, once their other exits fold away).
//   * if-conversion (LLVM's SpeculativelyExecuteBB / FoldTwoEntryPHINode): an
//     innermost if whose arms only compute values and store to invocation-local
//     memory runs both arms unconditionally and merges each stored location
//     with a select. See if_convert.
// See docs/PLAN_optimizer_llvm.md items 2 and 4.

#include "passes.h"
#include "../ir_cf.h"
#include "../ir_const.h"
#include "../ir_operands.h"
#include "../../tables/intrinsics.h"
#include "../../../vendor/spirv.h"

static bool is_dead(const svsl_ir_edit_t *ed, uint32_t i) {
	return ed->fn->insts.items[i].op == svsl_ir_nop || svsl_ir_edit_resolve(ed, i) == SVSL_IR_NONE;
}

// kills every live instruction in [lo, hi]
static void kill_range(svsl_ir_edit_t *ed, uint32_t lo, uint32_t hi) {
	for (uint32_t k = lo; k <= hi; k++)
		if (!is_dead(ed, k)) svsl_ir_edit_kill(ed, k);
}

// no live instruction strictly between lo and hi
static bool range_empty(const svsl_ir_edit_t *ed, uint32_t lo, uint32_t hi) {
	for (uint32_t k = lo + 1; k < hi; k++)
		if (!is_dead(ed, k)) return false;
	return true;
}

// nothing strictly between lo and hi has an effect (it only computes values)
static bool range_effect_free(const svsl_ir_edit_t *ed, const svsl_types_t *types, uint32_t lo, uint32_t hi) {
	for (uint32_t k = lo + 1; k < hi; k++)
		if (!is_dead(ed, k) && svsl_ir_has_side_effects(&ed->fn->insts.items[k], types)) return false;
	return true;
}

static bool const_bool(const svsl_ir_edit_t *ed, const svsl_types_t *types, uint32_t id, bool *out) {
	const svsl_ir_inst_t *c = svsl_ir_edit_inst(ed, id);
	if (c->op != svsl_ir_const) return false;
	const svsl_type_t *t = svsl_type_get(types, c->type);
	if (t->kind != svsl_type_scalar || t->scalar != svsl_scalar_bool) return false;
	*out = (c->args[0] & 1) != 0;
	return true;
}

static void simplify_if(svsl_ir_edit_t *ed, const svsl_ir_cf_t *cf, const svsl_types_t *types, uint32_t i) {
	svsl_ir_func_t *fn       = ed->fn;
	svsl_ir_inst_t *in       = &fn->insts.items[i];
	uint32_t        end      = cf->end[i];
	uint32_t        mid      = cf->arm_end[i];
	bool            has_else = mid != end;
	bool            c;

	if (const_bool(ed, types, in->args[0], &c)) {
		if (in->flags & svsl_ir_flag_loop_exit) {
			if (!c) { kill_range(ed, i, end); return; } // never exits here
			uint32_t loop = cf->arm[i];                  // the test sits directly in its loop's body
			if (loop != SVSL_IR_NONE && fn->insts.items[loop].op == svsl_ir_loop &&
			    range_effect_free(ed, types, loop, i)) {
				kill_range(ed, loop, cf->end[loop]);     // zero trips
				return;
			}
			svsl_ir_edit_kill(ed, i);                    // always exits: an unconditional break
			kill_range(ed, mid, end);                    // (and an else arm, were there one)
			return;
		}
		if (c) {
			svsl_ir_edit_kill(ed, i);
			kill_range(ed, mid, end); // the else arm (or just end_if)
		} else {
			kill_range(ed, i, mid);   // the then arm (through else, when there is one)
			svsl_ir_edit_kill(ed, end);
		}
		return;
	}
	if (in->flags & svsl_ir_flag_loop_exit) return; // keep the exit shape drivers key on

	bool then_empty = range_empty(ed, i, mid);
	bool else_empty = !has_else || range_empty(ed, mid, end);
	if (then_empty && else_empty) {
		svsl_ir_edit_kill(ed, i);
		if (has_else) svsl_ir_edit_kill(ed, mid);
		svsl_ir_edit_kill(ed, end);
	} else if (has_else && else_empty) {
		svsl_ir_edit_kill(ed, mid);
	}
}

// the last live instruction strictly inside (lo, hi), or NONE
static uint32_t last_live(const svsl_ir_edit_t *ed, uint32_t lo, uint32_t hi) {
	for (uint32_t k = hi - 1; k > lo; k--)
		if (!is_dead(ed, k)) return k;
	return SVSL_IR_NONE;
}

static void simplify_loop(svsl_ir_edit_t *ed, const svsl_ir_cf_t *cf, uint32_t i) {
	svsl_ir_func_t *fn       = ed->fn;
	uint32_t        end      = cf->end[i];
	uint32_t        body_end = cf->arm_end[i];
	uint32_t        last     = last_live(ed, i, body_end);
	if (last == SVSL_IR_NONE || cf->arm[last] != i) return;
	svsl_ir_op_ lop = (svsl_ir_op_)fn->insts.items[last].op;
	if (lop != svsl_ir_break && lop != svsl_ir_return && lop != svsl_ir_discard) return;

	// any other way out of (or back into) this loop keeps it a loop
	for (uint32_t k = i + 1; k < end; k++) {
		if (k == last || is_dead(ed, k)) continue;
		svsl_ir_op_ op = (svsl_ir_op_)fn->insts.items[k].op;
		if (op == svsl_ir_break    && svsl_ir_cf_break_target(cf, fn, k, false) == i) return;
		if (op == svsl_ir_continue && svsl_ir_cf_break_target(cf, fn, k, true)  == i) return;
	}
	svsl_ir_edit_kill(ed, i);
	if (lop == svsl_ir_break) svsl_ir_edit_kill(ed, last);
	kill_range(ed, body_end, end); // the (unreachable) continue section and end_loop
}

// --- if-conversion ---------------------------------------------------------------
//
// `if c {A} else {B}` whose arms only compute values and store to
// invocation-local memory becomes A; B; then for each stored location
//     store p, select(c, A's value | old, B's value | old)   (old = load p)
// The stores become unconditional, so forwarding carries them past the merge -
// the phi-free half of mem2reg. Bit-exact: the select picks the value the
// branch would have left. Legality, per arm:
//   * value ops must be safe to run on the path not taken: no integer div/rem
//     by a non-constant (or zero, or signed -1) divisor, no dynamic vector
//     index; intrinsics only from pure math (never derivatives, subgroup/quad
//     ops, clip, helper/tile queries - moving those changes which invocations
//     take part);
//   * loads and stores only through constant indices (a dynamic one could be
//     out of bounds on the untaken path), loads from storage that is always
//     there (locals, params, private/const globals, uniform members), stores to
//     invocation-local storage; a load of a root must precede the arm's stores
//     to it; two stored locations of one root must be identical or disjoint;
//   * stored values must be selectable (scalar, vector, matrix: OpSelect on a
//     struct/array needs SPIR-V 1.4);
//   * no nested control flow (inner ifs convert first, on an earlier round).
// Policy: [branch] keeps the branch, a loop exit test always does; [flatten]
// lifts the cost budget. Otherwise each arm may hold IFCVT_ARM_BUDGET ops that
// cost something (constants, pointers and declarations are free).

#define IFCVT_ARM_BUDGET 8
#define IFCVT_MAX_LOCS   16
#define IFCVT_MAX_INDEX  8

typedef struct ifcvt_loc_t {
	uint32_t ptr, root;
	uint32_t index[IFCVT_MAX_INDEX]; // constant chain indices, root first
	int32_t  depth;
	uint32_t value[2];               // last value stored by the then / else arm (NONE = untouched)
} ifcvt_loc_t;

static bool speculatable_intrinsic(const svsl_ir_inst_t *in) {
	const svsl_intrinsic_t *intr = svsl_intrinsic_get((int32_t)in->args[0]);
	if (!intr) return false;
	switch ((svsl_emit_)intr->emit) {
	case svsl_emit_ext450: case svsl_emit_rcp: case svsl_emit_ldexp: case svsl_emit_log10:
	case svsl_emit_saturate: case svsl_emit_bitfield_extract: case svsl_emit_bitfield_insert:
	case svsl_emit_frexp_mant: case svsl_emit_frexp_exp: case svsl_emit_any: case svsl_emit_all:
	case svsl_emit_bitcast: case svsl_emit_f16tof32: case svsl_emit_f32tof16:
		return true;
	case svsl_emit_core: // derivatives depend on the neighbouring invocations
		return !(intr->op[0] >= SpvOpDPdx && intr->op[0] <= SpvOpFwidthCoarse);
	default:
		return false;
	}
}

// A pointer's location: its root and constant index path. False for a dynamic index.
static bool location_of(const svsl_ir_func_t *fn, uint32_t p, ifcvt_loc_t *out) {
	uint32_t path[IFCVT_MAX_INDEX * 2];
	int32_t  n = 0;
	while (fn->insts.items[p].op == svsl_ir_chain) {
		const svsl_ir_inst_t *ch = &fn->insts.items[p];
		for (int32_t k = (int32_t)ch->aux_count - 1; k >= 0; k--) {
			const svsl_ir_inst_t *idx = &fn->insts.items[fn->aux.items[ch->aux + (uint32_t)k]];
			if (idx->op != svsl_ir_const || n >= IFCVT_MAX_INDEX) return false;
			path[n++] = idx->args[0];
		}
		p = ch->args[0];
	}
	out->root  = p;
	out->depth = n;
	for (int32_t k = 0; k < n; k++) out->index[k] = path[n - 1 - k];
	return true;
}

// same location (true), disjoint (false), or overlapping (*out_overlap)
static bool same_location(const ifcvt_loc_t *a, const ifcvt_loc_t *b, bool *out_overlap) {
	*out_overlap = false;
	if (a->root != b->root) return false;
	int32_t n = a->depth < b->depth ? a->depth : b->depth;
	for (int32_t k = 0; k < n; k++)
		if (a->index[k] != b->index[k]) return false; // diverge: disjoint members
	if (a->depth == b->depth) return true;
	*out_overlap = true;                              // one contains the other
	return false;
}

// Scans arm (lo, hi) as side `side`; false if it can't be speculated.
static bool scan_arm(svsl_ir_edit_t *ed, const svsl_program_t *prog, uint32_t lo, uint32_t hi, int32_t side,
                     bool budgeted, ifcvt_loc_t *locs, int32_t *ref_nlocs) {
	const svsl_ir_func_t *fn      = ed->fn;
	const svsl_types_t   *types   = &prog->types;
	int32_t               cost    = 0;
	uint32_t              stored[IFCVT_MAX_LOCS]; // roots this arm has stored so far
	int32_t               nstored = 0;
	for (uint32_t k = lo + 1; k < hi; k++) {
		if (is_dead(ed, k)) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		svsl_ir_op_           op = (svsl_ir_op_)in->op;
		switch (op) {
		case svsl_ir_const: case svsl_ir_spec_const: case svsl_ir_undef:
		case svsl_ir_var: case svsl_ir_ptr:
			continue; // free
		case svsl_ir_chain: {
			ifcvt_loc_t l;
			if (!location_of(fn, (uint32_t)k, &l)) return false;
			continue;
		}
		case svsl_ir_load: {
			ifcvt_loc_t l;
			if (!location_of(fn, in->args[0], &l)) return false;
			const svsl_ir_inst_t *root = &fn->insts.items[l.root];
			if (!svsl_ir_forwardable_root(fn, prog, l.root) ||
			    (root->op == svsl_ir_ptr && (svsl_ref_)root->args[0] == svsl_ref_resource)) return false; // runtime-sized
			for (int32_t r = 0; r < nstored; r++)
				if (stored[r] == l.root) return false; // would read this arm's (deferred) store
			cost++;
			continue;
		}
		case svsl_ir_store: {
			ifcvt_loc_t l;
			if (!location_of(fn, in->args[0], &l) || !svsl_ir_is_invocation_local(fn, l.root)) return false;
			const svsl_type_t *vt = svsl_type_get(types, svsl_ir_edit_inst(ed, in->args[1])->type);
			if (vt->kind != svsl_type_scalar && vt->kind != svsl_type_vector && vt->kind != svsl_type_matrix)
				return false; // OpSelect on a struct/array needs SPIR-V 1.4
			int32_t at = -1;
			for (int32_t j = 0; j < *ref_nlocs; j++) {
				bool overlap;
				if (same_location(&locs[j], &l, &overlap)) { at = j; break; }
				if (overlap) return false;
			}
			if (at < 0) {
				if (*ref_nlocs >= IFCVT_MAX_LOCS) return false;
				at = (*ref_nlocs)++;
				locs[at] = l;
				locs[at].ptr      = in->args[0];
				locs[at].value[0] = locs[at].value[1] = SVSL_IR_NONE;
			}
			locs[at].value[side] = svsl_ir_edit_resolve(ed, in->args[1]);
			if (nstored >= IFCVT_MAX_LOCS) return false;
			stored[nstored++] = l.root;
			continue;
		}
		case svsl_ir_div: case svsl_ir_rem: { // an integer divisor may be zero (or -1) on the untaken path
			svsl_ir_lanes_t d;
			svsl_scalar_    s;
			int32_t         n;
			if (!svsl_ir_lane_type(types, in->type, &s, &n) || s < svsl_scalar_int8 || s > svsl_scalar_uint64)
				break; // float division never traps: x / 0 is just Inf or NaN
			if (!svsl_ir_const_lanes(ed, in->args[1], &d)) return false;
			for (int32_t l = 0; l < d.count; l++)
				if (d.bits[l] == 0 || d.bits[l] == ~(uint64_t)0) return false;
			break;
		}
		case svsl_ir_extract_dynamic:
			return false; // an out-of-range index is undefined behavior
		case svsl_ir_intrinsic:
			if (!speculatable_intrinsic(in)) return false;
			break;
		default:
			if (!svsl_ir_is_pure(in, types)) return false; // memory, effects, control flow (innermost only)
			break;
		}
		if (budgeted && ++cost > IFCVT_ARM_BUDGET) return false;
	}
	return !budgeted || cost <= IFCVT_ARM_BUDGET;
}

// Converts if `i` when legal.
static void if_convert(svsl_ir_edit_t *ed, const svsl_ir_cf_t *cf, const svsl_program_t *prog, uint32_t i) {
	svsl_ir_func_t       *fn  = ed->fn;
	const svsl_ir_inst_t *in  = &fn->insts.items[i];
	uint32_t              end = cf->end[i];
	uint32_t              mid = cf->arm_end[i];
	if (in->flags & svsl_ir_flag_loop_exit) return;
	if (in->args[1] & SpvSelectionControlDontFlattenMask) return; // [branch]
	bool budgeted = !(in->args[1] & SpvSelectionControlFlattenMask);   // [flatten] lifts the budget

	ifcvt_loc_t locs[IFCVT_MAX_LOCS];
	int32_t     nlocs = 0;
	if (!scan_arm(ed, prog, i, mid, 0, budgeted, locs, &nlocs)) return;
	if (mid != end && !scan_arm(ed, prog, mid, end, 1, budgeted, locs, &nlocs)) return;

	// legal: the arms run unconditionally, their stores merge at end_if
	uint32_t cond = in->args[0];
	svsl_ir_edit_kill(ed, i);
	if (mid != end) svsl_ir_edit_kill(ed, mid);
	svsl_ir_edit_kill(ed, end);
	for (uint32_t k = i + 1; k < end; k++)
		if (!is_dead(ed, k) && fn->insts.items[k].op == svsl_ir_store) svsl_ir_edit_kill(ed, k);
	for (int32_t j = 0; j < nlocs; j++) {
		const ifcvt_loc_t *l    = &locs[j];
		uint32_t           some = l->value[0] != SVSL_IR_NONE ? l->value[0] : l->value[1];
		svsl_type_id_t     type = svsl_ir_edit_inst(ed, some)->type;
		uint32_t           old  = SVSL_IR_NONE;
		if (l->value[0] == SVSL_IR_NONE || l->value[1] == SVSL_IR_NONE)
			old = svsl_ir_edit_insert(ed, end, (svsl_ir_inst_t){ .op = svsl_ir_load, .type = type,
				.args = { l->ptr, 0, 0, SVSL_IR_NONE }, .loc = in->loc }, NULL, 0);
		uint32_t sel = svsl_ir_edit_insert(ed, end, (svsl_ir_inst_t){ .op = svsl_ir_select, .type = type,
			.args = { cond, l->value[0] != SVSL_IR_NONE ? l->value[0] : old,
			                l->value[1] != SVSL_IR_NONE ? l->value[1] : old, SVSL_IR_NONE },
			.loc = in->loc }, NULL, 0);
		svsl_ir_edit_insert(ed, end, (svsl_ir_inst_t){ .op = svsl_ir_store, .type = SVSL_TYPE_NONE,
			.args = { l->ptr, sel, 0, SVSL_IR_NONE }, .loc = in->loc }, NULL, 0);
	}
}

void svsl_ir_cfg(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	(void)level;
	svsl_ir_func_t *fn = ed->fn;
	uint32_t        n  = (uint32_t)fn->insts.count;
	svsl_ir_cf_t    cf;
	svsl_ir_cf_build(ed->scratch, fn, &cf);

	// in reverse, so inner constructs settle before the loops and ifs around them
	for (uint32_t i = n; i-- > 0;) {
		if (is_dead(ed, i)) continue;
		switch ((svsl_ir_op_)fn->insts.items[i].op) {
		case svsl_ir_if:   simplify_if(ed, &cf, &prog->types, i); break;
		case svsl_ir_loop: simplify_loop(ed, &cf, i); break;
		case svsl_ir_break: case svsl_ir_continue: case svsl_ir_return: case svsl_ir_discard: {
			// unreachable: the rest of this arm (to the end of the function at top level)
			uint32_t a = cf.arm[i];
			uint32_t e = a == SVSL_IR_NONE ? n : cf.arm_end[a];
			if (e > i + 1) kill_range(ed, i + 1, e - 1);
			break;
		}
		default:
			break;
		}
	}
	// if-conversion inserts (the merge selects), so it runs after every kill, in a
	// forward sweep: a live if's arms and condition are never in a killed range
	for (uint32_t i = 0; i < n; i++)
		if (!is_dead(ed, i) && fn->insts.items[i].op == svsl_ir_if) if_convert(ed, &cf, prog, i);
}

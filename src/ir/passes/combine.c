// Instruction combining (LLVM's InstSimplify + InstCombine): local rewrites that
// either replace an instruction with a value that already dominates it (an
// operand, an operand's operand, or a constant), or rewrite it in place into a
// canonical form so CSE and the other passes see one spelling of each value.
// One forward sweep: each instruction's operands are resolved first, so a
// rewrite is visible to everything after it in the same sweep.
//
// Exact on every input (-O1):
//   * composites: extract(construct C, k) -> C[k]; extract(insert(v, k, x), k)
//     -> x and (.., j != k) -> extract(v, k); identity shuffle -> v;
//     construct(extract(v, 0..n-1)) -> v, other lane picks -> shuffle(v, ..);
//     nested vector constructs splice flat; shuffle(construct) picks the
//     components; extract/shuffle of a shuffle compose onto its source.
//   * integer identities, scalar or splat vector: x+0 x-0 x*1 x/1 x|0 x^0 x<<0
//     x>>0 -> x; x*0 -> 0; x&x x|x -> x.
//   * booleans: !!x -> x; !(a==b) <-> a!=b (floats too: !FOrdEqual is
//     FUnordNotEqual); !(a<b) -> a>=b etc. on integers only (a float NaN
//     breaks it) and never for a loop exit test (Adreno keys on its shape).
//   * select: (c, x, x) -> x; (!c, a, b) -> (c, b, a); (c, true, false) -> c;
//     (c, false, true) -> !c.
//   * integer reassociation (wrapping arithmetic is associative): (x op c1) op
//     c2 -> x op (c1 op c2) for add/sub (mixed), mul, and, or, xor, and
//     same-direction shifts whose total stays under the width.
//   * vector * splat(s) -> vector * s, so emit picks OpVectorTimesScalar.
// -O2 adds float x*1.0 and x/1.0 -> x (IEEE edge cases differ: not oracle-covered).
//
// Each rule earned its place on the corpus + sk_texenc (docs/PLAN_optimizer_llvm.md,
// "Ablation"): rules that never changed an output were removed.
// See docs/PLAN_optimizer_llvm.md item 5 and docs/OPTIMIZATION_PLAN.md section 4.

#include "passes.h"
#include "../ir_const.h"
#include "../ir_operands.h"

#define COMBINE_SETTLE_STEPS 8 // canonical rewrites per instruction per sweep (the corpus needs at most 6)

typedef struct combine_t {
	svsl_ir_edit_t *ed;
	svsl_types_t   *types;
	svsl_opt_level_ level;
	const uint8_t  *exit_cond; // per original id: the operand of a loop exit test
} combine_t;

static svsl_ir_inst_t *at(const combine_t *c, uint32_t id) {
	return svsl_ir_edit_inst(c->ed, id);
}

static bool lanes(const combine_t *c, svsl_type_id_t type, svsl_scalar_ *out_s, int32_t *out_n) {
	return svsl_ir_lane_type(c->types, type, out_s, out_n);
}

static bool is_int(svsl_scalar_ s) {
	return s >= svsl_scalar_int8 && s <= svsl_scalar_uint64;
}
static bool is_float32(svsl_scalar_ s) {
	return s == svsl_scalar_float32 || s == svsl_scalar_half;
}

// integer kind of a value's type (scalar or vector), or false
static bool int_value(const combine_t *c, uint32_t id, svsl_scalar_ *out_s) {
	int32_t n;
	return lanes(c, at(c, id)->type, out_s, &n) && is_int(*out_s);
}

// splat constant with exactly these bits (canonical svsl_ir_const encoding)
static bool splat_is(const combine_t *c, uint32_t id, uint64_t bits) {
	uint64_t b;
	return svsl_ir_const_splat(c->ed, id, &b) && b == bits;
}

// `id` if it already has the instruction's type (a valid replacement), else NONE
static uint32_t same_type(const combine_t *c, uint32_t id, svsl_type_id_t type) {
	return at(c, id)->type == type ? id : SVSL_IR_NONE;
}

// --- simplify: an existing (or constant) value for instruction i ------------------

static uint32_t simplify(const combine_t *c, uint32_t i) {
	const svsl_ir_inst_t *in = at(c, i);
	svsl_ir_op_           op = (svsl_ir_op_)in->op;
	uint32_t              a  = in->args[0], b = in->args[1];

	switch (op) {
	case svsl_ir_extract: {
		const svsl_ir_inst_t *src = at(c, a);
		svsl_scalar_          vs;
		int32_t               vn;
		// extract(construct C, k) -> C[k]: component k is C[k] unless C is a vector
		// built from wider parts (tex[]'s int3(coord, 0))
		if (src->op == svsl_ir_construct && b < src->aux_count &&
		    (!lanes(c, src->type, &vs, &vn) || (int32_t)src->aux_count == vn))
			return same_type(c, c->ed->fn->aux.items[src->aux + b], in->type);
		if (src->op == svsl_ir_insert && src->args[1] == b)           // extract(insert(v, k, x), k) -> x
			return same_type(c, src->args[2], in->type);
		return SVSL_IR_NONE;
	}
	case svsl_ir_shuffle: { // identity shuffle over a whole vector -> the vector
		int32_t n;
		svsl_scalar_ vs;
		if (in->type != at(c, a)->type || !lanes(c, in->type, &vs, &n) || (int32_t)in->args[2] != n)
			return SVSL_IR_NONE;
		for (int32_t k = 0; k < n; k++)
			if (((b >> (k * 4)) & 0xF) != (uint32_t)k) return SVSL_IR_NONE;
		return a;
	}
	case svsl_ir_construct: { // construct(extract(v, 0..n-1)) -> v
		const svsl_ir_func_t *fn = c->ed->fn;
		uint32_t v = SVSL_IR_NONE;
		for (uint32_t k = 0; k < in->aux_count; k++) {
			const svsl_ir_inst_t *e = at(c, fn->aux.items[in->aux + k]);
			if (e->op != svsl_ir_extract || e->args[1] != k || (v != SVSL_IR_NONE && e->args[0] != v))
				return SVSL_IR_NONE;
			v = e->args[0];
		}
		return v == SVSL_IR_NONE ? SVSL_IR_NONE : same_type(c, v, in->type);
	}
	case svsl_ir_select:
		if (in->args[1] == in->args[2]) return in->args[1];                    // (c, x, x)
		if (splat_is(c, in->args[1], 1) && splat_is(c, in->args[2], 0))       // (c, true, false) -> c
			return same_type(c, a, in->type);
		return SVSL_IR_NONE;
	case svsl_ir_log_not:
		return at(c, a)->op == svsl_ir_log_not ? at(c, a)->args[0] : SVSL_IR_NONE; // !!x
	default:
		break;
	}

	svsl_scalar_ rs;
	int32_t      rn;
	if (!lanes(c, in->type, &rs, &rn)) return SVSL_IR_NONE;
	if (is_float32(rs)) { // -O2 only: changes -0.0 / NaN / Inf edge results
		if (c->level < svsl_opt_aggressive) return SVSL_IR_NONE;
		uint64_t one = 0x3F800000u;
		if (op == svsl_ir_mul && splat_is(c, b, one)) return same_type(c, a, in->type);
		if (op == svsl_ir_mul && splat_is(c, a, one)) return same_type(c, b, in->type);
		if (op == svsl_ir_div && splat_is(c, b, one)) return same_type(c, a, in->type);
		return SVSL_IR_NONE;
	}
	if (!is_int(rs)) return SVSL_IR_NONE;
	switch (op) {
	case svsl_ir_add:
		if (splat_is(c, b, 0)) return same_type(c, a, in->type);
		if (splat_is(c, a, 0)) return same_type(c, b, in->type);
		return SVSL_IR_NONE;
	case svsl_ir_sub:
		return splat_is(c, b, 0) ? same_type(c, a, in->type) : SVSL_IR_NONE;
	case svsl_ir_mul:
		if (splat_is(c, b, 1)) return same_type(c, a, in->type);
		if (splat_is(c, a, 1)) return same_type(c, b, in->type);
		if (splat_is(c, b, 0)) return same_type(c, b, in->type); // x*0 -> 0 (exact for integers)
		if (splat_is(c, a, 0)) return same_type(c, a, in->type);
		return SVSL_IR_NONE;
	case svsl_ir_div:
		return splat_is(c, b, 1) ? same_type(c, a, in->type) : SVSL_IR_NONE;
	case svsl_ir_bit_or:
		if (a == b || splat_is(c, b, 0)) return same_type(c, a, in->type);
		if (splat_is(c, a, 0)) return same_type(c, b, in->type);
		return SVSL_IR_NONE;
	case svsl_ir_bit_and:
		return a == b ? a : SVSL_IR_NONE;
	case svsl_ir_bit_xor:
		if (splat_is(c, b, 0)) return same_type(c, a, in->type);
		if (splat_is(c, a, 0)) return same_type(c, b, in->type);
		return SVSL_IR_NONE;
	case svsl_ir_shl: case svsl_ir_shr:
		return splat_is(c, b, 0) ? same_type(c, a, in->type) : SVSL_IR_NONE;
	default:
		return SVSL_IR_NONE;
	}
}

// --- canonicalize: in-place rewrites of instruction i ----------------------------

static svsl_ir_op_ invert(svsl_ir_op_ op) { // !(a OP b) == a invert(OP) b, on integers
	switch (op) {
	case svsl_ir_eq: return svsl_ir_ne;
	case svsl_ir_ne: return svsl_ir_eq;
	case svsl_ir_lt: return svsl_ir_ge;
	case svsl_ir_ge: return svsl_ir_lt;
	case svsl_ir_le: return svsl_ir_gt;
	default:         return svsl_ir_le; // gt
	}
}

// (x op c1) op c2 -> x op (c1 op c2): returns true when rewritten. add and sub
// mix: (x +- c1) +- c2 -> x + k (or x - k when the outer op is a sub), so a new
// constant appears only where two ops actually merge.
static bool reassociate(const combine_t *c, uint32_t i, svsl_ir_inst_t *in, svsl_scalar_ s) {
	svsl_ir_op_     op     = (svsl_ir_op_)in->op;
	svsl_ir_inst_t *inner  = at(c, in->args[0]);
	bool            addsub = (op == svsl_ir_add || op == svsl_ir_sub) &&
	                         (inner->op == svsl_ir_add || inner->op == svsl_ir_sub);
	if ((inner->op != op && !addsub) || inner->type != in->type) return false;
	svsl_ir_lanes_t c1, c2;
	if (!svsl_ir_const_lanes(c->ed, inner->args[1], &c1) || !svsl_ir_const_lanes(c->ed, in->args[1], &c2))
		return false;
	if (c1.count != c2.count) return false;
	svsl_ir_lanes_t r = { .count = c1.count };
	for (int32_t l = 0; l < r.count; l++) {
		if (addsub) { // in the outer op's sense: k = c2 +- c1 (same op adds, mixed subtracts)
			svsl_ir_op_ merge = inner->op == op ? svsl_ir_add : svsl_ir_sub;
			if (!svsl_ir_eval_binary(merge, s, c2.bits[l], c1.bits[l], &r.bits[l])) return false;
		} else if (op == svsl_ir_shl || op == svsl_ir_shr) { // same direction: amounts add
			int32_t n;
			svsl_scalar_ as;
			if (!lanes(c, at(c, in->args[1])->type, &as, &n) || at(c, inner->args[1])->type != at(c, in->args[1])->type)
				return false;
			uint64_t lim, sum;
			if ((int64_t)c1.bits[l] < 0 || (int64_t)c2.bits[l] < 0) return false;
			if (!svsl_ir_eval_binary(svsl_ir_add, as, c1.bits[l], c2.bits[l], &sum)) return false;
			// the total must stay a valid shift: probe it on the shifted kind
			if (!svsl_ir_eval_shift(op, s, as, 0, sum, &lim)) return false;
			r.bits[l] = sum;
		} else if (!svsl_ir_eval_binary(op, s, c1.bits[l], c2.bits[l], &r.bits[l])) {
			return false;
		}
	}
	svsl_type_id_t ctype = at(c, in->args[1])->type;
	in->args[0] = inner->args[0];
	in->args[1] = svsl_ir_make_const(c->ed, c->types, i, ctype, &r);
	return true;
}

// vector * splat(s) -> vector * s (float), so emit picks OpVectorTimesScalar
static bool strip_vector_splat(const combine_t *c, svsl_ir_inst_t *in) {
	svsl_scalar_ rs;
	int32_t      rn;
	if (in->op != svsl_ir_mul || !lanes(c, in->type, &rs, &rn) || rn == 1) return false;
	if (rs != svsl_scalar_float32 && rs != svsl_scalar_half && rs != svsl_scalar_float16 &&
	    rs != svsl_scalar_float64) return false;
	for (int32_t w = 0; w < 2; w++) {
		if (at(c, in->args[w ^ 1])->type != in->type) continue; // the other operand stays a vector
		const svsl_ir_inst_t *sp = at(c, in->args[w]);
		if (sp->op != svsl_ir_construct || (int32_t)sp->aux_count != rn) continue;
		uint32_t s0 = c->ed->fn->aux.items[sp->aux];
		int32_t  sn;
		svsl_scalar_ ss;
		if (!lanes(c, at(c, s0)->type, &ss, &sn) || sn != 1) continue;
		bool splat = true;
		for (uint32_t k = 1; k < sp->aux_count; k++) splat &= c->ed->fn->aux.items[sp->aux + k] == s0;
		if (splat) { in->args[w] = s0; return true; }
	}
	return false;
}

// new aux slice for `in` (a construct) holding these components
static void set_components(const combine_t *c, svsl_ir_inst_t *in, const uint32_t *ids, uint32_t n) {
	svsl_ir_func_t *fn  = c->ed->fn;
	uint32_t        aux = (uint32_t)fn->aux.count;
	for (uint32_t k = 0; k < n; k++) svsl_array_push(c->ed->arena, &fn->aux, ids[k]);
	in->aux       = aux;
	in->aux_count = n;
}

static bool canonicalize(const combine_t *c, uint32_t i) {
	svsl_ir_inst_t *in = at(c, i);
	svsl_ir_op_     op = (svsl_ir_op_)in->op;
	svsl_scalar_    s;
	int32_t         n;

	switch (op) {
	case svsl_ir_log_not: {
		if (i < c->ed->base && c->exit_cond[i]) return false; // keep the loop exit's `!cond`
		svsl_ir_inst_t *cmp = at(c, in->args[0]);
		svsl_ir_op_     cop = (svsl_ir_op_)cmp->op;
		if (!(svsl_ir_op_traits(cop) & svsl_ir_trait_compare)) return false;
		if (!lanes(c, at(c, cmp->args[0])->type, &s, &n)) return false;
		if (!is_int(s) && !(s == svsl_scalar_bool) && cop != svsl_ir_eq && cop != svsl_ir_ne) return false;
		if (s == svsl_scalar_bool && cop != svsl_ir_eq && cop != svsl_ir_ne) return false;
		in->op      = (uint8_t)invert(cop);
		in->args[0] = cmp->args[0];
		in->args[1] = cmp->args[1];
		return true;
	}
	case svsl_ir_select: {
		svsl_ir_inst_t *cond = at(c, in->args[0]);
		if (cond->op == svsl_ir_log_not) { // (!c, a, b) -> (c, b, a)
			uint32_t t = in->args[1];
			in->args[0] = cond->args[0];
			in->args[1] = in->args[2];
			in->args[2] = t;
			return true;
		}
		if (splat_is(c, in->args[1], 0) && splat_is(c, in->args[2], 1) && at(c, in->args[0])->type == in->type) {
			in->op      = svsl_ir_log_not; // (c, false, true) -> !c
			in->args[1] = in->args[2] = 0;
			return true;
		}
		return false;
	}
	case svsl_ir_add: case svsl_ir_sub: case svsl_ir_mul: case svsl_ir_bit_and: case svsl_ir_bit_or: case svsl_ir_bit_xor:
	case svsl_ir_shl: case svsl_ir_shr:
		if (int_value(c, i, &s) && reassociate(c, i, in, s)) return true;
		return strip_vector_splat(c, in);
	case svsl_ir_extract: {
		const svsl_ir_inst_t *src = at(c, in->args[0]);
		if (src->op == svsl_ir_insert && src->args[1] != in->args[1]) { // reads past the insert
			in->args[0] = src->args[0];
			return true;
		}
		if (src->op == svsl_ir_shuffle && in->args[1] < src->args[2]) { // extract(shuffle(v, lanes), k)
			in->args[0] = src->args[0];
			in->args[1] = (src->args[1] >> (in->args[1] * 4)) & 0xF;
			return true;
		}
		return false;
	}
	case svsl_ir_shuffle: {
		const svsl_ir_inst_t *src = at(c, in->args[0]);
		if (src->op == svsl_ir_shuffle) { // shuffle(shuffle(v, l1), l2) -> shuffle(v, l1[l2])
			uint32_t mask = 0;
			for (uint32_t j = 0; j < in->args[2]; j++) {
				uint32_t outer = (in->args[1] >> (j * 4)) & 0xF;
				if (outer >= src->args[2]) return false;
				mask |= ((src->args[1] >> (outer * 4)) & 0xF) << (j * 4);
			}
			in->args[0] = src->args[0];
			in->args[1] = mask;
			return true;
		}
		if (src->op == svsl_ir_construct && lanes(c, src->type, &s, &n) && (int32_t)src->aux_count == n) {
			uint32_t picked[4]; // shuffle(construct(s0..sn), lanes) -> construct(picked s)
			for (uint32_t j = 0; j < in->args[2] && j < 4; j++) {
				uint32_t l = (in->args[1] >> (j * 4)) & 0xF;
				if (l >= src->aux_count) return false;
				picked[j] = c->ed->fn->aux.items[src->aux + l];
			}
			uint32_t count = in->args[2];
			in->op      = svsl_ir_construct;
			in->args[0] = 0; in->args[1] = 0; in->args[2] = 0;
			set_components(c, in, picked, count);
			return true;
		}
		return false;
	}
	case svsl_ir_construct: {
		if (!lanes(c, in->type, &s, &n) || n == 1) return false; // vectors only
		const svsl_ir_func_t *fn = c->ed->fn;
		uint32_t comps[4], count = 0, from = SVSL_IR_NONE, packed = 0;
		bool     spliced = false, all_extract = true;
		for (uint32_t k = 0; k < in->aux_count; k++) {
			uint32_t              v  = fn->aux.items[in->aux + k];
			const svsl_ir_inst_t *vi = at(c, v);
			int32_t               vn;
			svsl_scalar_          vs;
			if (vi->op == svsl_ir_construct && lanes(c, vi->type, &vs, &vn) && vn > 1 &&
			    (int32_t)vi->aux_count == vn) { // splice a nested vector construct
				for (uint32_t m = 0; m < vi->aux_count && count < 4; m++) comps[count++] = fn->aux.items[vi->aux + m];
				spliced = true;
				all_extract = false;
				continue;
			}
			if (count < 4) comps[count] = v;
			count++;
			if (vi->op == svsl_ir_extract && (from == SVSL_IR_NONE || vi->args[0] == from)) {
				from    = vi->args[0];
				packed |= vi->args[1] << ((count - 1) * 4);
			} else {
				all_extract = false;
			}
		}
		if (count != (uint32_t)n) return false;
		if (spliced) { set_components(c, in, comps, count); return true; }
		// construct(extract(v, i0..in)) -> shuffle(v, i0..in) when v is a vector of the same scalar
		if (all_extract && from != SVSL_IR_NONE && at(c, from)->op != svsl_ir_construct) {
			int32_t      fnn;
			svsl_scalar_ fs;
			if (!lanes(c, at(c, from)->type, &fs, &fnn) || fs != s || fnn == 1) return false;
			in->op        = svsl_ir_shuffle;
			in->args[0]   = from;
			in->args[1]   = packed;
			in->args[2]   = (uint32_t)n;
			in->aux_count = 0;
			return true;
		}
		return false;
	}
	default:
		return false;
	}
}

void svsl_ir_combine(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	svsl_ir_func_t *fn    = ed->fn;
	int32_t         count = fn->insts.count;
	uint8_t        *exit  = svsl_arena_alloc(ed->scratch, (size_t)(count > 0 ? count : 1)); // zeroed
	for (int32_t i = 0; i < count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op == svsl_ir_if && (in->flags & svsl_ir_flag_loop_exit)) exit[in->args[0]] = 1;
	}
	combine_t c = { .ed = ed, .types = &prog->types, .level = level, .exit_cond = exit };
	ed->resolved = true; // every instruction's operands resolve below, in order

	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op == svsl_ir_nop) continue;
		if (ed->replaced) svsl_ir_edit_resolve_operands(ed, (uint32_t)i);
		if (!(svsl_ir_op_traits((svsl_ir_op_)in->op) & svsl_ir_trait_pure)) continue;
		if (in->op == svsl_ir_const || in->op == svsl_ir_ptr || in->op == svsl_ir_chain) continue;

		uint32_t to = simplify(&c, (uint32_t)i);
		if (to != SVSL_IR_NONE && to != (uint32_t)i) {
			svsl_ir_edit_replace(ed, (uint32_t)i, to);
			continue;
		}
		// canonical forms compose (e.g. sub -> add, then reassociate): settle each
		// instruction before moving on. Every step is a strict normalization; a
		// chain longer than the bound (an extract walking a long insert run) just
		// resumes on combine's next sweep, which the touch guarantees
		for (int32_t step = 0; step < COMBINE_SETTLE_STEPS && canonicalize(&c, (uint32_t)i); step++)
			svsl_ir_edit_touch(ed);
	}
}

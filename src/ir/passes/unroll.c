// Profitability-driven full unrolling (LLVM's LoopFullUnroll, with its cost
// bonus for loads that turn constant). An `[unroll]` loop is copied out when
// that is what it takes to make a large local array constant-indexed - so SROA
// (or the driver) can hold it in registers instead of scratch memory. Policy,
// each part measured on Adreno (docs/PLAN_optimizer_llvm.md, Phase 2):
//   * only arrays of UNROLL_MIN_ELEMENTS or more (every level of a nested
//     array counts) qualify: drivers keep smaller dynamically indexed arrays
//     in registers themselves (glslang's 8x8hdr encoder keeps eight of them,
//     0 B scratch), and unrolling for them grew sk_texenc's 6x6 encoder 3.5x
//     and slowed its alpha path 15%;
//   * the split array must fit in registers: UNROLL_MAX_SCALARS once its
//     accesses are straight-line, UNROLL_MAX_SCALARS_IN_LOOP while a loop that
//     stays rolled encloses one (every element then lives across that loop's
//     back edge). Synthetic encoder-shaped shaders swept on Adreno: a split
//     float4[32] ran 11x faster straight-line but 2-3x slower inside a rolled
//     candidate loop, where float[64] still won;
//   * all or nothing per array: every index into it must derive from counted
//     loops (a data-dependent index anywhere keeps it an array), and all of
//     those loop nests must fit UNROLL_BUDGET together - unrolling only some
//     of an array's loops bought nothing but code and more scratch;
//   * loops that serve no such array stay rolled, Unroll hint intact for the
//     driver (unrolling everything, as spirv-opt does, made the 6x6 encoder
//     2.2x slower on Adreno).
// Result: 6x6 photo/alpha -37%/-24%, 8x8hdr -53%, 4x4 unchanged.
//
// The loop must have the counted shape ir_build gives a `for`:
//     store v, c0                 (a constant, in the loop's own arm)
//     loop
//       ...prefix: loads of v, constants, pure ops...
//       if (cond) break           (the svsl_ir_flag_loop_exit test)
//       ...body: never writes v, no break/continue of this loop...
//     loop_continue
//       ...loads of v, pure ops, exactly one `store v, next`...
//     end_loop
// The trip count is found by running the prefix and the continue section on
// constants with the exact evaluators (any step, any compare, signed or not),
// up to UNROLL_MAX_TRIPS. Each copy is prefix + body + continue section with
// the exit test dropped; forwarding and folding then make each copy's counter
// a constant. Innermost loops only: an enclosing loop unrolls on a later round,
// once its body is straight-line.

#include "passes.h"
#include "../ir_cf.h"
#include "../ir_const.h"
#include "../ir_operands.h"
#include "../../../vendor/spirv.h"

#define UNROLL_MAX_TRIPS 256

// The policy's thresholds. A research build may override them (-DUNROLL_MIN_ELEMENTS=0
// forces unrolling, a huge value forbids it): tests/perf/unroll_sweep.py compares such
// builds on a device.
#ifndef UNROLL_BUDGET
#define UNROLL_BUDGET              32768 // instructions an array's fully unrolled loop nests may cost
#endif
#ifndef UNROLL_MIN_ELEMENTS
#define UNROLL_MIN_ELEMENTS        32    // smaller arrays the driver keeps in registers itself
#endif
#ifndef UNROLL_MAX_SCALARS
#define UNROLL_MAX_SCALARS         256   // a split array's registers, once its accesses are straight-line
#endif
#ifndef UNROLL_MAX_SCALARS_IN_LOOP
#define UNROLL_MAX_SCALARS_IN_LOOP 64    // ... while a loop that stays rolled encloses an access
#endif

static bool is_dead(const svsl_ir_edit_t *ed, uint32_t i) {
	return ed->fn->insts.items[i].op == svsl_ir_nop || svsl_ir_edit_resolve(ed, i) == SVSL_IR_NONE;
}

// An operand of the simulated range: a value computed in it, or a constant
// from outside (CSE shares constants across arms).
static bool operand(const svsl_ir_edit_t *ed, uint32_t lo, uint32_t hi, const uint64_t *vals,
                    uint32_t x, uint64_t *out) {
	if (x > lo && x < hi) { *out = vals[x - lo]; return true; }
	svsl_ir_lanes_t c;
	if (!svsl_ir_const_lanes(ed, x, &c) || c.count != 1) return false;
	*out = c.bits[0];
	return true;
}

// Runs instructions (lo, hi) on constants with the counter (root `v`) holding
// `value`; `vals` receives each result. Loads of v read `value`; a store to v
// sets *out_stored. Markers, other loads and effects make it fail.
static bool simulate(const svsl_ir_edit_t *ed, const svsl_types_t *types, uint32_t lo, uint32_t hi,
                     uint32_t v, uint64_t value, uint64_t *vals, uint64_t *out_stored) {
	const svsl_ir_func_t *fn = ed->fn;
	for (uint32_t k = lo + 1; k < hi; k++) {
		if (is_dead(ed, k)) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		svsl_ir_op_           op = (svsl_ir_op_)in->op;
		svsl_scalar_          s, as;
		int32_t               n, an;
		uint64_t             *out = &vals[k - lo];
		if (op == svsl_ir_const) { *out = (uint64_t)in->args[0] | ((uint64_t)in->args[1] << 32); continue; }
		if (op == svsl_ir_load && in->args[0] == v) { *out = value; continue; }
		if (op == svsl_ir_store && in->args[0] == v && out_stored) {
			if (!operand(ed, lo, hi, vals, in->args[1], out_stored)) return false;
			continue;
		}
		// a scalar value op over values computed in this range
		if (!svsl_ir_is_pure(in, types) || op == svsl_ir_intrinsic || in->aux_count) return false;
		if (!svsl_ir_lane_type(types, in->type, &s, &n) || n != 1) return false;
		uint32_t mask = svsl_ir_value_arg_mask(in);
		uint64_t x = 0, y = 0, z = 0;
		if (!(mask & 1) || !svsl_ir_lane_type(types, svsl_ir_edit_inst(ed, in->args[0])->type, &as, &an)) return false;
		if (!operand(ed, lo, hi, vals, in->args[0], &x)) return false;
		if ((mask & 2) && !operand(ed, lo, hi, vals, in->args[1], &y)) return false;
		if ((mask & 4) && !operand(ed, lo, hi, vals, in->args[2], &z)) return false;
		switch (op) {
		case svsl_ir_neg: case svsl_ir_bit_not: case svsl_ir_log_not:
			if (!svsl_ir_eval_unary(op, as, x, out)) return false;
			break;
		case svsl_ir_convert:
			if (!svsl_ir_eval_convert(as, s, x, out)) return false;
			break;
		case svsl_ir_shl: case svsl_ir_shr: {
			svsl_scalar_ bs;
			int32_t      bn;
			if (!svsl_ir_lane_type(types, svsl_ir_edit_inst(ed, in->args[1])->type, &bs, &bn) ||
			    !svsl_ir_eval_shift(op, as, bs, x, y, out)) return false;
			break;
		}
		case svsl_ir_select:
			*out = (x & 1) ? y : z;
			break;
		default:
			if (mask != 0x3 || !svsl_ir_eval_binary(op, as, x, y, out)) return false;
			break;
		}
	}
	return true;
}

// A counted `[unroll]` loop: its markers, counter, trip count and size.
typedef struct counted_t {
	uint32_t loop, cont, end; // loop, loop_continue, end_loop
	uint32_t test, test_end;  // the exit test's if .. end_if
	uint32_t counter;         // the var it steps
	uint32_t trips;
	uint32_t size;            // live instructions inside
	bool     innermost;
	uint64_t cost;            // instructions once it and every counted loop inside are unrolled
} counted_t;

// The loop at `l` as a counted loop (see the file header); false when it isn't.
static bool counted_loop(svsl_ir_edit_t *ed, const svsl_ir_cf_t *cf, const svsl_types_t *types,
                         uint32_t l, counted_t *out) {
	const svsl_ir_func_t *fn   = ed->fn;
	uint32_t              end  = cf->end[l];
	uint32_t              cont = cf->arm_end[l]; // loop_continue (or end_loop: no continue section)
	if (fn->insts.items[cont].op != svsl_ir_loop_continue) return false;

	// the exit test: the first marker in the body, an `if (c) break` flagged as the loop's
	uint32_t test = SVSL_IR_NONE;
	for (uint32_t k = l + 1; k < cont && test == SVSL_IR_NONE; k++)
		if (!is_dead(ed, k) && svsl_ir_ends_run((svsl_ir_op_)fn->insts.items[k].op)) test = k;
	if (test == SVSL_IR_NONE || fn->insts.items[test].op != svsl_ir_if ||
	    !(fn->insts.items[test].flags & svsl_ir_flag_loop_exit) || cf->arm_end[test] != cf->end[test])
		return false;
	uint32_t cond     = fn->insts.items[test].args[0];
	uint32_t test_end = cf->end[test];
	if (cond <= l || cond >= test) return false;
	// the copies drop the whole test: it may hold nothing but its break
	uint32_t brk = SVSL_IR_NONE;
	for (uint32_t k = test + 1; k < test_end; k++) {
		if (is_dead(ed, k)) continue;
		if (brk != SVSL_IR_NONE || fn->insts.items[k].op != svsl_ir_break) return false;
		brk = k;
	}
	if (brk == SVSL_IR_NONE) return false;

	// the counter: the one var the continue section stores, a scalar the body never writes
	uint32_t v = SVSL_IR_NONE;
	for (uint32_t k = cont + 1; k < end; k++) {
		if (is_dead(ed, k)) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		if (in->op != svsl_ir_store) continue;
		if (v != SVSL_IR_NONE) return false;
		v = in->args[0];
	}
	if (v == SVSL_IR_NONE || fn->insts.items[v].op != svsl_ir_var) return false;
	bool     innermost = true;
	uint32_t size      = 0;
	for (uint32_t k = l + 1; k < end; k++) {
		if (is_dead(ed, k)) continue;
		size++;
		if (k >= cont) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		if ((in->op == svsl_ir_store || in->op == svsl_ir_atomic) && svsl_ir_root_ptr(fn, in->args[0]) == v)
			return false;
		if (in->op == svsl_ir_loop) innermost = false;
		if ((in->op == svsl_ir_break    && (k < test || k > test_end) && svsl_ir_cf_break_target(cf, fn, k, false) == l) ||
		    (in->op == svsl_ir_continue && svsl_ir_cf_break_target(cf, fn, k, true) == l))
			return false; // another way out of (or around) the loop
	}

	// the counter's value entering the loop: the closest store, in the loop's own arm
	uint64_t value = 0;
	bool     found = false;
	for (uint32_t k = l; k-- > 0 && !found;) {
		if (is_dead(ed, k)) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		if ((in->op != svsl_ir_store && in->op != svsl_ir_atomic) || svsl_ir_root_ptr(fn, in->args[0]) != v)
			continue;
		svsl_ir_lanes_t c;
		if (in->op != svsl_ir_store || in->args[0] != v || cf->arm[k] != cf->arm[l] ||
		    !svsl_ir_const_lanes(ed, in->args[1], &c) || c.count != 1) return false;
		value = c.bits[0];
		found = true;
	}
	if (!found) return false;

	// run it: prefix decides, continue section steps
	uint64_t *vals  = svsl_arena_alloc(ed->scratch, (size_t)(end - l) * sizeof(uint64_t));
	uint32_t  trips = 0;
	for (;; trips++) {
		if (!simulate(ed, types, l, test, v, value, vals, NULL)) return false;
		if (vals[cond - l] & 1) break;                  // the exit test fires
		if (trips >= UNROLL_MAX_TRIPS) return false;
		uint64_t next = value;
		if (!simulate(ed, types, cont, end, v, value, vals + (cont - l), &next) || next == value)
			return false;
		value = next;
	}
	*out = (counted_t){ .loop = l, .cont = cont, .end = end, .test = test, .test_end = test_end,
	                    .counter = v, .trips = trips, .size = size, .innermost = innermost };
	return true;
}

// Replaces counted loop `c` with `trips` copies of its prefix, body and continue section.
static void unroll(svsl_ir_edit_t *ed, const counted_t *c) {
	svsl_ir_func_t *fn   = ed->fn;
	uint32_t        l    = c->loop, end = c->end;
	uint32_t       *copy = svsl_arena_alloc(ed->scratch, (size_t)(end - l) * sizeof(uint32_t));
	uint32_t        aux[64];

	for (uint32_t t = 0; t < c->trips; t++) {
		for (uint32_t k = l + 1; k < end; k++) {
			if (is_dead(ed, k) || (k >= c->test && k <= c->test_end) || k == c->cont) continue;
			svsl_ir_inst_t in   = fn->insts.items[k];
			uint32_t       mask = svsl_ir_value_arg_mask(&in);
			for (int32_t a = 0; a < 4; a++)
				if ((mask & (1u << a)) && in.args[a] > l && in.args[a] < end) in.args[a] = copy[in.args[a] - l];
			const uint32_t *src = &fn->aux.items[in.aux];
			uint32_t       *dst = in.aux_count <= 64 ? aux
			                    : svsl_arena_alloc(ed->scratch, (size_t)in.aux_count * sizeof(uint32_t));
			bool values = svsl_ir_aux_holds_values(&in);
			for (uint32_t a = 0; a < in.aux_count; a++)
				dst[a] = values && src[a] > l && src[a] < end ? copy[src[a] - l] : src[a];
			copy[k - l] = svsl_ir_edit_insert(ed, end, in, dst, in.aux_count);
		}
	}
	for (uint32_t k = l; k <= end; k++)
		if (!is_dead(ed, k)) svsl_ir_edit_kill(ed, k);
}

// Sets of counted loops: bitsets of `w` words, one bit per loop.
static void set_or(uint64_t *dst, const uint64_t *src, int32_t w) {
	for (int32_t i = 0; i < w; i++) dst[i] |= src[i];
}
static bool set_has(const uint64_t *set, int32_t j) { return (set[j >> 6] >> (j & 63)) & 1; }
static bool set_any(const uint64_t *set, int32_t w) {
	for (int32_t i = 0; i < w; i++)
		if (set[i]) return true;
	return false;
}

// elements of a (possibly nested) array type: float[8][8] holds 64; 0 for a non-array
static int64_t array_elements(const svsl_types_t *types, svsl_type_id_t type) {
	const svsl_type_t *t = svsl_type_get(types, type);
	if (t->kind != svsl_type_array) return 0;
	int64_t inner = array_elements(types, t->elem);
	return (int64_t)t->array_count * (inner > 0 ? inner : 1);
}

// scalars a value of `type` occupies once split into registers: float4[32] is 128
static int64_t scalar_count(const svsl_types_t *types, svsl_type_id_t type) {
	const svsl_type_t *t = svsl_type_get(types, type);
	switch (t->kind) {
	case svsl_type_vector: return t->count;
	case svsl_type_matrix: return (int64_t)t->rows * t->cols;
	case svsl_type_array:  return (int64_t)t->array_count * scalar_count(types, t->elem);
	case svsl_type_struct: {
		const svsl_struct_info_t *si = &types->structs.items[t->struct_index];
		int64_t sum = 0;
		for (int32_t m = 0; m < si->members.count; m++) sum += scalar_count(types, si->members.items[m].type);
		return sum;
	}
	default:               return 1;
	}
}

// Whether a repeating loop that is not one of `set` (the loops an array's
// indices derive from) encloses instruction k: that loop stays rolled, so every
// element of the split array stays live across its back edge.
static bool in_rolled_loop(const svsl_ir_edit_t *ed, const svsl_ir_cf_t *cf, const int32_t *loop_index,
                           const uint64_t *set, uint32_t k) {
	const svsl_ir_func_t *fn = ed->fn;
	for (uint32_t m = cf->arm[k]; m != SVSL_IR_NONE;) {
		svsl_ir_op_ op     = (svsl_ir_op_)fn->insts.items[m].op;
		uint32_t    opener = op == svsl_ir_if || op == svsl_ir_loop || op == svsl_ir_switch ? m : cf->head[m];
		if (fn->insts.items[opener].op == svsl_ir_loop && (loop_index[opener] < 0 || !set_has(set, loop_index[opener])) &&
		    svsl_ir_cf_loop_repeats(cf, fn, opener)) // (no kills yet: needed_loops runs before any unroll)
			return true;
		m = cf->arm[opener];
	}
	return false;
}

// Which counted loops are worth unrolling: those whose counters index a large
// array that unrolling turns fully constant-indexed (so SROA splits it). Every value
// gets the set of counted loops whose counters it derives from (`mask`), or
// `runtime` when anything else feeds it; loads from a constant table count as
// their indices. An array is promotable when no index into it is runtime, and
// the loops its indices derive from are the ones it needs. Returns the union of
// those sets over the promotable arrays (w words).
static uint64_t *needed_loops(const svsl_ir_edit_t *ed, const svsl_types_t *types, const svsl_ir_cf_t *cf,
                              const counted_t *loops, int32_t nloops, int32_t w) {
	// (a pointer's own mask is its indices'; ptr/const/var bases contribute nothing)
	const svsl_ir_func_t *fn      = ed->fn;
	uint32_t              count   = (uint32_t)fn->insts.count;
	uint64_t             *mask    = svsl_arena_alloc(ed->scratch, (size_t)count * (size_t)w * sizeof(uint64_t));
	uint8_t              *runtime = svsl_arena_alloc(ed->scratch, (size_t)count);
	#define MASK(x) (mask + (size_t)(x) * (size_t)w)
	// each counter var's loops, so a load checks only the loops that step what it reads
	int32_t *first_loop = svsl_arena_alloc_raw(ed->scratch, (size_t)count * sizeof(int32_t));
	int32_t *next_loop  = svsl_arena_alloc_raw(ed->scratch, (size_t)(nloops > 0 ? nloops : 1) * sizeof(int32_t));
	for (uint32_t k = 0; k < count; k++) first_loop[k] = -1;
	for (int32_t j = nloops; j-- > 0;) {
		next_loop[j]                  = first_loop[loops[j].counter];
		first_loop[loops[j].counter] = j;
	}
	for (uint32_t k = 0; k < count; k++) {
		if (is_dead(ed, k)) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		svsl_ir_op_           op = (svsl_ir_op_)in->op;
		if (op == svsl_ir_const) continue;
		if (op == svsl_ir_load) {
			int32_t c = -1; // the innermost counted loop this load reads the counter of (the last in order)
			for (int32_t j = in->args[0] < count ? first_loop[in->args[0]] : -1; j >= 0; j = next_loop[j])
				if (k > loops[j].loop && k < loops[j].end) c = j;
			uint32_t root = svsl_ir_root_ptr(fn, in->args[0]);
			const svsl_ir_inst_t *r = &fn->insts.items[root];
			if (c >= 0) MASK(k)[c >> 6] |= (uint64_t)1 << (c & 63);
			else if (r->op == svsl_ir_ptr && (svsl_ref_)r->args[0] == svsl_ref_const_global && in->args[0] < count) {
				set_or(MASK(k), MASK(in->args[0]), w); runtime[k] = runtime[in->args[0]];
			} else runtime[k] = 1;
			continue;
		}
		if (op == svsl_ir_chain) { // an address: what its indices derive from (a table load reads it)
			for (uint32_t a = 0; a < in->aux_count; a++) {
				uint32_t x = fn->aux.items[in->aux + a];
				if (x < count) { set_or(MASK(k), MASK(x), w); runtime[k] |= runtime[x]; }
			}
			continue;
		}
		if (!svsl_ir_is_pure(in, types) || op == svsl_ir_var) { runtime[k] = 1; continue; }
		uint32_t m = svsl_ir_value_arg_mask(in);
		for (int32_t a = 0; a < 4; a++)
			if ((m & (1u << a)) && in->args[a] < count) { set_or(MASK(k), MASK(in->args[a]), w); runtime[k] |= runtime[in->args[a]]; }
		if (svsl_ir_aux_holds_values(in))
			for (uint32_t a = 0; a < in->aux_count; a++) {
				uint32_t x = fn->aux.items[in->aux + a];
				if (x < count) { set_or(MASK(k), MASK(x), w); runtime[k] |= runtime[x]; }
			}
	}

	// per array var: the union of its array-level index masks (in the var's own
	// slot - a var has no indices), and whether any index is runtime
	uint8_t *bad = svsl_arena_alloc(ed->scratch, (size_t)count);
	for (uint32_t k = 0; k < count; k++) {
		if (is_dead(ed, k)) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		uint32_t              m  = svsl_ir_value_arg_mask(in);
		for (int32_t a = 0; a < 4; a++) {
			uint32_t v = in->args[a];
			if (!(m & (1u << a)) || v >= count || fn->insts.items[v].op != svsl_ir_var) continue;
			if (svsl_type_get(types, fn->insts.items[v].type)->kind != svsl_type_array) continue;
			if (in->op == svsl_ir_chain && a == 0 && in->aux_count > 0) {
				// every index that selects an array level (SROA splits nested
				// arrays one level per round); vector/struct indices after them don't
				svsl_type_id_t at = fn->insts.items[v].type;
				for (uint32_t i = 0; i < in->aux_count; i++) {
					const svsl_type_t *t = svsl_type_get(types, at);
					if (t->kind != svsl_type_array) break;
					uint32_t x = fn->aux.items[in->aux + i];
					set_or(MASK(v), MASK(x), w);
					bad[v] |= runtime[x];
					at      = t->elem;
				}
			} else if (!(in->op == svsl_ir_load || (in->op == svsl_ir_store && a == 0))) {
				bad[v] = 1; // a use SROA can't take apart
			}
		}
	}
	// the register budget an array gets: smaller while a loop that stays rolled
	// encloses any access to it
	int32_t *loop_index = svsl_arena_alloc(ed->scratch, (size_t)count * sizeof(int32_t));
	for (uint32_t k = 0; k < count; k++) loop_index[k] = -1;
	for (int32_t j = 0; j < nloops; j++) loop_index[loops[j].loop] = j;
	uint8_t *hot = svsl_arena_alloc(ed->scratch, (size_t)count);
	for (uint32_t k = 0; k < count; k++) {
		if (is_dead(ed, k)) continue;
		const svsl_ir_inst_t *in = &fn->insts.items[k];
		if (in->op != svsl_ir_chain && in->op != svsl_ir_load && in->op != svsl_ir_store) continue;
		uint32_t v = in->args[0];
		if (v < count && fn->insts.items[v].op == svsl_ir_var && !hot[v] && set_any(MASK(v), w))
			hot[v] = in_rolled_loop(ed, cf, loop_index, MASK(v), k);
	}

	// all or nothing: an array whose loops don't all fit the budget stays an array,
	// so unrolling only some of them would buy nothing but code
	uint64_t *needed = svsl_arena_alloc(ed->scratch, (size_t)w * sizeof(uint64_t));
	for (uint32_t k = 0; k < count; k++) {
		if (is_dead(ed, k) || fn->insts.items[k].op != svsl_ir_var || bad[k] || !set_any(MASK(k), w) ||
		    array_elements(types, fn->insts.items[k].type) < UNROLL_MIN_ELEMENTS ||
		    scalar_count(types, fn->insts.items[k].type) > (hot[k] ? UNROLL_MAX_SCALARS_IN_LOOP : UNROLL_MAX_SCALARS))
			continue;
		uint64_t cost = 0; // every loop nest this array needs, fully unrolled
		for (int32_t j = 0; j < nloops; j++)
			if (set_has(MASK(k), j)) cost += loops[j].cost;
		if (cost <= UNROLL_BUDGET) set_or(needed, MASK(k), w);
	}
	#undef MASK
	return needed;
}

void svsl_ir_unroll(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	(void)level;
	svsl_ir_func_t *fn     = ed->fn;
	int32_t         marked = 0; // [unroll] loops; none, or no large local array: nothing to do
	bool            arr    = false;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		marked += in->op == svsl_ir_loop && (in->args[0] & SpvLoopControlUnrollMask);
		arr    |= in->op == svsl_ir_var && array_elements(&prog->types, in->type) >= UNROLL_MIN_ELEMENTS;
	}
	if (!marked || !arr) return;
	svsl_ir_cf_t cf;
	svsl_ir_cf_build(ed->scratch, fn, &cf);

	counted_t *loops  = svsl_arena_alloc(ed->scratch, (size_t)marked * sizeof(counted_t));
	int32_t    nloops = 0;
	for (uint32_t l = 0; l < (uint32_t)fn->insts.count; l++) {
		const svsl_ir_inst_t *in = &fn->insts.items[l];
		if (in->op != svsl_ir_loop || is_dead(ed, l)) continue;
		if (!(in->args[0] & SpvLoopControlUnrollMask)) continue; // [unroll] only
		if (counted_loop(ed, &cf, &prog->types, l, &loops[nloops])) nloops++;
	}
	if (!nloops) return;

	// full-nest cost, inner loops first (an inner loop's marker comes later): a
	// loop's copies each hold its counted children already unrolled
	for (int32_t j = nloops; j-- > 0;) {
		uint64_t body = loops[j].size;
		for (int32_t c = j + 1; c < nloops; c++) {
			if (loops[c].loop >= loops[j].end) break;
			bool direct = true; // no counted loop between j and c
			for (int32_t m = j + 1; m < c && direct; m++)
				direct = !(loops[c].loop > loops[m].loop && loops[c].loop < loops[m].end);
			if (direct) body += loops[c].cost - loops[c].size;
		}
		loops[j].cost = (uint64_t)loops[j].trips * body;
	}

	const uint64_t *needed = needed_loops(ed, &prog->types, &cf, loops, nloops, (nloops + 63) / 64);
	for (int32_t j = 0; j < nloops; j++)
		if (set_has(needed, j) && loops[j].innermost) unroll(ed, &loops[j]);
}

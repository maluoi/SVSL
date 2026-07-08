// Peephole simplification: local pattern rewrites that replace an instruction
// with an already-computed, dominating value. Because every replacement target
// is an operand of the matched instruction (or an operand's operand), it always
// dominates the use — no region tracking needed. Duplicates are nopped and
// their users redirected, exactly like CSE.
//
// -O1 (value-preserving + exact integer identities):
//   * extract(construct C, k)      -> C's k-th component   (#5)
//   * shuffle(v, identity)         -> v                    (#9)
//   * integer x+0 / x-0 / x*1 / x*0 / x/1 / x|0 / x^0 / x<<0 / x>>0 / x&x / x|x  (#12)
// -O2 (float-algebraic, may change IEEE edge cases — not oracle-covered):
//   * float x*1.0 / x/1.0          -> x                    (#13)
//
// See docs/OPTIMIZATION_PLAN.md §4.

#include "../ir.h"
#include "../ir_operands.h"

static bool is_scalar_int(const svsl_types_t *types, svsl_type_id_t tid) {
	const svsl_type_t *t = svsl_type_get(types, tid);
	return t->kind == svsl_type_scalar &&
	       t->scalar >= svsl_scalar_int8 && t->scalar <= svsl_scalar_uint64;
}

static bool is_scalar_float(const svsl_types_t *types, svsl_type_id_t tid) {
	const svsl_type_t *t = svsl_type_get(types, tid);
	return t->kind == svsl_type_scalar &&
	       (t->scalar == svsl_scalar_float32 || t->scalar == svsl_scalar_half);
}

static bool const_is(const svsl_ir_func_t *fn, uint32_t id, uint64_t want) {
	const svsl_ir_inst_t *in = &fn->insts.items[id];
	return in->op == svsl_ir_const &&
	       ((uint64_t)in->args[0] | ((uint64_t)in->args[1] << 32)) == want;
}

// float 1.0 as a 32-bit pattern (half is stored as relaxed float32 here)
static bool const_is_f1(const svsl_ir_func_t *fn, uint32_t id) {
	const svsl_ir_inst_t *in = &fn->insts.items[id];
	return in->op == svsl_ir_const && in->args[1] == 0 && in->args[0] == 0x3F800000u;
}

// component count of a vector type (1 for scalars)
static int32_t comp_count(const svsl_types_t *types, svsl_type_id_t tid) {
	const svsl_type_t *t = svsl_type_get(types, tid);
	return t->kind == svsl_type_vector ? t->count : 1;
}

static bool is_float_vector(const svsl_types_t *types, svsl_type_id_t tid) {
	const svsl_type_t *t = svsl_type_get(types, tid);
	return t->kind == svsl_type_vector &&
	       (t->scalar == svsl_scalar_float32 || t->scalar == svsl_scalar_half ||
	        t->scalar == svsl_scalar_float16 || t->scalar == svsl_scalar_float64);
}

// If `id` is a construct that splats a single scalar across `width` components,
// return that scalar's value id; otherwise SVSL_IR_NONE.
static uint32_t splat_source(const svsl_ir_func_t *fn, const svsl_types_t *types,
                             uint32_t id, int32_t width) {
	const svsl_ir_inst_t *c = &fn->insts.items[id];
	if (c->op != svsl_ir_construct || (int32_t)c->aux_count != width) return SVSL_IR_NONE;
	uint32_t s = fn->aux.items[c->aux];
	if (svsl_type_get(types, fn->insts.items[s].type)->kind != svsl_type_scalar) return SVSL_IR_NONE;
	for (uint32_t k = 1; k < c->aux_count; k++)
		if (fn->aux.items[c->aux + k] != s) return SVSL_IR_NONE;
	return s;
}

// Vector×scalar in disguise: a float-vector `mul` whose operand is a scalar splat
// construct is really OpVectorTimesScalar. Rewrite that operand to the splatted
// scalar in place — the splat construct then dies (DCE) and emit picks the
// single-instruction encoding instead of splat + component-wise FMul. Bit-exact:
// both multiply every component by the same scalar. Returns true if rewritten.
static bool strip_vector_splat(svsl_ir_func_t *fn, const svsl_types_t *types, uint32_t i) {
	svsl_ir_inst_t *in = &fn->insts.items[i];
	if (in->op != svsl_ir_mul || !is_float_vector(types, in->type)) return false;
	int32_t width = comp_count(types, in->type);
	for (int32_t w = 0; w < 2; w++) {
		if (svsl_type_get(types, fn->insts.items[in->args[w ^ 1]].type)->kind != svsl_type_vector)
			continue; // the other operand must stay a vector
		uint32_t s = splat_source(fn, types, in->args[w], width);
		if (s != SVSL_IR_NONE) { in->args[w] = s; return true; }
	}
	return false;
}

// Fold a single-source shuffle (swizzle) that feeds an extract or another
// shuffle, rewriting the consumer in place to read the shuffle's source
// directly. svsl shuffle: args[0] = source vector, args[1] = packed 4-bit lane
// indices, args[2] = lane count; every emitted lane is < the source's width.
// extract(shuffle(v, lanes), k)          -> extract(v, lanes[k])
// shuffle(shuffle(v, lanes1), lanes2, n) -> shuffle(v, lanes1∘lanes2, n)
// The inner shuffle then dies (DCE). Bit-exact: the mapped lane names the same
// component. Returns true if rewritten.
static bool compose_shuffle(svsl_ir_func_t *fn, uint32_t i) {
	svsl_ir_inst_t *in = &fn->insts.items[i];
	if (in->op == svsl_ir_extract) {
		const svsl_ir_inst_t *s = &fn->insts.items[in->args[0]];
		if (s->op != svsl_ir_shuffle || in->args[1] >= s->args[2]) return false;
		in->args[0] = s->args[0];
		in->args[1] = (s->args[1] >> (in->args[1] * 4)) & 0xF;
		return true;
	}
	if (in->op == svsl_ir_shuffle) {
		const svsl_ir_inst_t *s = &fn->insts.items[in->args[0]];
		if (s->op != svsl_ir_shuffle) return false;
		uint32_t mask = 0;
		for (uint32_t j = 0; j < in->args[2]; j++) {
			uint32_t outer = (in->args[1] >> (j * 4)) & 0xF;
			if (outer >= s->args[2]) return false;
			mask |= ((s->args[1] >> (outer * 4)) & 0xF) << (j * 4);
		}
		in->args[0] = s->args[0];
		in->args[1] = mask;
		return true;
	}
	return false;
}

// Try to rewrite inst i to an existing value; returns that value id or i itself.
static uint32_t simplify(const svsl_ir_func_t *fn, const svsl_types_t *types,
                         uint32_t i, svsl_opt_level_ level) {
	const svsl_ir_inst_t *in = &fn->insts.items[i];
	uint32_t a = in->args[0], b = in->args[1];

	switch ((svsl_ir_op_)in->op) {
	case svsl_ir_extract: { // extract(construct C, k) -> C.aux[k]
		const svsl_ir_inst_t *c = &fn->insts.items[a];
		if (c->op == svsl_ir_construct && b < c->aux_count) {
			uint32_t comp = fn->aux.items[c->aux + b];
			if (fn->insts.items[comp].type == in->type) return comp;
		}
		return i;
	}
	case svsl_ir_shuffle: { // identity shuffle over a whole vector -> the vector
		uint32_t src   = a, packed = b, n = in->args[2];
		if (in->type == fn->insts.items[src].type &&
		    (int32_t)n == comp_count(types, fn->insts.items[src].type)) {
			for (uint32_t k = 0; k < n; k++)
				if (((packed >> (k * 4)) & 0xF) != k) return i;
			return src;
		}
		return i;
	}
	case svsl_ir_add: // x+0, 0+x
		if (is_scalar_int(types, in->type)) {
			if (const_is(fn, b, 0)) return a;
			if (const_is(fn, a, 0)) return b;
		}
		return i;
	case svsl_ir_sub: // x-0
		if (is_scalar_int(types, in->type) && const_is(fn, b, 0)) return a;
		return i;
	case svsl_ir_mul:
		if (is_scalar_int(types, in->type)) {
			if (const_is(fn, b, 1)) return a;
			if (const_is(fn, a, 1)) return b;
			if (const_is(fn, b, 0)) return b; // x*0 -> 0 (exact for integers)
			if (const_is(fn, a, 0)) return a;
		} else if (level >= svsl_opt_aggressive && is_scalar_float(types, in->type)) {
			if (const_is_f1(fn, b)) return a; // x*1.0 (IEEE edge cases: -O2 only)
			if (const_is_f1(fn, a)) return b;
		}
		return i;
	case svsl_ir_div: // x/1, x/1.0
		if (is_scalar_int(types, in->type) && const_is(fn, b, 1)) return a;
		if (level >= svsl_opt_aggressive && is_scalar_float(types, in->type) && const_is_f1(fn, b))
			return a;
		return i;
	case svsl_ir_bit_or:  // x|0, 0|x, x|x
		if (is_scalar_int(types, in->type)) {
			if (const_is(fn, b, 0)) return a;
			if (const_is(fn, a, 0)) return b;
			if (a == b) return a;
		}
		return i;
	case svsl_ir_bit_xor: // x^0, 0^x
		if (is_scalar_int(types, in->type)) {
			if (const_is(fn, b, 0)) return a;
			if (const_is(fn, a, 0)) return b;
		}
		return i;
	case svsl_ir_bit_and: // x&x
		if (is_scalar_int(types, in->type) && a == b) return a;
		return i;
	case svsl_ir_shl: case svsl_ir_shr: // x<<0, x>>0
		if (is_scalar_int(types, in->type) && const_is(fn, b, 0)) return a;
		return i;
	default:
		return i;
	}
}

bool svsl_ir_peephole(svsl_arena_t *arena, svsl_ir_func_t *fn,
                      const svsl_types_t *types, svsl_opt_level_ level) {
	int32_t count = fn->insts.count;
	if (count == 0) return false;

	uint32_t *remap = svsl_arena_alloc(arena, (size_t)count * sizeof(uint32_t));
	for (int32_t i = 0; i < count; i++) remap[i] = (uint32_t)i;

	// Detect on raw operands — patterns that only appear after another rewrite
	// compose on the driver's next fixpoint iteration, so no per-instruction
	// canonicalization is needed here (keeping this pass cheap when it is idle).
	bool changed = false;
	for (int32_t i = 0; i < count; i++) {
		uint32_t to = simplify(fn, types, (uint32_t)i, level);
		if (to != (uint32_t)i) { remap[i] = to; changed = true; }
	}

	// one sweep: redirect every user to the surviving value, then nop the rest
	if (changed) {
		for (int32_t i = 0; i < count; i++) svsl_ir_remap_operands(fn, (uint32_t)i, remap, (uint32_t)count);
		for (int32_t i = 0; i < count; i++)
			if (remap[i] != (uint32_t)i) {
				svsl_ir_inst_t *inst = &fn->insts.items[i];
				inst->op        = svsl_ir_nop;
				inst->type      = SVSL_TYPE_NONE;
				inst->aux_count = 0;
			}
	}

	// in-place operand rewrites on the now-canonical IR (no value replacement):
	// strip vector×scalar splats so DCE reclaims the dead splat constructs, and
	// collapse shuffle-of-shuffle / extract-of-shuffle chains onto their source.
	bool stripped = false;
	for (int32_t i = 0; i < count; i++) {
		stripped |= strip_vector_splat(fn, types, (uint32_t)i);
		stripped |= compose_shuffle(fn, (uint32_t)i);
	}

	return changed || stripped;
}

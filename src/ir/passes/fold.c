// Constant folding: arithmetic on constant operands collapses in place.
// Instructions keep their indices (users are untouched); DCE sweeps the
// now-unused constants afterwards.

#include "../ir.h"

#include <string.h>

typedef union bits64_t {
	uint64_t u;
	int64_t  i;
	double   d;
} bits64_t;

static bool const_bits(const svsl_ir_func_t *fn, uint32_t id, uint64_t *out) {
	const svsl_ir_inst_t *inst = &fn->insts.items[id];
	if (inst->op != svsl_ir_const) return false;
	*out = (uint64_t)inst->args[0] | ((uint64_t)inst->args[1] << 32);
	return true;
}

static bool scalar_is_float_kind(const svsl_types_t *types, svsl_type_id_t id) {
	const svsl_type_t *t = svsl_type_get(types, id);
	if (t->kind != svsl_type_scalar) return false;
	return t->scalar == svsl_scalar_half || t->scalar == svsl_scalar_float16 ||
	       t->scalar == svsl_scalar_float32 || t->scalar == svsl_scalar_float64;
}

static int32_t int_bit_width(svsl_scalar_ s) {
	switch (s) {
	case svsl_scalar_int8:  case svsl_scalar_uint8:  return 8;
	case svsl_scalar_int16: case svsl_scalar_uint16: return 16;
	case svsl_scalar_int64: case svsl_scalar_uint64: return 64;
	default:                                         return 32; // int32/uint32
	}
}
static bool int_is_signed(svsl_scalar_ s) {
	return s == svsl_scalar_int8  || s == svsl_scalar_int16 ||
	       s == svsl_scalar_int32 || s == svsl_scalar_int64;
}
// interpret the low `w` bits of `bits` as a two's-complement signed value
static int64_t sign_ext(uint64_t bits, int32_t w) {
	if (w >= 64) return (int64_t)bits;
	uint64_t m = (uint64_t)1 << (w - 1);
	uint64_t v = bits & (((uint64_t)1 << w) - 1);
	return (int64_t)((v ^ m) - m);
}

static void replace_with_const(svsl_ir_inst_t *inst, uint64_t bits, bool *ref_changed) {
	*ref_changed    = true;
	inst->op        = svsl_ir_const;
	inst->args[0]   = (uint32_t)bits;
	inst->args[1]   = (uint32_t)(bits >> 32);
	inst->args[2]   = 0;
	inst->args[3]   = SVSL_IR_NONE;
	inst->aux_count = 0;
}

bool svsl_ir_fold(svsl_ir_func_t *fn, const svsl_types_t *types) {
	bool changed = false;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->type == SVSL_TYPE_NONE) continue;
		const svsl_type_t *t = svsl_type_get(types, inst->type);
		if (t->kind != svsl_type_scalar) continue; // scalar folding only (vectors via components)

		uint64_t a, v;
		bool     is_float = scalar_is_float_kind(types, inst->type);
		// float16 constants hold 16-bit patterns; folding them with the float32
		// interpretation below would corrupt them, so they stay runtime ops
		bool foldable_float = is_float && t->scalar != svsl_scalar_float64 &&
		                                  t->scalar != svsl_scalar_float16;

		switch ((svsl_ir_op_)inst->op) {
		case svsl_ir_add: case svsl_ir_sub: case svsl_ir_mul:
		case svsl_ir_div: case svsl_ir_rem: {
			if (!const_bits(fn, inst->args[0], &a) || !const_bits(fn, inst->args[1], &v)) break;
			if (foldable_float) {
				float fa, fb;
				uint32_t ua = (uint32_t)a, ub = (uint32_t)v;
				memcpy(&fa, &ua, 4);
				memcpy(&fb, &ub, 4);
				float r = inst->op == svsl_ir_add ? fa + fb :
				          inst->op == svsl_ir_sub ? fa - fb :
				          inst->op == svsl_ir_mul ? fa * fb :
				          inst->op == svsl_ir_div ? (fb != 0 ? fa / fb : 0) : 0;
				if ((inst->op == svsl_ir_div || inst->op == svsl_ir_rem) && fb == 0) break;
				if (inst->op == svsl_ir_rem) break; // frem folding not worth the ULP risk
				uint32_t bits;
				memcpy(&bits, &r, 4);
				replace_with_const(inst, bits, &changed);
			} else if (!is_float) {
				// fold at the scalar's true width and signedness: add/sub/mul agree in the
				// low bits either way, but div/rem and any 64-bit type need the real type
				int32_t  w    = int_bit_width(t->scalar);
				uint64_t mask = w >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << w) - 1);
				uint64_t r;
				if (int_is_signed(t->scalar)) {
					int64_t ia = sign_ext(a, w), ib = sign_ext(v, w);
					if ((inst->op == svsl_ir_div || inst->op == svsl_ir_rem) && ib == 0) break;
					bool ovf = ia == INT64_MIN && ib == -1; // INT_MIN/-1 overflows
					r = inst->op == svsl_ir_add ? (uint64_t)ia + (uint64_t)ib :
					    inst->op == svsl_ir_sub ? (uint64_t)ia - (uint64_t)ib :
					    inst->op == svsl_ir_mul ? (uint64_t)ia * (uint64_t)ib :
					    inst->op == svsl_ir_div ? (uint64_t)(ovf ? INT64_MIN : ia / ib)
					                            : (uint64_t)(ovf ? 0         : ia % ib);
				} else {
					uint64_t ua = a & mask, ub = v & mask;
					if ((inst->op == svsl_ir_div || inst->op == svsl_ir_rem) && ub == 0) break;
					r = inst->op == svsl_ir_add ? ua + ub :
					    inst->op == svsl_ir_sub ? ua - ub :
					    inst->op == svsl_ir_mul ? ua * ub :
					    inst->op == svsl_ir_div ? ua / ub : ua % ub;
				}
				replace_with_const(inst, r & mask, &changed);
			}
			break;
		}
		case svsl_ir_neg: {
			if (!const_bits(fn, inst->args[0], &a)) break;
			if (foldable_float) {
				uint32_t ua = (uint32_t)a;
				float    fa;
				memcpy(&fa, &ua, 4);
				fa = -fa;
				uint32_t bits;
				memcpy(&bits, &fa, 4);
				replace_with_const(inst, bits, &changed);
			} else if (!is_float) {
				// unsigned negate (signed -MIN overflows), at the scalar's true width
				int32_t  w    = int_bit_width(t->scalar);
				uint64_t mask = w >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << w) - 1);
				replace_with_const(inst, ((uint64_t)0 - (a & mask)) & mask, &changed);
			}
			break;
		}
		case svsl_ir_convert: { // int literal → float constant is the common case
			if (!const_bits(fn, inst->args[0], &a)) break;
			const svsl_type_t *from = svsl_type_get(types, fn->insts.items[inst->args[0]].type);
			if (from->kind != svsl_type_scalar) break;
			bool from_float = from->scalar == svsl_scalar_half || from->scalar == svsl_scalar_float32 ||
			                  from->scalar == svsl_scalar_float16 || from->scalar == svsl_scalar_float64;
			if (!from_float && foldable_float &&
			    from->scalar != svsl_scalar_int64 && from->scalar != svsl_scalar_uint64) {
				float r = from->scalar == svsl_scalar_uint32 ? (float)(uint32_t)a
				                                             : (float)(int32_t)(uint32_t)a;
				uint32_t bits;
				memcpy(&bits, &r, 4);
				replace_with_const(inst, bits, &changed);
			} else if (from_float && !is_float &&
			           from->scalar != svsl_scalar_float64 &&
			           from->scalar != svsl_scalar_float16 &&
			           (t->scalar == svsl_scalar_int32 || t->scalar == svsl_scalar_uint32)) {
				uint32_t ua = (uint32_t)a;
				float    fa;
				memcpy(&fa, &ua, 4);
				// out-of-range float→int is UB in C and undefined in SPIR-V; clamp, NaN → 0
				uint32_t bits;
				if (t->scalar == svsl_scalar_int32)
					bits = fa != fa            ? 0 :
					       fa <= -2147483648.f ? 0x80000000u :
					       fa >=  2147483648.f ? 0x7FFFFFFFu : (uint32_t)(int32_t)fa;
				else
					bits = fa != fa || fa <= 0.f ? 0 :
					       fa >= 4294967296.f    ? 0xFFFFFFFFu : (uint32_t)fa;
				replace_with_const(inst, bits, &changed);
			}
			break;
		}
		default:
			break;
		}
	}
	return changed;
}

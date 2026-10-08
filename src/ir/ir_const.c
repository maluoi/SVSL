// See ir_const.h.

#include "ir_const.h"
#include "ir_operands.h"
#include "../sema/const_eval.h"

#include <string.h>

// How a scalar folds: an integer of some width/signedness, a float32, a bool,
// or not at all (float16/float64).
typedef enum lane_cls_ { lane_none = 0, lane_int, lane_float, lane_bool } lane_cls_;
typedef struct lane_kind_t {
	uint8_t cls;   // lane_cls_
	uint8_t width; // integer bits
	bool    sgn;
} lane_kind_t;

static lane_kind_t lane_kind(svsl_scalar_ s) {
	switch (s) {
	case svsl_scalar_bool:    return (lane_kind_t){ lane_bool, 1, false };
	case svsl_scalar_int8:    return (lane_kind_t){ lane_int, 8, true };
	case svsl_scalar_uint8:   return (lane_kind_t){ lane_int, 8, false };
	case svsl_scalar_int16:   return (lane_kind_t){ lane_int, 16, true };
	case svsl_scalar_uint16:  return (lane_kind_t){ lane_int, 16, false };
	case svsl_scalar_int32:   return (lane_kind_t){ lane_int, 32, true };
	case svsl_scalar_uint32:  return (lane_kind_t){ lane_int, 32, false };
	case svsl_scalar_int64:   return (lane_kind_t){ lane_int, 64, true };
	case svsl_scalar_uint64:  return (lane_kind_t){ lane_int, 64, false };
	case svsl_scalar_half:
	case svsl_scalar_float32: return (lane_kind_t){ lane_float, 32, false };
	default:                  return (lane_kind_t){ lane_none, 0, false };
	}
}

// the low `w` bits set (w in 1..64)
static uint64_t width_mask(int32_t w) {
	return w >= 64 ? ~(uint64_t)0 : (((uint64_t)1 << w) - 1);
}
// interpret the low `w` bits of `bits` as a two's-complement signed value
static int64_t sign_ext(uint64_t bits, int32_t w) {
	if (w >= 64) return (int64_t)bits;
	uint64_t m = (uint64_t)1 << (w - 1);
	uint64_t v = bits & width_mask(w);
	return (int64_t)((v ^ m) - m);
}
// svsl_ir_const's integer encoding: signed sign-extended, unsigned zero-extended
static uint64_t int_canon(uint64_t bits, lane_kind_t k) {
	return k.sgn ? (uint64_t)sign_ext(bits, k.width) : bits & width_mask(k.width);
}
static float    as_f(uint64_t bits) { float f; uint32_t u = (uint32_t)bits; memcpy(&f, &u, 4); return f; }
static uint64_t f_bits(float f)     { uint32_t u; memcpy(&u, &f, 4); return u; }

static int32_t int_bit_width(svsl_scalar_ s) {
	switch (s) {
	case svsl_scalar_int8:  case svsl_scalar_uint8:  return 8;
	case svsl_scalar_int16: case svsl_scalar_uint16: return 16;
	case svsl_scalar_int64: case svsl_scalar_uint64: return 64;
	default:                                         return 32; // int32/uint32
	}
}
static bool scalar_is_int(svsl_scalar_ s) {
	return s >= svsl_scalar_int8 && s <= svsl_scalar_uint64;
}
static bool int_is_signed(svsl_scalar_ s) {
	return s == svsl_scalar_int8  || s == svsl_scalar_int16 ||
	       s == svsl_scalar_int32 || s == svsl_scalar_int64;
}

bool svsl_ir_int_convert_bits(uint64_t bits, svsl_scalar_ from, svsl_scalar_ to, uint64_t *out_bits) {
	if (!scalar_is_int(from) || !scalar_is_int(to)) return false;
	int32_t  fw = int_bit_width(from), tw = int_bit_width(to);
	uint64_t v  = int_is_signed(from) ? (uint64_t)sign_ext(bits, fw) : bits & width_mask(fw);
	*out_bits   = int_is_signed(to)   ? (uint64_t)sign_ext(v, tw)    : v & width_mask(tw);
	return true;
}

// --- the constant view ---------------------------------------------------------

static void push_lane(svsl_ir_lanes_t *ref_l, uint64_t bits) {
	if (ref_l->count < 4) ref_l->bits[ref_l->count] = bits;
	ref_l->count++;
}

// Lanes of a constant scalar/vector value (constructs flattened); false when
// any lane isn't constant.
// (through the edit's replacements: a value folded earlier in the same pass
// already reads as its constant)
static bool lanes_of(const svsl_ir_edit_t *ed, uint32_t id, svsl_ir_lanes_t *ref_l) {
	id = svsl_ir_edit_resolve(ed, id);
	if (id == SVSL_IR_NONE) return false;
	const svsl_ir_inst_t *in = svsl_ir_edit_inst(ed, id);
	if (in->op == svsl_ir_const) {
		push_lane(ref_l, (uint64_t)in->args[0] | ((uint64_t)in->args[1] << 32));
		return true;
	}
	if (in->op != svsl_ir_construct) return false;
	for (uint32_t k = 0; k < in->aux_count; k++)
		if (!lanes_of(ed, ed->fn->aux.items[in->aux + k], ref_l)) return false;
	return ref_l->count <= 4;
}

bool svsl_ir_const_lanes(const svsl_ir_edit_t *ed, uint32_t id, svsl_ir_lanes_t *out) {
	*out = (svsl_ir_lanes_t){0};
	return lanes_of(ed, id, out);
}

bool svsl_ir_const_splat(const svsl_ir_edit_t *ed, uint32_t id, uint64_t *out_bits) {
	svsl_ir_lanes_t l;
	if (!svsl_ir_const_lanes(ed, id, &l) || l.count == 0) return false;
	for (int32_t k = 1; k < l.count; k++)
		if (l.bits[k] != l.bits[0]) return false;
	*out_bits = l.bits[0];
	return true;
}

bool svsl_ir_lane_type(const svsl_types_t *types, svsl_type_id_t type,
                       svsl_scalar_ *out_scalar, int32_t *out_count) {
	if (type == SVSL_TYPE_NONE) return false;
	const svsl_type_t *t = svsl_type_get(types, type);
	if (t->kind != svsl_type_scalar && t->kind != svsl_type_vector) return false;
	*out_scalar = (svsl_scalar_)t->scalar;
	*out_count  = t->kind == svsl_type_vector ? t->count : 1;
	return true;
}

bool svsl_ir_const_global_lanes(const svsl_program_t *prog, uint32_t global, const uint32_t *index,
                                int32_t index_count, svsl_type_id_t type, svsl_ir_lanes_t *out) {
	const svsl_global_t *g = &prog->const_globals.items[global];
	if (!g->value) return false;
	// walk the path through the value's leaf layout (const_eval.h): array
	// elements, matrix rows, vector components
	const svsl_type_t *t      = svsl_type_get(&prog->types, g->type);
	int32_t            offset = 0;
	int32_t            k      = 0;
	for (; k < index_count && t->kind == svsl_type_array; k++) {
		if (index[k] >= (uint32_t)t->array_count) return false;
		offset += (int32_t)index[k] * svsl_const_leaf_count(&prog->types, t->elem);
		t       = svsl_type_get(&prog->types, t->elem);
	}
	svsl_scalar_ s = t->scalar;
	int32_t      n = t->kind == svsl_type_vector ? t->count : t->kind == svsl_type_matrix ? t->cols : 1;
	if (t->kind == svsl_type_matrix) { // a row
		if (k >= index_count || index[k] >= (uint32_t)t->rows) return false;
		offset += (int32_t)index[k++] * t->cols;
	} else if (t->kind != svsl_type_vector && t->kind != svsl_type_scalar) {
		return false;
	}
	if (k < index_count) { // a component
		if (n == 1 || index[k] >= (uint32_t)n) return false;
		offset += (int32_t)index[k++];
		n       = 1;
	}
	svsl_scalar_ ts;
	int32_t      tn;
	if (k != index_count || lane_kind(s).cls == lane_none ||
	    !svsl_ir_lane_type(&prog->types, type, &ts, &tn) || ts != s || tn != n) return false;
	out->count = n;
	memcpy(out->bits, g->value + offset, (size_t)n * sizeof(uint64_t));
	return true;
}

uint32_t svsl_ir_make_const(svsl_ir_edit_t *ed, svsl_types_t *types, uint32_t before,
                            svsl_type_id_t type, const svsl_ir_lanes_t *lanes) {
	svsl_scalar_ s;
	int32_t      n;
	if (!svsl_ir_lane_type(types, type, &s, &n) || n == 1)
		return svsl_ir_edit_const(ed, type, lanes->bits[0]);
	svsl_type_id_t comp = svsl_type_scalar_id(types, s);
	uint32_t       ids[4];
	for (int32_t l = 0; l < n; l++) ids[l] = svsl_ir_edit_const(ed, comp, lanes->bits[l]);
	return svsl_ir_edit_insert(ed, before, (svsl_ir_inst_t){
		.op = svsl_ir_construct, .type = type, .args = { 0, 0, 0, SVSL_IR_NONE } }, ids, (uint32_t)n);
}

// --- per-lane evaluation -----------------------------------------------------------

// a OP b on one lane of kind k (the operands' kind); result in r's encoding.
static bool eval_binary(svsl_ir_op_ op, lane_kind_t k, uint64_t a, uint64_t b, uint64_t *out) {
	if (svsl_ir_op_traits(op) & svsl_ir_trait_compare) {
		bool r;
		if (k.cls == lane_float) {
			float x = as_f(a), y = as_f(b);
			r = op == svsl_ir_eq ? x == y : op == svsl_ir_ne ? x != y : // ordered / unordered
			    op == svsl_ir_lt ? x <  y : op == svsl_ir_le ? x <= y :
			    op == svsl_ir_gt ? x >  y :                    x >= y;
		} else if (k.cls == lane_int && k.sgn) {
			int64_t x = (int64_t)int_canon(a, k), y = (int64_t)int_canon(b, k);
			r = op == svsl_ir_eq ? x == y : op == svsl_ir_ne ? x != y :
			    op == svsl_ir_lt ? x <  y : op == svsl_ir_le ? x <= y :
			    op == svsl_ir_gt ? x >  y :                    x >= y;
		} else {
			uint64_t x = int_canon(a, k), y = int_canon(b, k); // unsigned, or bool (0/1)
			if (k.cls == lane_bool && op != svsl_ir_eq && op != svsl_ir_ne) return false;
			r = op == svsl_ir_eq ? x == y : op == svsl_ir_ne ? x != y :
			    op == svsl_ir_lt ? x <  y : op == svsl_ir_le ? x <= y :
			    op == svsl_ir_gt ? x >  y :                    x >= y;
		}
		*out = r;
		return true;
	}
	if (k.cls == lane_bool) {
		if (op == svsl_ir_log_and) { *out = (a & 1) && (b & 1); return true; }
		if (op == svsl_ir_log_or)  { *out = (a & 1) || (b & 1); return true; }
		return false;
	}
	if (k.cls == lane_float) {
		float x = as_f(a), y = as_f(b), r;
		switch (op) {
		case svsl_ir_add: r = x + y; break;
		case svsl_ir_sub: r = x - y; break;
		case svsl_ir_mul: r = x * y; break;
		case svsl_ir_div: if (y == 0) return false; r = x / y; break;
		default:          return false; // frem: not worth the ULP risk
		}
		*out = f_bits(r);
		return true;
	}
	if (k.cls != lane_int) return false;
	int32_t  w = k.width;
	uint64_t r;
	switch (op) {
	case svsl_ir_add:     r = a + b; break; // wrapping: the low bits agree either way
	case svsl_ir_sub:     r = a - b; break;
	case svsl_ir_mul:     r = a * b; break;
	case svsl_ir_bit_and: r = a & b; break;
	case svsl_ir_bit_or:  r = a | b; break;
	case svsl_ir_bit_xor: r = a ^ b; break;
	case svsl_ir_div: case svsl_ir_rem:
		if (k.sgn) {
			int64_t x = sign_ext(a, w), y = sign_ext(b, w);
			if (y == 0 || (y == -1 && x == sign_ext((uint64_t)1 << (w - 1), w))) return false;
			r = (uint64_t)(op == svsl_ir_div ? x / y : x % y);
		} else {
			uint64_t x = a & width_mask(w), y = b & width_mask(w);
			if (y == 0) return false;
			r = op == svsl_ir_div ? x / y : x % y;
		}
		break;
	default:
		return false; // shifts take their amount's own kind: see eval_shift
	}
	*out = int_canon(r, k);
	return true;
}

static bool eval_shift(svsl_ir_op_ op, lane_kind_t k, lane_kind_t amount_k, uint64_t a, uint64_t b,
                       uint64_t *out) {
	if (k.cls != lane_int || amount_k.cls != lane_int) return false;
	int64_t s = amount_k.sgn ? sign_ext(b, amount_k.width) : (int64_t)(b & width_mask(amount_k.width));
	if (s < 0 || s >= k.width) return false; // undefined in SPIR-V
	uint64_t r = op == svsl_ir_shl ? a << s
	           : k.sgn            ? (uint64_t)(sign_ext(a, k.width) >> s) // arithmetic
	                              : (a & width_mask(k.width)) >> s;
	*out = int_canon(r, k);
	return true;
}

static bool eval_unary(svsl_ir_op_ op, lane_kind_t k, uint64_t a, uint64_t *out) {
	if (op == svsl_ir_log_not) {
		if (k.cls != lane_bool) return false;
		*out = !(a & 1);
		return true;
	}
	if (op == svsl_ir_neg && k.cls == lane_float) { *out = f_bits(-as_f(a)); return true; }
	if (k.cls != lane_int) return false;
	if (op == svsl_ir_neg)     { *out = int_canon((uint64_t)0 - a, k); return true; }
	if (op == svsl_ir_bit_not) { *out = int_canon(~a, k); return true; }
	return false;
}

static bool eval_convert(lane_kind_t from, svsl_scalar_ from_s, lane_kind_t to, svsl_scalar_ to_s,
                         uint64_t a, uint64_t *out) {
	if (from.cls == lane_bool) {
		if (to.cls == lane_int)   { *out = a & 1; return true; }
		if (to.cls == lane_float) { *out = f_bits((a & 1) ? 1.0f : 0.0f); return true; }
		return false;
	}
	if (to.cls == lane_bool) { // value != 0, unordered for floats (NaN -> true)
		if (from.cls == lane_int)   { *out = int_canon(a, from) != 0; return true; }
		if (from.cls == lane_float) { *out = as_f(a) != 0.0f; return true; }
		return false;
	}
	if (from.cls == lane_int && to.cls == lane_int) return svsl_ir_int_convert_bits(a, from_s, to_s, out);
	if (from.cls == lane_float && to.cls == lane_float) { *out = a & 0xFFFFFFFFu; return true; } // half <-> float32
	if (from.cls == lane_int && to.cls == lane_float) {
		if (from.width > 32) return false;
		*out = f_bits(from.sgn ? (float)sign_ext(a, from.width) : (float)(a & width_mask(from.width)));
		return true;
	}
	if (from.cls == lane_float && to.cls == lane_int && to.width == 32) {
		// out-of-range float->int is undefined in SPIR-V: any value is correct;
		// clamp (C would be UB), NaN -> 0
		float f = as_f(a);
		uint32_t bits;
		if (to.sgn)
			bits = f != f            ? 0 :
			       f <= -2147483648.f ? 0x80000000u :
			       f >=  2147483648.f ? 0x7FFFFFFFu : (uint32_t)(int32_t)f;
		else
			bits = f != f || f <= 0.f ? 0 :
			       f >= 4294967296.f  ? 0xFFFFFFFFu : (uint32_t)f;
		*out = int_canon(bits, to);
		return true;
	}
	return false;
}

// --- the public evaluators ----------------------------------------------------------

bool svsl_ir_eval_binary(svsl_ir_op_ op, svsl_scalar_ s, uint64_t a, uint64_t b, uint64_t *out) {
	return eval_binary(op, lane_kind(s), a, b, out);
}
bool svsl_ir_eval_shift(svsl_ir_op_ op, svsl_scalar_ s, svsl_scalar_ amount, uint64_t a, uint64_t b,
                        uint64_t *out) {
	return eval_shift(op, lane_kind(s), lane_kind(amount), a, b, out);
}
bool svsl_ir_eval_unary(svsl_ir_op_ op, svsl_scalar_ s, uint64_t a, uint64_t *out) {
	return eval_unary(op, lane_kind(s), a, out);
}
bool svsl_ir_eval_convert(svsl_scalar_ from, svsl_scalar_ to, uint64_t a, uint64_t *out) {
	return eval_convert(lane_kind(from), from, lane_kind(to), to, a, out);
}

// See const_eval.h.

#include "const_eval.h"

#include "../front/ast.h"

#include <string.h>

// one scalar during evaluation: floats as double (rounded at the leaf), the
// rest as canonical integer bits
typedef union lane_t { double f; uint64_t u; } lane_t;

// a scalar, vector, or matrix value; `type` is its shape
typedef struct value_t {
	svsl_type_id_t type;
	svsl_scalar_   scalar;
	int32_t        count; // lanes: 1, the vector width, or rows * cols
	lane_t         lane[16];
} value_t;

static bool shape_of(const svsl_types_t *types, svsl_type_id_t type, svsl_scalar_ *out_scalar, int32_t *out_count) {
	if (type == SVSL_TYPE_NONE) return false;
	const svsl_type_t *t = svsl_type_get(types, type);
	*out_scalar = t->scalar;
	switch (t->kind) {
	case svsl_type_scalar: *out_count = 1;                return true;
	case svsl_type_vector: *out_count = t->count;         return true;
	case svsl_type_matrix: *out_count = t->rows * t->cols; return true;
	default:               return false;
	}
}

int32_t svsl_const_leaf_count(const svsl_types_t *types, svsl_type_id_t type) {
	svsl_scalar_ s;
	int32_t      n;
	if (shape_of(types, type, &s, &n)) return n;
	const svsl_type_t *t = svsl_type_get(types, type);
	return t->kind == svsl_type_array ? t->array_count * svsl_const_leaf_count(types, t->elem) : 0;
}

// --- scalars -----------------------------------------------------------------------

static bool is_float(svsl_scalar_ s) {
	return s == svsl_scalar_float16 || s == svsl_scalar_float32 || s == svsl_scalar_float64 || s == svsl_scalar_half;
}
static bool is_signed(svsl_scalar_ s) {
	return s == svsl_scalar_int8 || s == svsl_scalar_int16 || s == svsl_scalar_int32 || s == svsl_scalar_int64;
}
static int32_t int_width(svsl_scalar_ s) {
	switch (s) {
	case svsl_scalar_bool:                           return 1;
	case svsl_scalar_int8:  case svsl_scalar_uint8:  return 8;
	case svsl_scalar_int16: case svsl_scalar_uint16: return 16;
	case svsl_scalar_int64: case svsl_scalar_uint64: return 64;
	default:                                         return 32;
	}
}

// integer bits wrapped to the scalar's width, sign- or zero-extended
static uint64_t canon(uint64_t u, svsl_scalar_ s) {
	int32_t w = int_width(s);
	if (w == 64) return u;
	uint64_t mask = (1ull << w) - 1;
	u &= mask;
	return is_signed(s) && (u >> (w - 1)) ? u | ~mask : u;
}

static bool convert_lane(lane_t v, svsl_scalar_ from, svsl_scalar_ to, lane_t *out) {
	if (is_float(from)) {
		if (is_float(to))           { out->f = v.f;               return true; }
		if (to == svsl_scalar_bool) { out->u = v.f != 0;          return true; }
		if (!(v.f > -9.2233720368547758e18 && v.f < 1.8446744073709552e19)) return false; // NaN, out of range
		out->u = canon(v.f < 0 ? (uint64_t)(int64_t)v.f : (uint64_t)v.f, to);
		return true;
	}
	if (is_float(to)) { out->f = is_signed(from) ? (double)(int64_t)v.u : (double)v.u; return true; }
	out->u = to == svsl_scalar_bool ? v.u != 0 : canon(v.u, to);
	return true;
}

static uint64_t f32_bits(double f) { float x = (float)f; uint32_t u; memcpy(&u, &x, 4); return u; }

static uint64_t encode(lane_t v, svsl_scalar_ s) {
	switch (s) {
	case svsl_scalar_float32: case svsl_scalar_half: return f32_bits(v.f);
	case svsl_scalar_float16: return svsl_f32_to_f16_bits((float)v.f);
	case svsl_scalar_float64: { uint64_t u; memcpy(&u, &v.f, 8); return u; }
	default:                  return v.u;
	}
}

static lane_t decode(uint64_t bits, svsl_scalar_ s) {
	lane_t v;
	switch (s) {
	case svsl_scalar_float32: case svsl_scalar_half: {
		uint32_t u = (uint32_t)bits;
		float    x;
		memcpy(&x, &u, 4);
		v.f = x;
		break;
	}
	case svsl_scalar_float16: v.f = svsl_f16_bits_to_f32((uint16_t)bits); break;
	case svsl_scalar_float64: memcpy(&v.f, &bits, 8);                     break;
	default:                  v.u = bits;                                  break;
	}
	return v;
}

// --- values ------------------------------------------------------------------------

// ir_build's convert_value: the scalar conversion, then a splat or truncation
static bool convert_to(svsl_program_t *prog, value_t *ref_v, svsl_type_id_t to) {
	if (to == SVSL_TYPE_NONE || to == ref_v->type) return true;
	svsl_scalar_ ts;
	int32_t      tc;
	if (!shape_of(&prog->types, to, &ts, &tc)) return false;
	bool from_matrix = svsl_type_get(&prog->types, ref_v->type)->kind == svsl_type_matrix;
	bool to_matrix   = svsl_type_get(&prog->types, to)->kind == svsl_type_matrix;
	if (from_matrix || to_matrix) return false; // matrices only convert through constructors
	if (ts != ref_v->scalar)
		for (int32_t i = 0; i < ref_v->count; i++)
			if (!convert_lane(ref_v->lane[i], ref_v->scalar, ts, &ref_v->lane[i])) return false;
	if (ref_v->count == 1)
		for (int32_t i = 1; i < tc; i++) ref_v->lane[i] = ref_v->lane[0];
	else if (tc > ref_v->count)
		return false;
	ref_v->type   = to;
	ref_v->scalar = ts;
	ref_v->count  = tc;
	return true;
}

// a OP b on one lane of scalar s (both operands' type); false when not foldable
static bool binary_lane(svsl_tok_ op, svsl_scalar_ s, lane_t a, lane_t b, lane_t *out) {
	if (is_float(s)) {
		double x = a.f, y = b.f;
		switch (op) {
		case svsl_tok_plus:  out->f = x + y; return true;
		case svsl_tok_minus: out->f = x - y; return true;
		case svsl_tok_star:  out->f = x * y; return true;
		case svsl_tok_slash: if (y == 0) return false; out->f = x / y; return true;
		case svsl_tok_eq:    out->u = x == y; return true; // ordered
		case svsl_tok_neq:   out->u = x != y; return true; // unordered
		case svsl_tok_lt:    out->u = x <  y; return true;
		case svsl_tok_le:    out->u = x <= y; return true;
		case svsl_tok_gt:    out->u = x >  y; return true;
		case svsl_tok_ge:    out->u = x >= y; return true;
		default:             return false;
		}
	}
	uint64_t x = a.u, y = b.u; // canonical: signed values sign-extended
	bool     sgn = is_signed(s);
	switch (op) {
	case svsl_tok_plus:  out->u = canon(x + y, s); return true;
	case svsl_tok_minus: out->u = canon(x - y, s); return true;
	case svsl_tok_star:  out->u = canon(x * y, s); return true;
	case svsl_tok_slash:
	case svsl_tok_percent:
		if (y == 0 || (sgn && (int64_t)x == INT64_MIN && (int64_t)y == -1)) return false;
		if (sgn) out->u = (uint64_t)(op == svsl_tok_slash ? (int64_t)x / (int64_t)y : (int64_t)x % (int64_t)y);
		else     out->u = op == svsl_tok_slash ? x / y : x % y;
		out->u = canon(out->u, s);
		return true;
	case svsl_tok_amp:    out->u = x & y; return true;
	case svsl_tok_pipe:   out->u = x | y; return true;
	case svsl_tok_caret:  out->u = x ^ y; return true;
	case svsl_tok_andand: out->u = x && y; return true;
	case svsl_tok_oror:   out->u = x || y; return true;
	case svsl_tok_shl:
	case svsl_tok_shr:
		if (y >= (uint64_t)int_width(s)) return false; // undefined shift
		out->u = op == svsl_tok_shl ? canon(x << y, s) : sgn ? (uint64_t)((int64_t)x >> y) : x >> y;
		return true;
	case svsl_tok_eq:  out->u = x == y; return true;
	case svsl_tok_neq: out->u = x != y; return true;
	case svsl_tok_lt:  out->u = sgn ? (int64_t)x <  (int64_t)y : x <  y; return true;
	case svsl_tok_le:  out->u = sgn ? (int64_t)x <= (int64_t)y : x <= y; return true;
	case svsl_tok_gt:  out->u = sgn ? (int64_t)x >  (int64_t)y : x >  y; return true;
	case svsl_tok_ge:  out->u = sgn ? (int64_t)x >= (int64_t)y : x >= y; return true;
	default:           return false;
	}
}

static bool is_compare(svsl_tok_ op) {
	return op == svsl_tok_eq || op == svsl_tok_neq || op == svsl_tok_lt ||
	       op == svsl_tok_le || op == svsl_tok_gt  || op == svsl_tok_ge;
}

static value_t scalar_value(svsl_program_t *prog, svsl_scalar_ s, lane_t v) {
	return (value_t){ .type = svsl_type_scalar_id(&prog->types, s),
	                  .scalar = s, .count = 1, .lane = { v } };
}

static bool eval(svsl_program_t *prog, const svsl_ast_expr_t *e, value_t *out);

// a constructor's arguments, flattened to lanes of scalar s (ir_build's flatten_components)
static bool flatten(svsl_program_t *prog, const svsl_ast_expr_t *e, svsl_scalar_ s, value_t *out) {
	out->count = 0;
	for (int32_t i = 0; i < e->ctor.arg_count; i++) {
		value_t arg;
		if (!eval(prog, e->ctor.args[i], &arg)) return false;
		if (svsl_type_get(&prog->types, arg.type)->kind == svsl_type_matrix) return false;
		for (int32_t k = 0; k < arg.count; k++) {
			if (out->count == 16) return false;
			if (!convert_lane(arg.lane[k], arg.scalar, s, &out->lane[out->count++])) return false;
		}
	}
	return true;
}

static bool eval_ctor(svsl_program_t *prog, const svsl_ast_expr_t *e, value_t *out) {
	const svsl_type_t *t = svsl_type_get(&prog->types, e->sema_type);
	svsl_scalar_       s;
	int32_t            n;
	if (!shape_of(&prog->types, e->sema_type, &s, &n)) return false;
	if (t->kind == svsl_type_scalar) {
		if (e->ctor.arg_count != 1 || !eval(prog, e->ctor.args[0], out)) return false;
		return convert_to(prog, out, e->sema_type);
	}
	if (!flatten(prog, e, s, out)) return false;
	if (out->count == 1 && n > 1) { // broadcast; a matrix takes it on the diagonal
		lane_t v = out->lane[0], zero;
		convert_lane((lane_t){ .u = 0 }, svsl_scalar_int32, s, &zero);
		for (int32_t i = 0; i < n; i++)
			out->lane[i] = t->kind != svsl_type_matrix || i / t->cols == i % t->cols ? v : zero;
	} else if (out->count != n) {
		return false;
	}
	out->type   = e->sema_type;
	out->scalar = s;
	out->count  = n;
	return true;
}

// the value of a scalar/vector/matrix expression, in its own sema_type
static bool eval(svsl_program_t *prog, const svsl_ast_expr_t *e, value_t *out) {
	switch (e->kind) {
	case svsl_expr_int_lit: {
		svsl_scalar_ s = e->int_lit.suffix == svsl_suffix_u  ? svsl_scalar_uint32 :
		                 e->int_lit.suffix == svsl_suffix_l  ? svsl_scalar_int64 :
		                 e->int_lit.suffix == svsl_suffix_ul ? svsl_scalar_uint64 : svsl_scalar_int32;
		*out = scalar_value(prog, s, (lane_t){ .u = canon(e->int_lit.value, s) });
		break;
	}
	case svsl_expr_float_lit: {
		svsl_scalar_ s = e->float_lit.suffix == svsl_suffix_h  ? svsl_scalar_half :
		                 e->float_lit.suffix == svsl_suffix_lf ? svsl_scalar_float64 : svsl_scalar_float32;
		*out = scalar_value(prog, s, (lane_t){ .f = e->float_lit.value });
		break;
	}
	case svsl_expr_bool_lit:
		*out = scalar_value(prog, svsl_scalar_bool, (lane_t){ .u = e->bool_lit });
		break;
	case svsl_expr_ident:
		if (e->sema_ref.kind == svsl_ref_enum_const) {
			const svsl_enum_const_t *ec = &prog->enum_consts.items[e->sema_ref.a];
			svsl_scalar_             s  = svsl_type_get(&prog->types, ec->type)->scalar;
			*out = (value_t){ .type = ec->type, .scalar = s, .count = 1,
			                  .lane = { { .u = canon((uint64_t)ec->value, s) } } };
			break;
		}
		if (e->sema_ref.kind == svsl_ref_const_global) {
			const svsl_global_t *g = &prog->const_globals.items[e->sema_ref.a];
			if (!g->value || !shape_of(&prog->types, g->type, &out->scalar, &out->count)) return false;
			out->type = g->type;
			for (int32_t i = 0; i < out->count; i++) out->lane[i] = decode(g->value[i], out->scalar);
			break;
		}
		return false;
	case svsl_expr_unary: {
		if (!eval(prog, e->unary.operand, out)) return false;
		svsl_tok_ op = e->unary.op;
		for (int32_t i = 0; i < out->count; i++) {
			lane_t *v = &out->lane[i];
			if      (op == svsl_tok_plus)                           {}
			else if (op == svsl_tok_minus && is_float(out->scalar)) v->f = -v->f;
			else if (op == svsl_tok_minus)                          v->u = canon(0 - v->u, out->scalar);
			else if (op == svsl_tok_tilde)                          v->u = canon(~v->u, out->scalar);
			else if (op == svsl_tok_not)                            v->u = !v->u; // sema made it bool
			else return false;
		}
		break;
	}
	case svsl_expr_binary: {
		value_t b;
		if (!eval(prog, e->binary.lhs, out) || !eval(prog, e->binary.rhs, &b) ||
		    !convert_to(prog, &b, out->type) || b.count != out->count) return false;
		for (int32_t i = 0; i < out->count; i++)
			if (!binary_lane(e->binary.op, out->scalar, out->lane[i], b.lane[i], &out->lane[i])) return false;
		if (is_compare(e->binary.op)) {
			out->scalar = svsl_scalar_bool;
			out->type   = out->count == 1 ? svsl_type_scalar_id(&prog->types, svsl_scalar_bool)
			                              : svsl_type_vector_id(&prog->types, svsl_scalar_bool, out->count);
		}
		break;
	}
	case svsl_expr_ternary: {
		value_t c, b;
		if (!eval(prog, e->ternary.cond, &c) || !eval(prog, e->ternary.then_expr, out) ||
		    !eval(prog, e->ternary.else_expr, &b) || b.count != out->count) return false;
		for (int32_t i = 0; i < out->count; i++)
			if (!c.lane[c.count == 1 ? 0 : i].u) out->lane[i] = b.lane[i];
		break;
	}
	case svsl_expr_cast: // ir_build converts straight to the annotation
		return eval(prog, e->cast.operand, out) && convert_to(prog, out, e->sema_type);
	case svsl_expr_ctor:
		if (!eval_ctor(prog, e, out)) return false;
		break;
	default:
		return false;
	}
	return convert_to(prog, out, e->sema_type); // the conversion sema annotated
}

// --- initializers ------------------------------------------------------------------

// ir_build's lower_init: init lists by the declared type, anything else converted to it
static bool eval_init(svsl_program_t *prog, const svsl_ast_expr_t *e, svsl_type_id_t type, uint64_t *out) {
	svsl_types_t       *types = &prog->types;
	const svsl_type_t  *t     = svsl_type_get(types, type);
	if (e->kind != svsl_expr_init_list) {
		if (t->kind == svsl_type_array) { // a copy of another constant table
			if (e->kind != svsl_expr_ident || e->sema_ref.kind != svsl_ref_const_global) return false;
			const svsl_global_t *g = &prog->const_globals.items[e->sema_ref.a];
			if (g->type != type || !g->value) return false;
			memcpy(out, g->value, (size_t)svsl_const_leaf_count(types, type) * sizeof(uint64_t));
			return true;
		}
		value_t v;
		if (!eval(prog, e, &v) || !convert_to(prog, &v, type) || v.type != type) return false;
		for (int32_t i = 0; i < v.count; i++) out[i] = encode(v.lane[i], v.scalar);
		return true;
	}
	int32_t        count = e->init_list.count;
	svsl_type_id_t item  = SVSL_TYPE_NONE;
	if      (t->kind == svsl_type_vector && count == t->count)           item = svsl_type_scalar_id(types, t->scalar);
	else if (t->kind == svsl_type_matrix && count == t->rows * t->cols)  item = svsl_type_scalar_id(types, t->scalar);
	else if (t->kind == svsl_type_array  && count == t->array_count)     item = t->elem;
	else return false;
	int32_t stride = svsl_const_leaf_count(types, item);
	for (int32_t i = 0; i < count; i++)
		if (!eval_init(prog, e->init_list.items[i], item, out + i * stride)) return false;
	return true;
}

const uint64_t *svsl_const_eval(svsl_arena_t *arena, svsl_program_t *prog,
                                const svsl_ast_expr_t *expr, svsl_type_id_t type) {
	if (!expr || type == SVSL_TYPE_NONE) return NULL;
	int32_t n = svsl_const_leaf_count(&prog->types, type);
	if (n == 0) return NULL;
	uint64_t *leaves = svsl_arena_alloc(arena, (size_t)n * sizeof(uint64_t));
	return eval_init(prog, expr, type, leaves) ? leaves : NULL;
}

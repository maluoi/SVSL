// See const_eval.h. Moved verbatim from emit_spirv.c when the WGSL backend
// gained static-const materialization; behavior is unchanged.

#include "const_eval.h"

#include "../front/ast.h"

bool svsl_const_eval_num(const svsl_program_t *prog, const svsl_ast_expr_t *v, double *out) {
	switch (v->kind) {
	case svsl_expr_float_lit: *out = v->float_lit.value; return true;
	case svsl_expr_int_lit:   *out = (double)(int64_t)v->int_lit.value; return true;
	case svsl_expr_bool_lit:  *out = v->bool_lit ? 1 : 0; return true;
	case svsl_expr_unary:
		if (v->unary.op != svsl_tok_minus && v->unary.op != svsl_tok_plus) return false;
		if (!svsl_const_eval_num(prog, v->unary.operand, out)) return false;
		if (v->unary.op == svsl_tok_minus) *out = -*out;
		return true;
	case svsl_expr_ident: {
		// global initializers aren't body-checked, so resolve by name
		for (int32_t i = 0; i < prog->const_globals.count; i++) {
			const svsl_global_t *g = &prog->const_globals.items[i];
			if (svsl_str_eq(g->name, v->ident) && g->var && g->var->init)
				return svsl_const_eval_num(prog, g->var->init, out);
		}
		return false;
	}
	case svsl_expr_binary: {
		double a, b;
		if (!svsl_const_eval_num(prog, v->binary.lhs, &a) ||
		    !svsl_const_eval_num(prog, v->binary.rhs, &b)) return false;
		switch (v->binary.op) {
		case svsl_tok_plus:  *out = a + b; return true;
		case svsl_tok_minus: *out = a - b; return true;
		case svsl_tok_star:  *out = a * b; return true;
		case svsl_tok_slash: if (b == 0) return false; *out = a / b; return true;
		default: return false;
		}
	}
	case svsl_expr_cast:
		return svsl_const_eval_num(prog, v->cast.operand, out);
	default:
		return false;
	}
}

bool svsl_const_eval_vec(const svsl_program_t *prog, const svsl_ast_expr_t *v,
                         int32_t n, double *out) {
	switch (v->kind) {
	case svsl_expr_init_list:
	case svsl_expr_ctor: {
		const svsl_ast_expr_t **items = v->kind == svsl_expr_init_list
			? (const svsl_ast_expr_t **)v->init_list.items
			: (const svsl_ast_expr_t **)v->ctor.args;
		int32_t count = v->kind == svsl_expr_init_list ? v->init_list.count : v->ctor.arg_count;
		if (count == 1) {
			double s;
			if (!svsl_const_eval_num(prog, items[0], &s)) return false;
			for (int32_t i = 0; i < n; i++) out[i] = s;
			return true;
		}
		if (count != n) return false;
		for (int32_t i = 0; i < n; i++)
			if (!svsl_const_eval_num(prog, items[i], &out[i])) return false;
		return true;
	}
	case svsl_expr_binary: {
		double a[4], b[4];
		if (!svsl_const_eval_vec(prog, v->binary.lhs, n, a) ||
		    !svsl_const_eval_vec(prog, v->binary.rhs, n, b)) return false;
		for (int32_t i = 0; i < n; i++) {
			switch (v->binary.op) {
			case svsl_tok_plus:  out[i] = a[i] + b[i]; break;
			case svsl_tok_minus: out[i] = a[i] - b[i]; break;
			case svsl_tok_star:  out[i] = a[i] * b[i]; break;
			case svsl_tok_slash: if (b[i] == 0) return false; out[i] = a[i] / b[i]; break;
			default: return false;
			}
		}
		return true;
	}
	case svsl_expr_ident: // another const global of vector type
		for (int32_t i = 0; i < prog->const_globals.count; i++) {
			const svsl_global_t *g = &prog->const_globals.items[i];
			if (svsl_str_eq(g->name, v->ident) && g->var && g->var->init)
				return svsl_const_eval_vec(prog, g->var->init, n, out);
		}
		return false;
	default: { // scalar expression splats across the vector
		double s;
		if (!svsl_const_eval_num(prog, v, &s)) return false;
		for (int32_t i = 0; i < n; i++) out[i] = s;
		return true;
	}
	}
}

// See ir_verify.h.

#include "ir_verify.h"
#include "ir_cf.h"
#include "ir_operands.h"

// entities emit declares outside the body, so any use of them is dominated
static bool function_scope(svsl_ir_op_ op) {
	return op == svsl_ir_const || op == svsl_ir_spec_const || op == svsl_ir_undef || op == svsl_ir_var;
}

static bool fail(svsl_ir_verify_error_t *out, uint32_t inst, const char *what) {
	*out = (svsl_ir_verify_error_t){ .inst = inst, .what = what };
	return false;
}

// One operand x of instruction i, checked against the definition scopes.
static bool check_operand(const svsl_ir_func_t *fn, const svsl_ir_scope_t *scope, const int32_t *depth,
                          const uint32_t *serial, uint32_t i, uint32_t x, svsl_ir_verify_error_t *out) {
	if (x >= i) return fail(out, i, "operand is not an earlier instruction");
	const svsl_ir_inst_t *d = &fn->insts.items[x];
	if (d->op == svsl_ir_nop) return fail(out, i, "operand is a killed instruction");
	if (d->type == SVSL_TYPE_NONE || svsl_ir_ends_run((svsl_ir_op_)d->op))
		return fail(out, i, "operand produces no value");
	if (!function_scope((svsl_ir_op_)d->op) && !svsl_ir_scope_live(scope, depth[x], serial[x]))
		return fail(out, i, "operand escapes the arm that computes it");
	return true;
}

bool svsl_ir_verify(svsl_arena_t *scratch, const svsl_ir_func_t *fn, svsl_ir_verify_error_t *out_error) {
	uint32_t  n      = (uint32_t)fn->insts.count;
	int32_t  *depth  = svsl_arena_alloc(scratch, (size_t)(n > 0 ? n : 1) * sizeof(int32_t));
	uint32_t *serial = svsl_arena_alloc(scratch, (size_t)(n > 0 ? n : 1) * sizeof(uint32_t));
	uint8_t  *kind   = svsl_arena_alloc(scratch, (size_t)n + 1); // open construct per depth: the opening op
	svsl_ir_scope_t scope;
	svsl_ir_scope_begin(&scope, scratch, fn);

	for (uint32_t i = 0; i < n; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		svsl_ir_op_           op = (svsl_ir_op_)in->op;
		if (op == svsl_ir_nop) continue;
		if (in->aux_count && (uint64_t)in->aux + in->aux_count > (uint64_t)fn->aux.count)
			return fail(out_error, i, "aux range outside the aux pool");

		uint32_t mask = svsl_ir_value_arg_mask(in);
		for (int32_t a = 0; a < 4; a++)
			if ((mask & (1u << a)) && !check_operand(fn, &scope, depth, serial, i, in->args[a], out_error))
				return false;
		if (svsl_ir_aux_holds_values(in))
			for (uint32_t k = 0; k < in->aux_count; k++)
				if (!check_operand(fn, &scope, depth, serial, i, fn->aux.items[in->aux + k], out_error))
					return false;

		// marker nesting, against the construct open at this depth
		uint8_t open = scope.depth > 0 ? kind[scope.depth] : svsl_ir_nop;
		switch (op) {
		case svsl_ir_else:
			if (open != svsl_ir_if) return fail(out_error, i, "else outside an if");
			break;
		case svsl_ir_end_if:
			if (open != svsl_ir_if) return fail(out_error, i, "end_if outside an if");
			break;
		case svsl_ir_loop_continue: case svsl_ir_end_loop:
			if (open != svsl_ir_loop) return fail(out_error, i, "loop marker outside a loop");
			break;
		case svsl_ir_case: case svsl_ir_end_switch:
			if (open != svsl_ir_switch) return fail(out_error, i, "switch marker outside a switch");
			break;
		case svsl_ir_break: case svsl_ir_continue: {
			bool found = false;
			for (int32_t d = scope.depth; d > 0 && !found; d--)
				found = kind[d] == svsl_ir_loop || (op == svsl_ir_break && kind[d] == svsl_ir_switch);
			if (!found) return fail(out_error, i, op == svsl_ir_break ? "break outside a loop or switch"
			                                                           : "continue outside a loop");
			break;
		}
		default:
			break;
		}

		svsl_ir_scope_step(&scope, op);
		if (op == svsl_ir_if || op == svsl_ir_loop || op == svsl_ir_switch) kind[scope.depth] = (uint8_t)op;
		depth[i]  = scope.depth;
		serial[i] = scope.serial[scope.depth];
	}
	if (scope.depth != 0) return fail(out_error, n ? n - 1 : 0, "unclosed construct at the end");
	return true;
}

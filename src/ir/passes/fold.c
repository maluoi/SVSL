// Constant folding (LLVM's ConstantFolding/InstSimplify on constants): an op
// whose operands are all constant becomes a constant, lane by lane, through
// the exact evaluators in ir_const.c (which documents what is and isn't
// folded). A scalar result rewrites the instruction in place; a vector result
// becomes a construct of head constants, which emit writes as one
// OpConstantComposite. A `select` on a constant condition picks its side even
// when the sides are not constant, and a load from a constant table through
// constant indices reads the table's initializer. See docs/PLAN_optimizer_llvm.md
// items 1 and 6.

#include "passes.h"
#include "../ir_const.h"
#include "../ir_operands.h"

// The lanes of operand `id`, broadcast to `n` when it is a scalar.
static bool operand_lanes(const svsl_ir_edit_t *ed, uint32_t id, int32_t n, svsl_ir_lanes_t *out) {
	if (!svsl_ir_const_lanes(ed, id, out)) return false;
	if (out->count == 1)
		for (int32_t k = 1; k < n; k++) out->bits[k] = out->bits[0];
	else if (out->count != n) return false;
	out->count = n;
	return true;
}

static bool value_lanes(const svsl_ir_edit_t *ed, const svsl_types_t *types, uint32_t id,
                        svsl_scalar_ *out_s, int32_t *out_n) {
	return svsl_ir_lane_type(types, svsl_ir_edit_inst(ed, id)->type, out_s, out_n);
}

// A load from a constant global through constant indices reads the table's
// initializer (svsl_ir_const_global_lanes); false when it isn't one.
static bool const_table_load(const svsl_ir_edit_t *ed, const svsl_program_t *prog, const svsl_ir_inst_t *in,
                             svsl_ir_lanes_t *out) {
	uint32_t path[16];
	int32_t  n = 0;
	uint32_t p = svsl_ir_edit_resolve(ed, in->args[0]);
	// collect chain indices innermost-last, then reverse into root-first order
	while (p != SVSL_IR_NONE && svsl_ir_edit_inst(ed, p)->op == svsl_ir_chain) {
		const svsl_ir_inst_t *ch = svsl_ir_edit_inst(ed, p);
		for (int32_t k = (int32_t)ch->aux_count - 1; k >= 0; k--) {
			svsl_ir_lanes_t idx;
			if (n >= 16 || !svsl_ir_const_lanes(ed, ed->fn->aux.items[ch->aux + (uint32_t)k], &idx) ||
			    idx.count != 1) return false;
			path[n++] = (uint32_t)idx.bits[0];
		}
		p = svsl_ir_edit_resolve(ed, ch->args[0]);
	}
	if (p == SVSL_IR_NONE) return false;
	const svsl_ir_inst_t *root = svsl_ir_edit_inst(ed, p);
	if (root->op != svsl_ir_ptr || (svsl_ref_)root->args[0] != svsl_ref_const_global) return false;
	for (int32_t a = 0, b = n - 1; a < b; a++, b--) { uint32_t t = path[a]; path[a] = path[b]; path[b] = t; }
	return svsl_ir_const_global_lanes(prog, root->args[1], path, n, in->type, out);
}

// Every fold (and the constant-condition select) needs a constant first operand.
static bool first_operand_constant(const svsl_ir_edit_t *ed, const svsl_ir_inst_t *in) {
	uint32_t a = (svsl_ir_value_arg_mask(in) & 1) ? in->args[0] : SVSL_IR_NONE;
	if (a == SVSL_IR_NONE || (a = svsl_ir_edit_resolve(ed, a)) == SVSL_IR_NONE) return false;
	svsl_ir_op_ op = (svsl_ir_op_)svsl_ir_edit_inst(ed, a)->op;
	return op == svsl_ir_const || op == svsl_ir_construct;
}

// Folds `in` (result scalar rs, n lanes) into `out`; false when any lane can't.
static bool fold_inst(const svsl_ir_edit_t *ed, const svsl_types_t *types, const svsl_ir_inst_t *in,
                      svsl_scalar_ rs, int32_t n, svsl_ir_lanes_t *out) {
	svsl_ir_op_     op = (svsl_ir_op_)in->op;
	svsl_ir_lanes_t a, b, c;
	svsl_scalar_    as, bs;
	int32_t         an, bn;
	out->count = n;

	if (!(svsl_ir_value_arg_mask(in) & 1) || !value_lanes(ed, types, in->args[0], &as, &an)) return false;
	switch (op) {
	case svsl_ir_neg: case svsl_ir_bit_not: case svsl_ir_log_not:
		if (!operand_lanes(ed, in->args[0], n, &a)) return false;
		for (int32_t l = 0; l < n; l++)
			if (!svsl_ir_eval_unary(op, as, a.bits[l], &out->bits[l])) return false;
		return true;
	case svsl_ir_convert:
		if (!operand_lanes(ed, in->args[0], n, &a)) return false;
		for (int32_t l = 0; l < n; l++)
			if (!svsl_ir_eval_convert(as, rs, a.bits[l], &out->bits[l])) return false;
		return true;
	case svsl_ir_shl: case svsl_ir_shr:
		if (!value_lanes(ed, types, in->args[1], &bs, &bn)) return false;
		if (!operand_lanes(ed, in->args[0], n, &a) || !operand_lanes(ed, in->args[1], n, &b)) return false;
		for (int32_t l = 0; l < n; l++)
			if (!svsl_ir_eval_shift(op, as, bs, a.bits[l], b.bits[l], &out->bits[l])) return false;
		return true;
	case svsl_ir_select:
		if (!operand_lanes(ed, in->args[0], n, &c) || !operand_lanes(ed, in->args[1], n, &a) ||
		    !operand_lanes(ed, in->args[2], n, &b)) return false;
		for (int32_t l = 0; l < n; l++) out->bits[l] = (c.bits[l] & 1) ? a.bits[l] : b.bits[l];
		return true;
	default:
		break;
	}
	// binary: add sub mul div rem, bit ops, compares, log and/or. Operands share
	// a kind; a scalar operand broadcasts, as in vector*scalar.
	if (svsl_ir_value_arg_mask(in) != 0x3 || op == svsl_ir_mat_mul || op == svsl_ir_extract_dynamic)
		return false;
	if (!value_lanes(ed, types, in->args[1], &bs, &bn) || bs != as) return false;
	if (!operand_lanes(ed, in->args[0], n, &a) || !operand_lanes(ed, in->args[1], n, &b)) return false;
	for (int32_t l = 0; l < n; l++)
		if (!svsl_ir_eval_binary(op, as, a.bits[l], b.bits[l], &out->bits[l])) return false;
	return true;
}

void svsl_ir_fold(svsl_ir_edit_t *ed, svsl_program_t *prog, svsl_opt_level_ level) {
	(void)level;
	svsl_ir_func_t *fn    = ed->fn;
	svsl_types_t   *types = &prog->types;
	int32_t         count = fn->insts.count;

	for (int32_t i = 0; i < count; i++) {
		svsl_ir_inst_t *in = &fn->insts.items[i];
		svsl_ir_op_     op = (svsl_ir_op_)in->op;
		if (op == svsl_ir_const || op == svsl_ir_construct || op == svsl_ir_nop) continue;
		if (op == svsl_ir_load) { // a constant table never changes: read its initializer
			svsl_ir_lanes_t r;
			if (const_table_load(ed, prog, in, &r))
				svsl_ir_edit_replace(ed, (uint32_t)i, svsl_ir_make_const(ed, types, (uint32_t)i, in->type, &r));
			continue;
		}
		if (!svsl_ir_is_pure(in, types) || op == svsl_ir_ptr || op == svsl_ir_chain) continue;
		if (!first_operand_constant(ed, in)) continue; // cheap reject before any type work

		// a constant condition picks its side, constant or not
		if (op == svsl_ir_select) {
			svsl_ir_lanes_t c;
			if (svsl_ir_const_lanes(ed, in->args[0], &c)) {
				bool all = true, none = true;
				for (int32_t l = 0; l < c.count; l++) { all &= (c.bits[l] & 1) != 0; none &= !(c.bits[l] & 1); }
				if (all || none) { svsl_ir_edit_replace(ed, (uint32_t)i, all ? in->args[1] : in->args[2]); continue; }
			}
		}

		svsl_scalar_    rs;
		int32_t         n;
		svsl_ir_lanes_t r;
		if (!svsl_ir_lane_type(types, in->type, &rs, &n)) continue;
		if (!fold_inst(ed, types, in, rs, n, &r)) continue;

		if (n == 1) {
			*in = (svsl_ir_inst_t){ .op = svsl_ir_const, .type = in->type,
			                        .args = { (uint32_t)r.bits[0], (uint32_t)(r.bits[0] >> 32), 0, SVSL_IR_NONE },
			                        .loc = in->loc, .name = in->name };
			svsl_ir_edit_touch(ed);
		} else {
			// a vector constant: a construct of head scalars (emit: OpConstantComposite)
			svsl_ir_edit_replace(ed, (uint32_t)i, svsl_ir_make_const(ed, types, (uint32_t)i, in->type, &r));
		}
	}
}

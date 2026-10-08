// See ir_cf.h.

#include "ir_cf.h"

void svsl_ir_cf_build(svsl_arena_t *scratch, const svsl_ir_func_t *fn, svsl_ir_cf_t *out) {
	int32_t n    = fn->insts.count;
	size_t  size = (size_t)(n > 0 ? n : 1) * sizeof(uint32_t);
	out->arm     = svsl_arena_alloc(scratch, size);
	out->arm_end = svsl_arena_alloc(scratch, size);
	out->end     = svsl_arena_alloc(scratch, size);
	out->head    = svsl_arena_alloc(scratch, size);

	// stack of open constructs; each entry also remembers its current arm opener
	uint32_t *construct = svsl_arena_alloc(scratch, size);
	uint32_t *arm       = svsl_arena_alloc(scratch, size);
	int32_t   depth     = 0;
	for (int32_t i = 0; i < n; i++) {
		svsl_ir_op_ op = (svsl_ir_op_)fn->insts.items[i].op;
		out->arm[i] = depth > 0 ? arm[depth - 1] : SVSL_IR_NONE;
		switch (op) {
		case svsl_ir_if: case svsl_ir_loop: case svsl_ir_switch:
			construct[depth] = (uint32_t)i;
			arm[depth++]     = (uint32_t)i;
			break;
		case svsl_ir_else: case svsl_ir_loop_continue: case svsl_ir_case:
			if (depth == 0) break;
			out->arm_end[arm[depth - 1]] = (uint32_t)i;
			out->head[i]                 = construct[depth - 1];
			arm[depth - 1]               = (uint32_t)i;
			out->arm[i]                  = construct[depth - 1]; // the marker itself sits at the construct's level
			break;
		case svsl_ir_end_if: case svsl_ir_end_loop: case svsl_ir_end_switch:
			if (depth == 0) break;
			depth--;
			out->arm_end[arm[depth]] = (uint32_t)i;
			out->end[construct[depth]] = (uint32_t)i;
			out->head[i]               = construct[depth];
			out->arm[i]                = depth > 0 ? arm[depth - 1] : SVSL_IR_NONE;
			break;
		default:
			break;
		}
	}
}

uint32_t svsl_ir_cf_break_target(const svsl_ir_cf_t *cf, const svsl_ir_func_t *fn, uint32_t i,
                                 bool continue_only) {
	for (uint32_t a = cf->arm[i]; a != SVSL_IR_NONE; a = cf->arm[a]) {
		uint32_t     c  = fn->insts.items[a].op == svsl_ir_if || fn->insts.items[a].op == svsl_ir_loop ||
		                  fn->insts.items[a].op == svsl_ir_switch ? a : cf->head[a];
		svsl_ir_op_  op = (svsl_ir_op_)fn->insts.items[c].op;
		if (op == svsl_ir_loop || (!continue_only && op == svsl_ir_switch)) return c;
		a = c; // continue outward from the construct itself
	}
	return SVSL_IR_NONE;
}

bool svsl_ir_cf_loop_repeats(const svsl_ir_cf_t *cf, const svsl_ir_func_t *fn, uint32_t l) {
	uint32_t k = cf->arm_end[l];
	while (k-- > l + 1 && fn->insts.items[k].op == svsl_ir_nop) {}
	svsl_ir_op_ op = (svsl_ir_op_)fn->insts.items[k].op;
	return !(k > l && (op == svsl_ir_break || op == svsl_ir_return || op == svsl_ir_discard));
}

bool svsl_ir_cf_in_repeating_loop(const svsl_ir_cf_t *cf, const svsl_ir_func_t *fn, uint32_t i) {
	for (uint32_t a = cf->arm[i]; a != SVSL_IR_NONE;) {
		svsl_ir_op_ op = (svsl_ir_op_)fn->insts.items[a].op;
		uint32_t    c  = op == svsl_ir_if || op == svsl_ir_loop || op == svsl_ir_switch ? a : cf->head[a];
		if (fn->insts.items[c].op == svsl_ir_loop && svsl_ir_cf_loop_repeats(cf, fn, c)) return true;
		a = cf->arm[c];
	}
	return false;
}

void svsl_ir_scope_begin(svsl_ir_scope_t *s, svsl_arena_t *scratch, const svsl_ir_func_t *fn) {
	int32_t n = fn->insts.count + 1; // nesting can't exceed the instruction count
	*s = (svsl_ir_scope_t){ .serial = svsl_arena_alloc(scratch, (size_t)n * sizeof(uint32_t)), .next = 1 };
	s->serial[0] = s->next++;
}

void svsl_ir_scope_step(svsl_ir_scope_t *s, svsl_ir_op_ op) {
	switch (op) {
	case svsl_ir_if: case svsl_ir_loop: case svsl_ir_switch:
		s->serial[++s->depth] = s->next++;
		break;
	case svsl_ir_else: case svsl_ir_loop_continue: case svsl_ir_case:
		s->serial[s->depth] = s->next++; // a sibling arm: the previous one doesn't dominate it
		break;
	case svsl_ir_end_if: case svsl_ir_end_loop: case svsl_ir_end_switch:
		if (s->depth > 0) s->depth--;
		break;
	default:
		break;
	}
}

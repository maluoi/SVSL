// IR text dump for --dump-ir and golden tests.

#include "ir.h"

#include "../tables/intrinsics.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef svsl_array_t(char) dump_buf_t;

typedef struct dump_t {
	svsl_arena_t         *arena;
	dump_buf_t            out;
	const svsl_program_t *prog;
	const svsl_ir_func_t *fn;
	int32_t               indent;
} dump_t;

static void put(dump_t *d, const char *fmt, ...) {
	char    tmp[256];
	va_list args;
	va_start(args, fmt);
	int32_t len = vsnprintf(tmp, sizeof(tmp), fmt, args);
	va_end(args);
	if (len > (int32_t)sizeof(tmp) - 1) len = (int32_t)sizeof(tmp) - 1;
	for (int32_t i = 0; i < len; i++)
		svsl_array_push(d->arena, &d->out, tmp[i]);
}

static const char *op_names[] = {
	"nop", "const", "spec_const", "undef",
	"var", "param", "ptr", "chain", "load", "store",
	"construct", "extract", "insert", "shuffle", "extract_dyn",
	"add", "sub", "mul", "div", "rem",
	"neg", "bit_not", "log_not",
	"bit_and", "bit_or", "bit_xor", "shl", "shr",
	"eq", "ne", "lt", "le", "gt", "ge",
	"log_and", "log_or", "select", "convert", "mat_mul",
	"intrinsic", "tex",
	"image_load", "image_store", "image_atomic", "atomic",
	"spirv_asm", "bitfield_extract", "bitfield_insert",
	"if", "else", "end_if",
	"loop", "loop_continue", "end_loop", "break", "continue",
	"switch", "case", "end_switch",
	"return", "discard", "demote",
};

static bool is_value_op(svsl_ir_op_ op) {
	switch (op) {
	case svsl_ir_nop: case svsl_ir_store: case svsl_ir_image_store:
	case svsl_ir_if: case svsl_ir_else: case svsl_ir_end_if:
	case svsl_ir_loop: case svsl_ir_loop_continue: case svsl_ir_end_loop:
	case svsl_ir_break: case svsl_ir_continue:
	case svsl_ir_switch: case svsl_ir_case: case svsl_ir_end_switch:
	case svsl_ir_return: case svsl_ir_discard: case svsl_ir_demote:
		return false;
	default:
		return true;
	}
}

static void dump_inst(dump_t *d, uint32_t id) {
	const svsl_ir_inst_t *inst = &d->fn->insts.items[id];
	svsl_ir_op_           op   = (svsl_ir_op_)inst->op;
	if (op == svsl_ir_nop) return;

	if (op == svsl_ir_else || op == svsl_ir_end_if || op == svsl_ir_end_loop ||
	    op == svsl_ir_loop_continue || op == svsl_ir_case || op == svsl_ir_end_switch)
		d->indent--;

	put(d, "  ");
	for (int32_t i = 0; i < d->indent; i++) put(d, "  ");

	if (is_value_op(op)) {
		put(d, "%%%u = %s", id, op_names[op]);
		if (inst->type != SVSL_TYPE_NONE)
			put(d, " %s", svsl_type_name(&d->prog->types, inst->type));
	} else {
		put(d, "%s", op_names[op]);
	}

	switch (op) {
	case svsl_ir_const: {
		uint64_t bits = (uint64_t)inst->args[0] | ((uint64_t)inst->args[1] << 32);
		const svsl_type_t *t = svsl_type_get(&d->prog->types, inst->type);
		if (t->scalar == svsl_scalar_float32 || t->scalar == svsl_scalar_half ||
		    t->scalar == svsl_scalar_float16) {
			float f;
			uint32_t u = (uint32_t)bits;
			memcpy(&f, &u, 4);
			put(d, " %g", f);
		} else if (t->scalar == svsl_scalar_float64) {
			double f;
			memcpy(&f, &bits, 8);
			put(d, " %g", f);
		} else {
			put(d, " %lld", (long long)(int64_t)(int32_t)(uint32_t)bits);
		}
		break;
	}
	case svsl_ir_spec_const:
		put(d, " #%u", inst->args[0]);
		break;
	case svsl_ir_param:
		put(d, " #%u", inst->args[0]);
		break;
	case svsl_ir_ptr: {
		static const char *kinds[] = { "?", "local", "param", "buffer", "resource",
		                               "spec", "const_global", "workgroup", "func",
		                               "intrinsic", "method", "builtin", "swizzle", "mat_elem" };
		put(d, " %s %u %u", kinds[inst->args[0] < 14 ? inst->args[0] : 0],
		    inst->args[1], inst->args[2]);
		break;
	}
	case svsl_ir_extract:
	case svsl_ir_insert:
		put(d, " %%%u [%u]", inst->args[0], inst->args[1]);
		if (op == svsl_ir_insert) put(d, " %%%u", inst->args[2]);
		break;
	case svsl_ir_extract_dynamic:
		put(d, " %%%u [%%%u]", inst->args[0], inst->args[1]); // index is a value
		break;
	case svsl_ir_shuffle: {
		put(d, " %%%u .", inst->args[0]);
		static const char comps[] = "xyzw";
		for (uint32_t i = 0; i < inst->args[2]; i++)
			put(d, "%c", comps[(inst->args[1] >> (i * 4)) & 0xF]);
		break;
	}
	case svsl_ir_intrinsic:
		put(d, " %s", svsl_intrinsic_get_name((int32_t)inst->args[0]));
		break;
	case svsl_ir_tex: {
		const svsl_resource_t *res = &d->prog->resources.items[inst->args[0]];
		put(d, " %.*s method=%u", res->name.len, res->name.ptr, inst->args[2] & 0xFF);
		if (inst->args[1] != SVSL_IR_NONE) {
			const svsl_resource_t *smp = &d->prog->resources.items[inst->args[1]];
			put(d, " sampler=%.*s", smp->name.len, smp->name.ptr);
		}
		break;
	}
	case svsl_ir_image_load: case svsl_ir_image_store: case svsl_ir_image_atomic: {
		const svsl_resource_t *res = &d->prog->resources.items[inst->args[0]];
		put(d, " %.*s", res->name.len, res->name.ptr);
		for (int32_t a = 1; a < 3; a++)
			if (inst->args[a] != SVSL_IR_NONE && (op != svsl_ir_image_load || a < 2))
				put(d, " %%%u", inst->args[a]);
		break;
	}
	case svsl_ir_atomic:
		put(d, " op=%u %%%u %%%u", inst->args[3] & 0xFF, inst->args[0], inst->args[1]);
		if (inst->args[3] >> 8) put(d, " order=%u", inst->args[3] >> 8);
		break;
	case svsl_ir_spirv_asm: // aux carries the $value operand ids
		for (uint32_t k = 0; k < inst->aux_count; k++)
			put(d, " %%%u", d->fn->aux.items[inst->aux + k]);
		break;
	case svsl_ir_return:
		if (inst->args[0] != SVSL_IR_NONE) put(d, " %%%u", inst->args[0]);
		break;
	case svsl_ir_case:
		put(d, " %u%s", inst->args[0], inst->args[1] ? " default" : "");
		break;
	case svsl_ir_var:
	case svsl_ir_construct:
	case svsl_ir_undef:
	case svsl_ir_else: case svsl_ir_end_if:
	case svsl_ir_loop: case svsl_ir_loop_continue: case svsl_ir_end_loop:
	case svsl_ir_break: case svsl_ir_continue: case svsl_ir_end_switch:
	case svsl_ir_discard: case svsl_ir_demote:
		break;
	default: // simple value args
		for (int32_t a = 0; a < 4; a++) {
			if (inst->args[a] == SVSL_IR_NONE) break;
			if ((op == svsl_ir_load || op == svsl_ir_chain || op == svsl_ir_neg ||
			     op == svsl_ir_bit_not || op == svsl_ir_log_not || op == svsl_ir_convert ||
			     op == svsl_ir_if || op == svsl_ir_switch) && a > 0) break;
			if ((op == svsl_ir_store || op == svsl_ir_add || op == svsl_ir_sub ||
			     op == svsl_ir_mul || op == svsl_ir_div || op == svsl_ir_rem ||
			     op == svsl_ir_bit_and || op == svsl_ir_bit_or || op == svsl_ir_bit_xor ||
			     op == svsl_ir_shl || op == svsl_ir_shr || op == svsl_ir_eq ||
			     op == svsl_ir_ne || op == svsl_ir_lt || op == svsl_ir_le ||
			     op == svsl_ir_gt || op == svsl_ir_ge || op == svsl_ir_log_and ||
			     op == svsl_ir_log_or || op == svsl_ir_mat_mul) && a > 1) break;
			if (op == svsl_ir_select && a > 2) break;
			put(d, " %%%u", inst->args[a]);
		}
		break;
	}

	if (inst->aux_count > 0 &&
	    (op == svsl_ir_chain || op == svsl_ir_construct || op == svsl_ir_intrinsic || op == svsl_ir_tex)) {
		put(d, " (");
		for (uint32_t i = 0; i < inst->aux_count; i++)
			put(d, "%s%%%u", i ? " " : "", d->fn->aux.items[inst->aux + i]);
		put(d, ")");
	}
	if (inst->name.len) put(d, " ; %.*s", inst->name.len, inst->name.ptr);
	put(d, "\n");

	if (op == svsl_ir_if || op == svsl_ir_else || op == svsl_ir_loop ||
	    op == svsl_ir_loop_continue || op == svsl_ir_case || op == svsl_ir_switch)
		d->indent++;
}

const char *svsl_ir_dump(svsl_arena_t *arena, const svsl_ir_module_t *module,
                         const svsl_program_t *prog) {
	dump_t d = { .arena = arena, .prog = prog };

	for (int32_t f = 0; f < module->func_count; f++) {
		const svsl_ir_func_t *fn = &module->funcs[f];
		const char *stage = fn->entry->stage == svsl_stage_vertex ? "vertex" :
		                    fn->entry->stage == svsl_stage_pixel  ? "pixel" : "compute";
		put(&d, "func %.*s %s\n", fn->entry->name.len, fn->entry->name.ptr, stage);
		d.fn     = fn;
		d.indent = 0;
		for (int32_t i = 0; i < fn->insts.count; i++)
			dump_inst(&d, (uint32_t)i);
	}
	svsl_array_push(arena, &d.out, '\0');
	return d.out.items;
}

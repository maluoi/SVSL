// AST dump: compact s-expression text for golden tests and debugging.

#include "ast.h"

#include "../util/array.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef svsl_array_t(char) dump_buf_t;

typedef struct dump_t {
	svsl_arena_t *arena;
	dump_buf_t    out;
} dump_t;

static void put(dump_t *d, const char *fmt, ...) {
	char    tmp[512];
	va_list args;
	va_start(args, fmt);
	int32_t len = vsnprintf(tmp, sizeof(tmp), fmt, args);
	va_end(args);
	if (len > (int32_t)sizeof(tmp) - 1) len = (int32_t)sizeof(tmp) - 1;
	for (int32_t i = 0; i < len; i++)
		svsl_array_push(d->arena, &d->out, tmp[i]);
}

static void put_str(dump_t *d, svsl_str_t s) { put(d, "%.*s", s.len, s.ptr); }

static const char *op_name(svsl_tok_ op) {
	switch (op) {
	case svsl_tok_plus:           return "+";
	case svsl_tok_minus:          return "-";
	case svsl_tok_star:           return "*";
	case svsl_tok_slash:          return "/";
	case svsl_tok_percent:        return "%";
	case svsl_tok_assign:         return "=";
	case svsl_tok_plus_assign:    return "+=";
	case svsl_tok_minus_assign:   return "-=";
	case svsl_tok_star_assign:    return "*=";
	case svsl_tok_slash_assign:   return "/=";
	case svsl_tok_percent_assign: return "%=";
	case svsl_tok_and_assign:     return "&=";
	case svsl_tok_or_assign:      return "|=";
	case svsl_tok_xor_assign:     return "^=";
	case svsl_tok_shl_assign:     return "<<=";
	case svsl_tok_shr_assign:     return ">>=";
	case svsl_tok_eq:             return "==";
	case svsl_tok_neq:            return "!=";
	case svsl_tok_lt:             return "<";
	case svsl_tok_gt:             return ">";
	case svsl_tok_le:             return "<=";
	case svsl_tok_ge:             return ">=";
	case svsl_tok_andand:         return "&&";
	case svsl_tok_oror:           return "||";
	case svsl_tok_not:            return "!";
	case svsl_tok_amp:            return "&";
	case svsl_tok_pipe:           return "|";
	case svsl_tok_caret:          return "^";
	case svsl_tok_tilde:          return "~";
	case svsl_tok_shl:            return "<<";
	case svsl_tok_shr:            return ">>";
	case svsl_tok_plusplus:       return "++";
	case svsl_tok_minusminus:     return "--";
	case svsl_tok_comma:          return ",";
	default:                      return "?";
	}
}

static void dump_expr(dump_t *d, const svsl_ast_expr_t *e);

static void dump_type(dump_t *d, const svsl_ast_type_t *t) {
	if (t->inline_enum) {
		put(d, "enum ");
		put_str(d, t->inline_enum->name);
		put(d, "{");
		for (int32_t m = 0; m < t->inline_enum->item_count; m++) {
			if (m) put(d, " ");
			put_str(d, t->inline_enum->items[m].name);
		}
		put(d, "}");
		return;
	}
	put_str(d, t->name);
	if (t->elem) {
		put(d, "<");
		dump_type(d, t->elem);
		if (t->format.len) { put(d, ","); put_str(d, t->format); }
		if (t->index)      { put(d, ","); dump_expr(d, t->index); }
		put(d, ">");
	}
	for (int32_t i = 0; i < t->array_dim_count; i++) {
		put(d, "[");
		if (t->array_dims[i]) dump_expr(d, t->array_dims[i]);
		put(d, "]");
	}
}

static void dump_expr(dump_t *d, const svsl_ast_expr_t *e) {
	if (!e) { put(d, "()"); return; }
	switch (e->kind) {
	case svsl_expr_int_lit:    put(d, "%llu", (unsigned long long)e->int_lit.value); break;
	case svsl_expr_float_lit:  put(d, "%g", e->float_lit.value); break;
	case svsl_expr_bool_lit:   put(d, e->bool_lit ? "true" : "false"); break;
	case svsl_expr_string_lit: put(d, "\""); put_str(d, e->string_lit); put(d, "\""); break;
	case svsl_expr_ident:      put_str(d, e->ident); break;
	case svsl_expr_binary:
		put(d, "(%s ", op_name(e->binary.op));
		dump_expr(d, e->binary.lhs);
		put(d, " ");
		dump_expr(d, e->binary.rhs);
		put(d, ")");
		break;
	case svsl_expr_unary:
		put(d, "(%s ", op_name(e->unary.op));
		dump_expr(d, e->unary.operand);
		put(d, ")");
		break;
	case svsl_expr_post:
		put(d, "(post%s ", op_name(e->post.op));
		dump_expr(d, e->post.operand);
		put(d, ")");
		break;
	case svsl_expr_ternary:
		put(d, "(?: ");
		dump_expr(d, e->ternary.cond);
		put(d, " ");
		dump_expr(d, e->ternary.then_expr);
		put(d, " ");
		dump_expr(d, e->ternary.else_expr);
		put(d, ")");
		break;
	case svsl_expr_call:
		put(d, "(call ");
		dump_expr(d, e->call.callee);
		for (int32_t i = 0; i < e->call.arg_count; i++) {
			put(d, " ");
			dump_expr(d, e->call.args[i]);
		}
		put(d, ")");
		break;
	case svsl_expr_ctor:
		put(d, "(ctor ");
		dump_type(d, e->ctor.type);
		for (int32_t i = 0; i < e->ctor.arg_count; i++) {
			put(d, " ");
			dump_expr(d, e->ctor.args[i]);
		}
		put(d, ")");
		break;
	case svsl_expr_cast:
		put(d, "(cast ");
		dump_type(d, e->cast.type);
		put(d, " ");
		dump_expr(d, e->cast.operand);
		put(d, ")");
		break;
	case svsl_expr_member:
		put(d, "(. ");
		dump_expr(d, e->member.object);
		put(d, " ");
		put_str(d, e->member.name);
		put(d, ")");
		break;
	case svsl_expr_index:
		put(d, "([] ");
		dump_expr(d, e->index.object);
		put(d, " ");
		dump_expr(d, e->index.index);
		put(d, ")");
		break;
	case svsl_expr_init_list:
		put(d, "(init");
		for (int32_t i = 0; i < e->init_list.count; i++) {
			put(d, " ");
			dump_expr(d, e->init_list.items[i]);
		}
		put(d, ")");
		break;
	case svsl_expr_spirv_asm:
		put(d, "(spirv_asm");
		for (int32_t i = 0; i < e->spirv_asm.inst_count; i++) {
			put(d, " (");
			put_str(d, e->spirv_asm.insts[i].opcode);
			for (int32_t o = 0; o < e->spirv_asm.insts[i].operand_count; o++) {
				const svsl_ast_spv_operand_t *op = &e->spirv_asm.insts[i].operands[o];
				put(d, " ");
				switch ((svsl_spv_operand_)op->kind) {
				case svsl_spv_operand_local:   put(d, "%%"); put_str(d, op->local);       break;
				case svsl_spv_operand_value:   put(d, "$"); dump_expr(d, op->value);       break;
				case svsl_spv_operand_type:    put(d, "$$"); put_str(d, op->type->name);   break;
				case svsl_spv_operand_literal: put(d, "%u", op->literal);                  break;
				case svsl_spv_operand_glsl450: put(d, "glsl450");                          break;
				}
			}
			put(d, ")");
		}
		put(d, ")");
		break;
	}
}

static void dump_attrs(dump_t *d, svsl_ast_attrs_t attrs) {
	for (int32_t i = 0; i < attrs.count; i++) {
		put(d, "[");
		put_str(d, attrs.items[i].name);
		for (int32_t a = 0; a < attrs.items[i].arg_count; a++) {
			put(d, " ");
			dump_expr(d, attrs.items[i].args[a]);
		}
		put(d, "]");
	}
}

static void dump_var(dump_t *d, const svsl_ast_var_t *v, const char *tag) {
	put(d, "(%s ", tag);
	put_str(d, v->name);
	put(d, " ");
	dump_attrs(d, v->attrs);
	if (v->flags & svsl_var_flag_specialization) put(d, "spec ");
	if (v->flags & svsl_var_flag_static)         put(d, "static ");
	if (v->flags & svsl_var_flag_const)          put(d, "const ");
	if (v->flags & svsl_var_flag_workgroup)      put(d, "workgroup ");
	if (v->flags & svsl_var_flag_readonly)       put(d, "readonly ");
	if (v->flags & svsl_var_flag_writeonly)      put(d, "writeonly ");
	if (v->flags & svsl_var_flag_coherent)       put(d, "coherent ");
	if (v->flags & svsl_var_flag_volatile)       put(d, "volatile ");
	if (v->flags & svsl_var_flag_precise)        put(d, "precise ");
	if (v->flags & svsl_var_flag_uniform)        put(d, "uniform ");
	if (v->pack == svsl_pack_1)                  put(d, "pack1 ");
	if (v->pack == svsl_pack_8)                  put(d, "pack8 ");
	if (v->pack == svsl_pack_16)                 put(d, "pack16 ");
	if (v->pack == svsl_pack_430)                put(d, "std430 ");
	if (v->interp & svsl_interp_flat)            put(d, "flat ");
	if (v->interp & svsl_interp_noperspective)   put(d, "noperspective ");
	if (v->interp & svsl_interp_centroid)        put(d, "centroid ");
	if (v->interp & svsl_interp_sample)          put(d, "sample ");
	if (v->interp & svsl_interp_invariant)       put(d, "invariant ");
	if (v->dir == svsl_dir_out)                  put(d, "out ");
	if (v->dir == svsl_dir_inout)                put(d, "inout ");
	dump_type(d, v->type);
	if (v->semantic.len) { put(d, " :"); put_str(d, v->semantic); }
	if (v->reg.present) {
		if (v->reg.direct) put(d, " :register(%d,%d)", v->reg.slot, v->reg.space);
		else               put(d, " :register(%c%d,space%d)", v->reg.cls, v->reg.slot, v->reg.space);
	}
	if (v->init) {
		put(d, " = ");
		dump_expr(d, v->init);
	}
	put(d, ")");
}

static void dump_stmt(dump_t *d, const svsl_ast_stmt_t *s) {
	if (!s) { put(d, "()"); return; }
	dump_attrs(d, s->attrs);
	switch (s->kind) {
	case svsl_stmt_block:
		put(d, "(block");
		for (int32_t i = 0; i < s->block.count; i++) {
			put(d, " ");
			dump_stmt(d, s->block.stmts[i]);
		}
		put(d, ")");
		break;
	case svsl_stmt_expr:
		put(d, "(expr ");
		dump_expr(d, s->expr);
		put(d, ")");
		break;
	case svsl_stmt_var_decl:
		put(d, "(decl");
		for (int32_t i = 0; i < s->var_decl.count; i++) {
			put(d, " ");
			dump_var(d, s->var_decl.vars[i], "var");
		}
		put(d, ")");
		break;
	case svsl_stmt_if:
		put(d, "(if ");
		dump_expr(d, s->if_stmt.cond);
		put(d, " ");
		dump_stmt(d, s->if_stmt.then_stmt);
		if (s->if_stmt.else_stmt) {
			put(d, " ");
			dump_stmt(d, s->if_stmt.else_stmt);
		}
		put(d, ")");
		break;
	case svsl_stmt_for:
		put(d, "(for ");
		dump_stmt(d, s->for_stmt.init);
		put(d, " ");
		dump_expr(d, s->for_stmt.cond);
		put(d, " ");
		dump_expr(d, s->for_stmt.inc);
		put(d, " ");
		dump_stmt(d, s->for_stmt.body);
		put(d, ")");
		break;
	case svsl_stmt_while:
		put(d, "(while ");
		dump_expr(d, s->while_stmt.cond);
		put(d, " ");
		dump_stmt(d, s->while_stmt.body);
		put(d, ")");
		break;
	case svsl_stmt_do:
		put(d, "(do ");
		dump_stmt(d, s->while_stmt.body);
		put(d, " ");
		dump_expr(d, s->while_stmt.cond);
		put(d, ")");
		break;
	case svsl_stmt_switch:
		put(d, "(switch ");
		dump_expr(d, s->switch_stmt.value);
		for (int32_t i = 0; i < s->switch_stmt.case_count; i++) {
			const svsl_ast_case_t *c = &s->switch_stmt.cases[i];
			if (c->value) { put(d, " (case "); dump_expr(d, c->value); }
			else          put(d, " (default");
			for (int32_t k = 0; k < c->stmt_count; k++) {
				put(d, " ");
				dump_stmt(d, c->stmts[k]);
			}
			put(d, ")");
		}
		put(d, ")");
		break;
	case svsl_stmt_break:    put(d, "(break)");    break;
	case svsl_stmt_continue: put(d, "(continue)"); break;
	case svsl_stmt_discard:  put(d, "(discard)");  break;
	case svsl_stmt_demote:   put(d, "(demote)");   break;
	case svsl_stmt_return:
		put(d, "(return");
		if (s->return_value) {
			put(d, " ");
			dump_expr(d, s->return_value);
		}
		put(d, ")");
		break;
	}
}

const char *svsl_ast_dump(svsl_arena_t *arena, const svsl_ast_t *ast) {
	dump_t d = { .arena = arena };

	for (int32_t i = 0; i < ast->decl_count; i++) {
		const svsl_ast_decl_t *decl = ast->decls[i];
		switch (decl->kind) {
		case svsl_decl_struct:
			put(&d, "(struct ");
			put_str(&d, decl->struct_decl.name);
			for (int32_t m = 0; m < decl->struct_decl.member_count; m++) {
				put(&d, " ");
				dump_var(&d, decl->struct_decl.members[m], "member");
			}
			put(&d, ")");
			break;
		case svsl_decl_block: {
			const char *kind =
				decl->block.kind == svsl_block_uniform       ? (decl->block.legacy ? "cbuffer" : "uniform") :
				decl->block.kind == svsl_block_storagebuffer ? "storagebuffer" : "pushconstant";
			put(&d, "(%s ", kind);
			put_str(&d, decl->block.name);
			dump_attrs(&d, decl->block.attrs);
			if (decl->block.pack == svsl_pack_1)   put(&d, " pack1");
			if (decl->block.pack == svsl_pack_8)   put(&d, " pack8");
			if (decl->block.pack == svsl_pack_16)  put(&d, " pack16");
			if (decl->block.pack == svsl_pack_430) put(&d, " std430");
			if (decl->block.flags & svsl_var_flag_readonly) put(&d, " readonly");
			if (decl->block.reg.present) {
				if (decl->block.reg.direct) put(&d, " :register(%d,%d)", decl->block.reg.slot, decl->block.reg.space);
				else put(&d, " :register(%c%d,space%d)", decl->block.reg.cls, decl->block.reg.slot, decl->block.reg.space);
			}
			for (int32_t m = 0; m < decl->block.member_count; m++) {
				put(&d, " ");
				dump_var(&d, decl->block.members[m], "member");
			}
			put(&d, ")");
			break;
		}
		case svsl_decl_enum:
			put(&d, "(enum ");
			put_str(&d, decl->enum_decl.name);
			if (decl->enum_decl.underlying) { put(&d, " : "); dump_type(&d, decl->enum_decl.underlying); }
			for (int32_t m = 0; m < decl->enum_decl.item_count; m++) {
				put(&d, " ");
				put_str(&d, decl->enum_decl.items[m].name);
				if (decl->enum_decl.items[m].value) { put(&d, "="); dump_expr(&d, decl->enum_decl.items[m].value); }
			}
			put(&d, ")");
			break;
		case svsl_decl_var:
			dump_var(&d, &decl->var, "global");
			break;
		case svsl_decl_func:
			put(&d, "(func ");
			put_str(&d, decl->func.name);
			put(&d, " ");
			dump_attrs(&d, decl->func.attrs);
			put(&d, "(ret ");
			dump_type(&d, decl->func.return_type);
			if (decl->func.return_semantic.len) {
				put(&d, " :");
				put_str(&d, decl->func.return_semantic);
			}
			put(&d, ")");
			for (int32_t a = 0; a < decl->func.param_count; a++) {
				put(&d, " ");
				dump_var(&d, decl->func.params[a], "param");
			}
			if (decl->func.body) {
				put(&d, " ");
				dump_stmt(&d, decl->func.body);
			}
			put(&d, ")");
			break;
		case svsl_decl_include:
			put(&d, "(include \"");
			put_str(&d, decl->include_path);
			put(&d, "\")");
			break;
		}
		put(&d, "\n");
	}

	svsl_array_push(arena, &d.out, '\0');
	return d.out.items;
}

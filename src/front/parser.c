#include "parser.h"

#include "../tables/keywords.h"
#include "../tables/types_builtin.h"
#include "../util/array.h"

#include <string.h>

typedef svsl_array_t(svsl_ast_expr_t *) expr_list_t;
typedef svsl_array_t(svsl_ast_stmt_t *) stmt_list_t;
typedef svsl_array_t(svsl_ast_var_t *)  var_list_t;
typedef svsl_array_t(svsl_ast_attr_t)   attr_list_t;
typedef svsl_array_t(svsl_ast_decl_t *) decl_list_t;
typedef svsl_array_t(svsl_ast_case_t)   case_list_t;
typedef svsl_array_t(svsl_str_t)        name_list_t;

typedef struct parse_t {
	svsl_arena_t       *arena;
	const svsl_token_t *toks;
	int32_t             count;
	int32_t             pos;
	svsl_diag_list_t   *diags;
	name_list_t         struct_names; // pass 1
	int32_t             depth;        // recursion guard: parens, blocks, unary chains
	bool                too_deep;     // reported once per file
} parse_t;

// recursion budget shared by expressions and statements; every later pass
// (sema, IR, folding) walks the tree recursively, so the parser is the one
// place deep enough input can be refused instead of overflowing the stack
#define PARSE_MAX_DEPTH 256
#define PARSE_MAX_CHAIN 512 // a op b op c ... builds tree height iteratively; the
                            // cap keeps sanitizer builds (large frames) safe too

// --- token helpers ---------------------------------------------------------------

static const svsl_token_t *cur (const parse_t *p)            { return &p->toks[p->pos]; }
static const svsl_token_t *peek(const parse_t *p, int32_t n) {
	int32_t i = p->pos + n;
	return &p->toks[i < p->count ? i : p->count - 1]; // last token is always eof
}
static void advance(parse_t *p) { if (p->pos < p->count - 1) p->pos++; }

static bool at(const parse_t *p, svsl_tok_ kind)  { return cur(p)->kind == kind; }
static bool at_kw(const parse_t *p, svsl_kw_ kw)  { return cur(p)->kind == svsl_tok_ident && cur(p)->keyword == (int16_t)kw; }

static bool accept(parse_t *p, svsl_tok_ kind) {
	if (!at(p, kind)) return false;
	advance(p);
	return true;
}
static bool accept_kw(parse_t *p, svsl_kw_ kw) {
	if (!at_kw(p, kw)) return false;
	advance(p);
	return true;
}

static void error_at(parse_t *p, svsl_loc_t loc, const char *fmt, svsl_str_t arg) {
	svsl_diag_add(p->arena, p->diags, svsl_severity_error, loc, fmt, arg.len, arg.ptr);
}

static bool expect(parse_t *p, svsl_tok_ kind, const char *what) {
	if (accept(p, kind)) return true;
	svsl_diag_add(p->arena, p->diags, svsl_severity_error, cur(p)->loc,
	              "expected %s, got '%.*s'", what, cur(p)->text.len ? cur(p)->text.len : 5,
	              cur(p)->text.len ? cur(p)->text.ptr : "<eof>");
	return false;
}

static svsl_str_t expect_ident(parse_t *p, const char *what) {
	if (at(p, svsl_tok_ident)) {
		svsl_str_t name = cur(p)->text;
		advance(p);
		return name;
	}
	svsl_diag_add(p->arena, p->diags, svsl_severity_error, cur(p)->loc, "expected %s", what);
	return (svsl_str_t){0};
}

static bool is_type_name(const parse_t *p, svsl_str_t name) {
	if (svsl_type_name_is_builtin(name)) return true;
	for (int32_t i = 0; i < p->struct_names.count; i++)
		if (svsl_str_eq(p->struct_names.items[i], name)) return true;
	return false;
}

// --- node allocation ----------------------------------------------------------------

static svsl_ast_expr_t *expr_new(parse_t *p, svsl_expr_ kind, svsl_loc_t loc) {
	svsl_ast_expr_t *e = svsl_arena_alloc(p->arena, sizeof(svsl_ast_expr_t));
	e->kind      = kind;
	e->loc       = loc;
	e->sema_type = -1; // SVSL_TYPE_NONE until sema runs
	return e;
}
static svsl_ast_stmt_t *stmt_new(parse_t *p, svsl_stmt_ kind, svsl_loc_t loc) {
	svsl_ast_stmt_t *s = svsl_arena_alloc(p->arena, sizeof(svsl_ast_stmt_t));
	s->kind = kind;
	s->loc  = loc;
	return s;
}

// --- forward declarations --------------------------------------------------------------

static svsl_ast_expr_t *parse_expr       (parse_t *p);            // includes comma
static svsl_ast_expr_t *parse_expr_assign(parse_t *p);            // no comma
static svsl_ast_expr_t *parse_unary      (parse_t *p);
static svsl_ast_stmt_t *parse_stmt       (parse_t *p);
static svsl_ast_stmt_t *parse_stmt_inner (parse_t *p);
static svsl_ast_type_t *parse_type       (parse_t *p);

// --- attributes ---------------------------------------------------------------------------

// [name], [name(args)], [[vk::name(args)]]
static svsl_ast_attrs_t parse_attrs(parse_t *p) {
	attr_list_t list = {0};

	while (at(p, svsl_tok_lbracket)) {
		svsl_loc_t loc = cur(p)->loc;
		advance(p);
		bool dbl = accept(p, svsl_tok_lbracket);

		svsl_str_t name = expect_ident(p, "attribute name");
		if (dbl && accept(p, svsl_tok_coloncolon)) {
			svsl_str_t part2 = expect_ident(p, "attribute name after '::'");
			char *joined = svsl_arena_alloc(p->arena, (size_t)name.len + 2 + (size_t)part2.len + 1);
			if (name.len)  memcpy(joined, name.ptr, (size_t)name.len); // idents are empty when expect_ident failed
			memcpy(joined + name.len, "::", 2);
			if (part2.len) memcpy(joined + name.len + 2, part2.ptr, (size_t)part2.len);
			name = (svsl_str_t){ .ptr = joined, .len = name.len + 2 + part2.len };
		}

		expr_list_t args = {0};
		if (accept(p, svsl_tok_lparen)) {
			if (!at(p, svsl_tok_rparen)) {
				do {
					svsl_array_push(p->arena, &args, parse_expr_assign(p));
				} while (accept(p, svsl_tok_comma));
			}
			expect(p, svsl_tok_rparen, "')' after attribute arguments");
		}
		expect(p, svsl_tok_rbracket, "']' after attribute");
		if (dbl) expect(p, svsl_tok_rbracket, "']]' after attribute");

		svsl_array_push(p->arena, &list, (svsl_ast_attr_t){
			.name           = name,
			.double_bracket = dbl,
			.args           = args.items,
			.arg_count      = args.count,
			.loc            = loc });
	}
	return (svsl_ast_attrs_t){ .items = list.items, .count = list.count };
}

// --- types -----------------------------------------------------------------------------------

static void parse_enum_body(parse_t *p, svsl_ast_enum_t *out); // fills name/underlying/items

// base type name with optional <element[, format-or-index]> template arguments,
// or an inline 'enum {...}' declaration used directly in a type position
static svsl_ast_type_t *parse_type(parse_t *p) {
	svsl_ast_type_t *type = svsl_arena_alloc(p->arena, sizeof(svsl_ast_type_t));
	type->loc  = cur(p)->loc;
	if (at_kw(p, svsl_kw_enum)) {
		svsl_ast_enum_t *en = svsl_arena_alloc(p->arena, sizeof(svsl_ast_enum_t));
		parse_enum_body(p, en);
		type->name        = en->name; // empty when anonymous
		type->inline_enum = en;
		return type;
	}
	type->name = expect_ident(p, "type name");

	if (at(p, svsl_tok_lt) && svsl_type_name_is_resource(type->name)) {
		advance(p);
		// unorm/snorm texel modifiers parse and drop: glslang ignores them even
		// for storage image format inference (RWTexture3D<unorm float4> is
		// Rgba32f, same as plain float4 — verified against skshaderc)
		if (at(p, svsl_tok_ident) &&
		    (svsl_str_eq_cstr(cur(p)->text, "unorm") || svsl_str_eq_cstr(cur(p)->text, "snorm")))
			advance(p);
		type->elem = parse_type(p);
		if (accept(p, svsl_tok_comma)) {
			// Image2D<float4, rgba8> format, or SubpassInput<float4, 1> index
			if (at(p, svsl_tok_ident)) {
				type->format = cur(p)->text;
				advance(p);
			} else {
				type->index = parse_unary(p); // constrained: '>' must close the template
			}
		}
		expect(p, svsl_tok_gt, "'>' after template argument");
	}
	return type;
}

// shallow per-declarator copy, so `float a[2], b;` gets distinct array dims
static svsl_ast_type_t *type_clone(parse_t *p, const svsl_ast_type_t *type) {
	svsl_ast_type_t *copy = svsl_arena_alloc(p->arena, sizeof(svsl_ast_type_t));
	*copy = *type;
	return copy;
}

// [4][2] or [] — appended to the declarator's own type copy
static void parse_array_dims(parse_t *p, svsl_ast_type_t *type) {
	expr_list_t dims = {0};
	while (accept(p, svsl_tok_lbracket)) {
		if (accept(p, svsl_tok_rbracket)) {
			svsl_array_push(p->arena, &dims, (svsl_ast_expr_t *)NULL); // runtime-sized
			continue;
		}
		svsl_array_push(p->arena, &dims, parse_expr_assign(p));
		expect(p, svsl_tok_rbracket, "']' after array size");
	}
	type->array_dims      = dims.items;
	type->array_dim_count = dims.count;
}

// --- register() and colon clauses ---------------------------------------------------------------

// register(t0), register(t0, space1), register(0, 1)
static svsl_ast_reg_t parse_register(parse_t *p) {
	svsl_ast_reg_t reg = { .present = true, .slot = -1, .loc = cur(p)->loc };
	expect(p, svsl_tok_lparen, "'(' after register");

	if (at(p, svsl_tok_int_lit)) { // direct (binding, set) form
		reg.direct = true;
		reg.slot   = (int32_t)cur(p)->int_value;
		advance(p);
		if (accept(p, svsl_tok_comma)) {
			if (at(p, svsl_tok_int_lit)) { reg.space = (int32_t)cur(p)->int_value; advance(p); }
			else expect(p, svsl_tok_int_lit, "descriptor set number");
		}
	} else if (at(p, svsl_tok_ident)) { // t0 form
		svsl_str_t text = cur(p)->text;
		char       cls  = text.ptr[0];
		if ((cls == 'b' || cls == 't' || cls == 's' || cls == 'u') && text.len > 1) {
			int32_t slot = 0;
			bool    ok   = true;
			for (int32_t i = 1; i < text.len; i++) {
				if (text.ptr[i] < '0' || text.ptr[i] > '9' || slot > 0xFFFFFF) { ok = false; break; }
				slot = slot * 10 + (text.ptr[i] - '0');
			}
			if (ok) { reg.cls = cls; reg.slot = slot; }
			else error_at(p, cur(p)->loc, "bad register '%.*s'", text);
		} else {
			error_at(p, cur(p)->loc, "bad register '%.*s'", text);
		}
		advance(p);
		if (accept(p, svsl_tok_comma)) {
			svsl_str_t space = expect_ident(p, "space after ','");
			if (svsl_str_starts_with(space, "space") && space.len > 5) {
				int32_t v = 0;
				for (int32_t i = 5; i < space.len && v <= 0xFFFFFF; i++) {
					if (space.ptr[i] < '0' || space.ptr[i] > '9') break;
					v = v * 10 + (space.ptr[i] - '0');
				}
				reg.space = v;
			} else if (space.len > 0) {
				error_at(p, reg.loc, "expected spaceN, got '%.*s'", space);
			}
		}
	} else {
		svsl_diag_add(p->arena, p->diags, svsl_severity_error, cur(p)->loc, "expected register slot");
	}
	expect(p, svsl_tok_rparen, "')' after register");
	return reg;
}

// : semantic and/or : register(...) in either order
static void parse_colon_clauses(parse_t *p, svsl_str_t *out_semantic, svsl_ast_reg_t *out_reg) {
	while (at(p, svsl_tok_colon)) {
		advance(p);
		if (at_kw(p, svsl_kw_register)) {
			advance(p);
			svsl_ast_reg_t reg = parse_register(p);
			if (out_reg) *out_reg = reg;
			else svsl_diag_add(p->arena, p->diags, svsl_severity_error, reg.loc, "register() not allowed here");
			continue;
		}
		svsl_str_t sem = expect_ident(p, "semantic name");
		if (out_semantic) *out_semantic = sem;
	}
}

// --- expressions ------------------------------------------------------------------------------------

static int32_t binary_prec(svsl_tok_ op) {
	switch (op) {
	case svsl_tok_comma:                                     return 1;
	case svsl_tok_assign:        case svsl_tok_plus_assign:
	case svsl_tok_minus_assign:  case svsl_tok_star_assign:
	case svsl_tok_slash_assign:  case svsl_tok_percent_assign:
	case svsl_tok_and_assign:    case svsl_tok_or_assign:
	case svsl_tok_xor_assign:    case svsl_tok_shl_assign:
	case svsl_tok_shr_assign:                                return 2;
	// ternary '?' handled explicitly at precedence 3
	case svsl_tok_oror:                                      return 4;
	case svsl_tok_andand:                                    return 5;
	case svsl_tok_pipe:                                      return 6;
	case svsl_tok_caret:                                     return 7;
	case svsl_tok_amp:                                       return 8;
	case svsl_tok_eq: case svsl_tok_neq:                     return 9;
	case svsl_tok_lt: case svsl_tok_gt:
	case svsl_tok_le: case svsl_tok_ge:                      return 10;
	case svsl_tok_shl: case svsl_tok_shr:                    return 11;
	case svsl_tok_plus: case svsl_tok_minus:                 return 12;
	case svsl_tok_star: case svsl_tok_slash:
	case svsl_tok_percent:                                   return 13;
	default:                                                 return 0;
	}
}

static bool is_assign_op(svsl_tok_ op) { return binary_prec(op) == 2; }

static svsl_ast_expr_t *parse_unary(parse_t *p);

static svsl_ast_expr_t *parse_args_into(parse_t *p, expr_list_t *out) {
	if (!at(p, svsl_tok_rparen)) {
		do {
			svsl_array_push(p->arena, out, parse_expr_assign(p));
		} while (accept(p, svsl_tok_comma));
	}
	expect(p, svsl_tok_rparen, "')' after arguments");
	return NULL;
}

// spirv_asm(T) { OpXxx <operand>* ; ... } — inline SPIR-V. Operands are written
// in binary order: %id (a named local, %result carries the block's value), $expr
// (an SVSL value), $$type (an SVSL type), an integer literal, or glsl450.
static svsl_ast_expr_t *parse_spirv_asm(parse_t *p) {
	svsl_loc_t loc = cur(p)->loc;
	advance(p); // 'spirv' / 'spirv_asm'

	svsl_ast_expr_t *e = expr_new(p, svsl_expr_spirv_asm, loc);
	e->spirv_asm.result_type = NULL;
	e->spirv_asm.insts       = NULL;
	e->spirv_asm.inst_count  = 0;

	expect(p, svsl_tok_lparen, "'(' and a result type after spirv_asm");
	e->spirv_asm.result_type = parse_type(p);
	expect(p, svsl_tok_rparen, "')'");
	expect(p, svsl_tok_lbrace, "'{'");

	svsl_array_t(svsl_ast_spv_inst_t) insts = {0};
	while (!at(p, svsl_tok_rbrace) && !at(p, svsl_tok_eof)) {
		svsl_ast_spv_inst_t inst = { .loc = cur(p)->loc, .spv_op = -1 };
		inst.opcode = expect_ident(p, "a SPIR-V opcode mnemonic (e.g. OpFAdd)");

		svsl_array_t(svsl_ast_spv_operand_t) ops = {0};
		while (!at(p, svsl_tok_semicolon) && !at(p, svsl_tok_rbrace) && !at(p, svsl_tok_eof)) {
			svsl_ast_spv_operand_t op = { .loc = cur(p)->loc, .type_id = -1 };
			if (accept(p, svsl_tok_percent)) {
				op.kind  = svsl_spv_operand_local;
				op.local = expect_ident(p, "an id name after '%'");
			} else if (at(p, svsl_tok_dollar) && peek(p, 1)->kind == svsl_tok_dollar) {
				advance(p); advance(p); // '$$'
				op.kind = svsl_spv_operand_type;
				op.type = parse_type(p);
			} else if (accept(p, svsl_tok_dollar)) {
				op.kind = svsl_spv_operand_value;
				if (accept(p, svsl_tok_lparen)) {
					op.value = parse_expr_assign(p);
					expect(p, svsl_tok_rparen, "')'");
				} else {
					svsl_ast_expr_t *id = expr_new(p, svsl_expr_ident, cur(p)->loc);
					id->ident = expect_ident(p, "a value name after '$'");
					op.value  = id;
				}
			} else if (at(p, svsl_tok_int_lit)) {
				op.kind    = svsl_spv_operand_literal;
				op.literal = (uint32_t)cur(p)->int_value;
				advance(p);
			} else if (at(p, svsl_tok_ident) && svsl_str_eq_cstr(cur(p)->text, "glsl450")) {
				op.kind = svsl_spv_operand_glsl450;
				advance(p);
			} else {
				svsl_diag_add(p->arena, p->diags, svsl_severity_error, cur(p)->loc,
				              "expected a spirv_asm operand: %%id, $value, $$type, an integer, "
				              "or glsl450 (got '%.*s')", cur(p)->text.len ? cur(p)->text.len : 5,
				              cur(p)->text.len ? cur(p)->text.ptr : "<eof>");
				advance(p); // make progress
			}
			svsl_array_push(p->arena, &ops, op);
		}
		inst.operands      = ops.items;
		inst.operand_count = ops.count;
		expect(p, svsl_tok_semicolon, "';' after a spirv_asm instruction");
		svsl_array_push(p->arena, &insts, inst);
	}
	expect(p, svsl_tok_rbrace, "'}'");
	e->spirv_asm.insts      = insts.items;
	e->spirv_asm.inst_count = insts.count;
	return e;
}

static svsl_ast_expr_t *parse_primary(parse_t *p) {
	svsl_loc_t loc = cur(p)->loc;

	if (at_kw(p, svsl_kw_spirv)) return parse_spirv_asm(p);
	if (at(p, svsl_tok_int_lit)) {
		svsl_ast_expr_t *e = expr_new(p, svsl_expr_int_lit, loc);
		e->int_lit.value  = cur(p)->int_value;
		e->int_lit.suffix = cur(p)->suffix;
		advance(p);
		return e;
	}
	if (at(p, svsl_tok_float_lit)) {
		svsl_ast_expr_t *e = expr_new(p, svsl_expr_float_lit, loc);
		e->float_lit.value  = cur(p)->float_value;
		e->float_lit.suffix = cur(p)->suffix;
		advance(p);
		return e;
	}
	if (at(p, svsl_tok_string_lit)) {
		svsl_ast_expr_t *e = expr_new(p, svsl_expr_string_lit, loc);
		e->string_lit = cur(p)->text;
		advance(p);
		return e;
	}
	if (at_kw(p, svsl_kw_true) || at_kw(p, svsl_kw_false)) {
		svsl_ast_expr_t *e = expr_new(p, svsl_expr_bool_lit, loc);
		e->bool_lit = cur(p)->keyword == svsl_kw_true;
		advance(p);
		return e;
	}
	if (at(p, svsl_tok_ident)) {
		svsl_str_t name = cur(p)->text;
		// constructor: type name used as a function — float4(...), float4x4(1)
		if (peek(p, 1)->kind == svsl_tok_lparen && is_type_name(p, name)) {
			svsl_ast_type_t *type = parse_type(p);
			expect(p, svsl_tok_lparen, "'('");
			expr_list_t args = {0};
			parse_args_into(p, &args);
			svsl_ast_expr_t *e = expr_new(p, svsl_expr_ctor, loc);
			e->ctor.type      = type;
			e->ctor.args      = args.items;
			e->ctor.arg_count = args.count;
			return e;
		}
		svsl_ast_expr_t *e = expr_new(p, svsl_expr_ident, loc);
		e->ident = name;
		advance(p);
		return e;
	}
	if (accept(p, svsl_tok_lparen)) {
		svsl_ast_expr_t *e = parse_expr(p);
		expect(p, svsl_tok_rparen, "')'");
		return e;
	}

	svsl_diag_add(p->arena, p->diags, svsl_severity_error, loc,
	              "expected expression, got '%.*s'", cur(p)->text.len ? cur(p)->text.len : 5,
	              cur(p)->text.len ? cur(p)->text.ptr : "<eof>");
	advance(p); // make progress
	svsl_ast_expr_t *e = expr_new(p, svsl_expr_int_lit, loc);
	return e;
}

static svsl_ast_expr_t *parse_postfix(parse_t *p) {
	svsl_ast_expr_t *e = parse_primary(p);

	for (;;) {
		svsl_loc_t loc = cur(p)->loc;
		if (accept(p, svsl_tok_dot)) {
			svsl_ast_expr_t *m = expr_new(p, svsl_expr_member, loc);
			m->member.object = e;
			m->member.name   = expect_ident(p, "member name after '.'");
			e = m;
			continue;
		}
		if (accept(p, svsl_tok_lbracket)) {
			svsl_ast_expr_t *idx = expr_new(p, svsl_expr_index, loc);
			idx->index.object = e;
			idx->index.index  = parse_expr(p);
			expect(p, svsl_tok_rbracket, "']'");
			e = idx;
			continue;
		}
		if (at(p, svsl_tok_lparen) && (e->kind == svsl_expr_ident || e->kind == svsl_expr_member)) {
			advance(p);
			expr_list_t args = {0};
			parse_args_into(p, &args);
			svsl_ast_expr_t *call = expr_new(p, svsl_expr_call, loc);
			call->call.callee    = e;
			call->call.args      = args.items;
			call->call.arg_count = args.count;
			e = call;
			continue;
		}
		if (at(p, svsl_tok_plusplus) || at(p, svsl_tok_minusminus)) {
			svsl_ast_expr_t *post = expr_new(p, svsl_expr_post, loc);
			post->post.op      = cur(p)->kind;
			post->post.operand = e;
			advance(p);
			e = post;
			continue;
		}
		return e;
	}
}

static svsl_ast_expr_t *parse_unary(parse_t *p);

static svsl_ast_expr_t *parse_unary_inner(parse_t *p) {
	svsl_loc_t loc = cur(p)->loc;

	// cast: '(' type-name ')' unary
	if (at(p, svsl_tok_lparen) && peek(p, 1)->kind == svsl_tok_ident &&
	    peek(p, 2)->kind == svsl_tok_rparen && is_type_name(p, peek(p, 1)->text)) {
		advance(p);
		svsl_ast_type_t *type = parse_type(p);
		expect(p, svsl_tok_rparen, "')' after cast type");
		svsl_ast_expr_t *e = expr_new(p, svsl_expr_cast, loc);
		e->cast.type    = type;
		e->cast.operand = parse_unary(p);
		return e;
	}

	svsl_tok_ k = cur(p)->kind;
	if (k == svsl_tok_minus || k == svsl_tok_plus || k == svsl_tok_not || k == svsl_tok_tilde ||
	    k == svsl_tok_plusplus || k == svsl_tok_minusminus) {
		advance(p);
		svsl_ast_expr_t *e = expr_new(p, svsl_expr_unary, loc);
		e->unary.op      = k;
		e->unary.operand = parse_unary(p);
		return e;
	}
	return parse_postfix(p);
}

static svsl_ast_expr_t *parse_unary(parse_t *p) {
	if (p->depth >= PARSE_MAX_DEPTH) {
		if (!p->too_deep) {
			p->too_deep = true;
			error_at(p, cur(p)->loc, "expression nesting too deep%.*s", (svsl_str_t){0});
		}
		advance(p); // make progress, stop recursing
		return expr_new(p, svsl_expr_int_lit, cur(p)->loc);
	}
	p->depth++;
	svsl_ast_expr_t *e = parse_unary_inner(p);
	p->depth--;
	return e;
}

static svsl_ast_expr_t *parse_binary(parse_t *p, int32_t min_prec);

// A binary/ternary right-hand side recurses through parse_binary; right-associative
// forms (assignment, ternary arms) nest without bound, which parse_unary's guard
// doesn't cover — so track and cap depth across the RHS descent here too.
static svsl_ast_expr_t *parse_binary_rhs(parse_t *p, int32_t min_prec) {
	if (p->depth >= PARSE_MAX_DEPTH) {
		if (!p->too_deep) {
			p->too_deep = true;
			error_at(p, cur(p)->loc, "expression nesting too deep%.*s", (svsl_str_t){0});
		}
		advance(p); // make progress, stop recursing
		return expr_new(p, svsl_expr_int_lit, cur(p)->loc);
	}
	p->depth++;
	svsl_ast_expr_t *e = parse_binary(p, min_prec);
	p->depth--;
	return e;
}

static svsl_ast_expr_t *parse_binary(parse_t *p, int32_t min_prec) {
	svsl_ast_expr_t *lhs = parse_unary(p);

	int32_t chain = 0;
	for (;;) {
		if (++chain > PARSE_MAX_CHAIN) { // left-leaning trees grow height per iteration
			if (!p->too_deep) {
				p->too_deep = true;
				error_at(p, cur(p)->loc, "expression too long%.*s", (svsl_str_t){0});
			}
			return lhs;
		}
		// ternary sits between assignment (2) and || (4)
		if (at(p, svsl_tok_question) && min_prec <= 3) {
			svsl_loc_t loc = cur(p)->loc;
			advance(p);
			svsl_ast_expr_t *e = expr_new(p, svsl_expr_ternary, loc);
			e->ternary.cond      = lhs;
			e->ternary.then_expr = parse_binary_rhs(p, 2);
			expect(p, svsl_tok_colon, "':' in ternary");
			e->ternary.else_expr = parse_binary_rhs(p, 2); // right-associative
			lhs = e;
			continue;
		}

		svsl_tok_ op   = cur(p)->kind;
		int32_t   prec = binary_prec(op);
		if (prec == 0 || prec < min_prec) return lhs;

		svsl_loc_t loc = cur(p)->loc;
		advance(p);
		// assignment is right-associative; everything else left
		svsl_ast_expr_t *rhs = parse_binary_rhs(p, is_assign_op(op) ? prec : prec + 1);
		svsl_ast_expr_t *e   = expr_new(p, svsl_expr_binary, loc);
		e->binary.op  = op;
		e->binary.lhs = lhs;
		e->binary.rhs = rhs;
		lhs = e;
	}
}

static svsl_ast_expr_t *parse_expr       (parse_t *p) { return parse_binary(p, 1); }
static svsl_ast_expr_t *parse_expr_assign(parse_t *p) { return parse_binary(p, 2); }

// initializer: assignment expression or {init, init, ...}
static svsl_ast_expr_t *parse_initializer(parse_t *p) {
	if (at(p, svsl_tok_lbrace)) {
		svsl_loc_t loc = cur(p)->loc;
		advance(p);
		expr_list_t items = {0};
		if (!at(p, svsl_tok_rbrace)) {
			do {
				if (at(p, svsl_tok_rbrace)) break; // trailing comma
				svsl_array_push(p->arena, &items, parse_initializer(p));
			} while (accept(p, svsl_tok_comma));
		}
		expect(p, svsl_tok_rbrace, "'}' after initializer list");
		svsl_ast_expr_t *e = expr_new(p, svsl_expr_init_list, loc);
		e->init_list.items = items.items;
		e->init_list.count = items.count;
		return e;
	}
	return parse_expr_assign(p);
}

// --- variable declarations ------------------------------------------------------------------------

typedef struct decl_mods_t {
	uint32_t flags;
	uint8_t  interp;
	uint8_t  dir;
	uint8_t  pack;   // svsl_pack_
	bool     any;
} decl_mods_t;

// Layout keywords by their standards names — context-sensitive identifiers, not
// reserved words ('relaxed' is also an atomic order argument, 'scalar' a likely
// variable name). Only recognized when what follows can continue a declaration:
// a type name, a block keyword, or another modifier.
static bool accept_pack_alias(parse_t *p, uint8_t *out_pack) {
	if (!at(p, svsl_tok_ident)) return false;
	uint8_t pack;
	if      (svsl_str_eq_cstr(cur(p)->text, "scalar"))  pack = svsl_pack_1;
	else if (svsl_str_eq_cstr(cur(p)->text, "relaxed")) pack = svsl_pack_8;
	else if (svsl_str_eq_cstr(cur(p)->text, "std140"))  pack = svsl_pack_16;
	else if (svsl_str_eq_cstr(cur(p)->text, "std430"))  pack = svsl_pack_430;
	else return false;

	const svsl_token_t *nx = peek(p, 1);
	bool decl_follows = nx->kind == svsl_tok_ident &&
		(is_type_name(p, nx->text) ||
		 nx->keyword == svsl_kw_storagebuffer || nx->keyword == svsl_kw_pushconstant ||
		 nx->keyword == svsl_kw_uniform       || nx->keyword == svsl_kw_cbuffer      ||
		 nx->keyword == svsl_kw_readonly      || nx->keyword == svsl_kw_writeonly    ||
		 nx->keyword == svsl_kw_coherent      || nx->keyword == svsl_kw_volatile);
	if (!decl_follows) return false;
	*out_pack = pack;
	advance(p);
	return true;
}

// static/const/workgroup/etc. and interpolation modifiers, in any order
static decl_mods_t parse_decl_mods(parse_t *p, bool allow_dir) {
	decl_mods_t mods = {0};
	for (;;) {
		if      (accept_kw(p, svsl_kw_static))          mods.flags |= svsl_var_flag_static;
		else if (accept_kw(p, svsl_kw_const))           mods.flags |= svsl_var_flag_const;
		else if (accept_kw(p, svsl_kw_workgroup))       mods.flags |= svsl_var_flag_workgroup;
		else if (accept_kw(p, svsl_kw_groupshared))     mods.flags |= svsl_var_flag_workgroup | svsl_var_flag_legacy_spelling;
		else if (accept_kw(p, svsl_kw_specialization))  mods.flags |= svsl_var_flag_specialization;
		else if (accept_kw(p, svsl_kw_readonly))        mods.flags |= svsl_var_flag_readonly;
		else if (accept_kw(p, svsl_kw_writeonly))       mods.flags |= svsl_var_flag_writeonly;
		else if (accept_kw(p, svsl_kw_coherent))        mods.flags |= svsl_var_flag_coherent;
		else if (accept_kw(p, svsl_kw_volatile))        mods.flags |= svsl_var_flag_volatile;
		else if (accept_kw(p, svsl_kw_flat))            mods.interp |= svsl_interp_flat;
		else if (accept_kw(p, svsl_kw_nointerpolation)) mods.interp |= svsl_interp_flat; // legacy
		else if (accept_kw(p, svsl_kw_noperspective))   mods.interp |= svsl_interp_noperspective;
		else if (accept_kw(p, svsl_kw_centroid))        mods.interp |= svsl_interp_centroid;
		else if (accept_kw(p, svsl_kw_invariant))       mods.interp |= svsl_interp_invariant;
		else if (accept_kw(p, svsl_kw_precise))         mods.flags  |= svsl_var_flag_precise;
		else if (accept_kw(p, svsl_kw_pack1))           mods.pack = svsl_pack_1;
		else if (accept_kw(p, svsl_kw_pack8))           mods.pack = svsl_pack_8;
		else if (accept_kw(p, svsl_kw_pack16))          mods.pack = svsl_pack_16;
		else if (accept_pack_alias(p, &mods.pack))      {} // std430/std140/scalar/relaxed
		// 'sample' and 'linear' are context-sensitive: only modifiers when followed by a type
		else if ((at_kw(p, svsl_kw_sample) || (at(p, svsl_tok_ident) && svsl_str_eq_cstr(cur(p)->text, "linear"))) &&
		         peek(p, 1)->kind == svsl_tok_ident && is_type_name(p, peek(p, 1)->text)) {
			if (at_kw(p, svsl_kw_sample)) mods.interp |= svsl_interp_sample;
			else                          mods.interp |= svsl_interp_linear;
			advance(p);
		}
		else if (allow_dir && accept_kw(p, svsl_kw_in))    mods.dir = svsl_dir_in;
		else if (allow_dir && accept_kw(p, svsl_kw_out))   mods.dir = svsl_dir_out;
		else if (allow_dir && accept_kw(p, svsl_kw_inout)) mods.dir = svsl_dir_inout;
		else if (at(p, svsl_tok_ident) && svsl_str_eq_cstr(cur(p)->text, "inline") &&
		         peek(p, 1)->kind == svsl_tok_ident) {
			advance(p); // HLSL 'inline' — accepted, meaningless (all calls are inlined anyway)
			continue;   // doesn't count as a modifier
		}
		else break;
		mods.any = true;
	}
	return mods;
}

// layout keywords name a whole buffer's memory layout — they have no meaning on
// locals, parameters, or individual members
static void reject_pack_mods(parse_t *p, decl_mods_t *mods, svsl_loc_t loc) {
	if (!mods->pack) return;
	svsl_diag_add(p->arena, p->diags, svsl_severity_error, loc,
	              "layout keywords only apply to buffer declarations");
	mods->pack = 0;
}

// one declarator: name [dims] [: semantic] [: register] [= init]
// packed-struct bit field: ': N' (raw), ': unN' (unorm), ': snN' (snorm). Only
// consumes the ':' when it is genuinely a bit field — a plain ': SEMANTIC' or
// ': register(...)' is left for parse_colon_clauses.
static void parse_bitfield_spec(parse_t *p, svsl_ast_var_t *var) {
	if (!at(p, svsl_tok_colon)) return;
	const svsl_token_t *nx = peek(p, 1);
	if (nx->kind == svsl_tok_int_lit) {                 // ': N' — raw bits
		advance(p); // ':'
		var->bit_width  = (int16_t)cur(p)->int_value;
		var->bit_format = svsl_bitfmt_raw;
		advance(p); // int
		return;
	}
	if (nx->kind == svsl_tok_ident && nx->text.len >= 3) { // ': unN' / ': snN'
		svsl_str_t t   = nx->text;
		uint8_t    fmt = (t.ptr[0] == 'u' && t.ptr[1] == 'n') ? svsl_bitfmt_unorm :
		                 (t.ptr[0] == 's' && t.ptr[1] == 'n') ? svsl_bitfmt_snorm : 255;
		if (fmt == 255) return;
		int32_t w = 0;
		for (int32_t i = 2; i < t.len; i++) {
			if (t.ptr[i] < '0' || t.ptr[i] > '9') return; // not un/sn<digits> → a semantic
			w = w * 10 + (t.ptr[i] - '0');
		}
		advance(p); // ':'
		advance(p); // ident
		var->bit_width  = (int16_t)w;
		var->bit_format = fmt;
	}
}

static svsl_ast_var_t *parse_declarator(parse_t *p, const svsl_ast_type_t *base_type,
                                        const decl_mods_t *mods, svsl_ast_attrs_t attrs) {
	svsl_ast_var_t *var = svsl_arena_alloc(p->arena, sizeof(svsl_ast_var_t));
	var->loc    = cur(p)->loc;
	var->name   = expect_ident(p, "variable name");
	var->type   = type_clone(p, base_type);
	var->flags  = mods->flags;
	var->interp = mods->interp;
	var->dir    = mods->dir;
	var->pack   = mods->pack;
	var->attrs  = attrs;

	for (int32_t a = 0; a < attrs.count; a++)
		if (svsl_str_eq_cstr(attrs.items[a].name, "specialization") ||
		    svsl_str_eq_cstr(attrs.items[a].name, "vk::constant_id"))
			var->flags |= svsl_var_flag_specialization;

	parse_array_dims(p, var->type);
	var->bit_width  = -1;
	var->bit_format = svsl_bitfmt_raw;
	parse_bitfield_spec(p, var);
	parse_colon_clauses(p, &var->semantic, &var->reg);
	if (accept(p, svsl_tok_assign))
		var->init = parse_initializer(p);
	return var;
}

// type declarator [, declarator]* ';' — shared by locals, globals, and block members
static void parse_var_decl_list(parse_t *p, const decl_mods_t *mods, svsl_ast_attrs_t attrs,
                                var_list_t *out) {
	svsl_ast_type_t *base = parse_type(p);
	do {
		svsl_array_push(p->arena, out, parse_declarator(p, base, mods, attrs));
	} while (accept(p, svsl_tok_comma));
	expect(p, svsl_tok_semicolon, "';' after declaration");
}

// --- statements --------------------------------------------------------------------------------------

// does a variable declaration start here?
static bool stmt_starts_decl(const parse_t *p) {
	if (at_kw(p, svsl_kw_static) || at_kw(p, svsl_kw_const) ||
	    at_kw(p, svsl_kw_workgroup) || at_kw(p, svsl_kw_groupshared)) return true;
	if (at_kw(p, svsl_kw_enum)) return true; // 'enum {...} local;' — inline in a type position
	if (!at(p, svsl_tok_ident)) return false;
	if (peek(p, 1)->kind == svsl_tok_ident) {
		// 'type name' — including keyword-flavored names like 'float sample'
		if (is_type_name(p, cur(p)->text)) return true;
		// 'ident ident' is never a valid expression, so it's a declaration even when
		// the type name is unknown (sema reports the unknown type with a better message)
		if (peek(p, 1)->keyword == (int16_t)svsl_kw_none) return true;
	}
	// 'type <' — templated resource local (illegal, sema rejects with context)
	return is_type_name(p, cur(p)->text) && peek(p, 1)->kind == svsl_tok_lt;
}

static void stmt_sync(parse_t *p) {
	int32_t depth = 0;
	while (!at(p, svsl_tok_eof)) {
		svsl_tok_ k = cur(p)->kind;
		if (depth == 0 && k == svsl_tok_semicolon) { advance(p); return; }
		if (k == svsl_tok_lbrace) depth++;
		if (k == svsl_tok_rbrace) {
			if (depth == 0) return; // let the caller close its block
			depth--;
		}
		advance(p);
	}
}

static svsl_ast_stmt_t *parse_block(parse_t *p) {
	svsl_ast_stmt_t *block = stmt_new(p, svsl_stmt_block, cur(p)->loc);
	expect(p, svsl_tok_lbrace, "'{'");

	stmt_list_t stmts = {0};
	while (!at(p, svsl_tok_rbrace) && !at(p, svsl_tok_eof)) {
		int32_t before = p->pos;
		svsl_ast_stmt_t *s = parse_stmt(p);
		if (s) svsl_array_push(p->arena, &stmts, s);
		if (p->pos == before) { // no progress: recover
			stmt_sync(p);
			if (p->pos == before) advance(p);
		}
	}
	expect(p, svsl_tok_rbrace, "'}'");
	block->block.stmts = stmts.items;
	block->block.count = stmts.count;
	return block;
}

static svsl_ast_stmt_t *parse_switch(parse_t *p) {
	svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_switch, cur(p)->loc);
	advance(p); // switch
	expect(p, svsl_tok_lparen, "'(' after switch");
	s->switch_stmt.value = parse_expr(p);
	expect(p, svsl_tok_rparen, "')'");
	expect(p, svsl_tok_lbrace, "'{'");

	case_list_t cases = {0};
	while (!at(p, svsl_tok_rbrace) && !at(p, svsl_tok_eof)) {
		svsl_ast_case_t c = { .loc = cur(p)->loc };
		if (accept_kw(p, svsl_kw_case)) {
			c.value = parse_expr(p);
		} else if (!accept_kw(p, svsl_kw_default)) {
			svsl_diag_add(p->arena, p->diags, svsl_severity_error, cur(p)->loc, "expected 'case' or 'default'");
			stmt_sync(p);
			continue;
		}
		expect(p, svsl_tok_colon, "':' after case label");

		stmt_list_t stmts = {0};
		while (!at(p, svsl_tok_rbrace) && !at_kw(p, svsl_kw_case) && !at_kw(p, svsl_kw_default) &&
		       !at(p, svsl_tok_eof)) {
			int32_t before = p->pos;
			svsl_ast_stmt_t *stmt = parse_stmt(p);
			if (stmt) svsl_array_push(p->arena, &stmts, stmt);
			if (p->pos == before) { stmt_sync(p); if (p->pos == before) advance(p); }
		}
		c.stmts      = stmts.items;
		c.stmt_count = stmts.count;
		svsl_array_push(p->arena, &cases, c);
	}
	expect(p, svsl_tok_rbrace, "'}' after switch");
	s->switch_stmt.cases      = cases.items;
	s->switch_stmt.case_count = cases.count;
	return s;
}

static svsl_ast_stmt_t *parse_stmt(parse_t *p) {
	if (p->depth >= PARSE_MAX_DEPTH) {
		if (!p->too_deep) {
			p->too_deep = true;
			error_at(p, cur(p)->loc, "statement nesting too deep%.*s", (svsl_str_t){0});
		}
		advance(p); // make progress, stop recursing
		return stmt_new(p, svsl_stmt_block, cur(p)->loc);
	}
	p->depth++;
	svsl_ast_stmt_t *s = parse_stmt_inner(p);
	p->depth--;
	return s;
}

static svsl_ast_stmt_t *parse_stmt_inner(parse_t *p) {
	svsl_ast_attrs_t attrs = parse_attrs(p); // [unroll], [branch], ...
	svsl_loc_t       loc   = cur(p)->loc;

	if (at(p, svsl_tok_lbrace)) {
		svsl_ast_stmt_t *s = parse_block(p);
		s->attrs = attrs;
		return s;
	}
	if (accept(p, svsl_tok_semicolon)) { // empty statement
		svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_block, loc);
		s->attrs = attrs;
		return s;
	}
	if (at_kw(p, svsl_kw_if)) {
		advance(p);
		svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_if, loc);
		s->attrs = attrs;
		expect(p, svsl_tok_lparen, "'(' after if");
		s->if_stmt.cond = parse_expr(p);
		expect(p, svsl_tok_rparen, "')'");
		s->if_stmt.then_stmt = parse_stmt(p);
		if (accept_kw(p, svsl_kw_else))
			s->if_stmt.else_stmt = parse_stmt(p);
		return s;
	}
	if (at_kw(p, svsl_kw_for)) {
		advance(p);
		svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_for, loc);
		s->attrs = attrs;
		expect(p, svsl_tok_lparen, "'(' after for");
		if (!accept(p, svsl_tok_semicolon)) {
			if (stmt_starts_decl(p)) {
				svsl_ast_stmt_t *init = stmt_new(p, svsl_stmt_var_decl, cur(p)->loc);
				decl_mods_t      mods = parse_decl_mods(p, false);
				reject_pack_mods(p, &mods, cur(p)->loc);
				var_list_t       vars = {0};
				parse_var_decl_list(p, &mods, (svsl_ast_attrs_t){0}, &vars);
				init->var_decl.vars  = vars.items;
				init->var_decl.count = vars.count;
				s->for_stmt.init = init;
			} else {
				svsl_ast_stmt_t *init = stmt_new(p, svsl_stmt_expr, cur(p)->loc);
				init->expr = parse_expr(p);
				expect(p, svsl_tok_semicolon, "';' after for initializer");
				s->for_stmt.init = init;
			}
		}
		if (!at(p, svsl_tok_semicolon))
			s->for_stmt.cond = parse_expr(p);
		expect(p, svsl_tok_semicolon, "';' after for condition");
		if (!at(p, svsl_tok_rparen))
			s->for_stmt.inc = parse_expr(p);
		expect(p, svsl_tok_rparen, "')' after for");
		s->for_stmt.body = parse_stmt(p);
		return s;
	}
	if (at_kw(p, svsl_kw_while)) {
		advance(p);
		svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_while, loc);
		s->attrs = attrs;
		expect(p, svsl_tok_lparen, "'(' after while");
		s->while_stmt.cond = parse_expr(p);
		expect(p, svsl_tok_rparen, "')'");
		s->while_stmt.body = parse_stmt(p);
		return s;
	}
	if (at_kw(p, svsl_kw_do)) {
		advance(p);
		svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_do, loc);
		s->attrs = attrs;
		s->while_stmt.body = parse_stmt(p);
		if (!accept_kw(p, svsl_kw_while))
			svsl_diag_add(p->arena, p->diags, svsl_severity_error, cur(p)->loc, "expected 'while' after do body");
		expect(p, svsl_tok_lparen, "'('");
		s->while_stmt.cond = parse_expr(p);
		expect(p, svsl_tok_rparen, "')'");
		expect(p, svsl_tok_semicolon, "';' after do-while");
		return s;
	}
	if (at_kw(p, svsl_kw_switch))   { svsl_ast_stmt_t *s = parse_switch(p); s->attrs = attrs; return s; }
	if (at_kw(p, svsl_kw_break))    { advance(p); expect(p, svsl_tok_semicolon, "';'"); svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_break, loc);    s->attrs = attrs; return s; }
	if (at_kw(p, svsl_kw_continue)) { advance(p); expect(p, svsl_tok_semicolon, "';'"); svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_continue, loc); s->attrs = attrs; return s; }
	if (at_kw(p, svsl_kw_discard))  { advance(p); expect(p, svsl_tok_semicolon, "';'"); svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_discard, loc);  s->attrs = attrs; return s; }
	if (at_kw(p, svsl_kw_demote))   { advance(p); expect(p, svsl_tok_semicolon, "';'"); svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_demote, loc);   s->attrs = attrs; return s; }
	if (at_kw(p, svsl_kw_return)) {
		advance(p);
		svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_return, loc);
		s->attrs = attrs;
		if (!at(p, svsl_tok_semicolon))
			s->return_value = parse_expr(p);
		expect(p, svsl_tok_semicolon, "';' after return");
		return s;
	}

	if (stmt_starts_decl(p)) {
		svsl_ast_stmt_t *s    = stmt_new(p, svsl_stmt_var_decl, loc);
		decl_mods_t      mods = parse_decl_mods(p, false);
		reject_pack_mods(p, &mods, loc);
		var_list_t       vars = {0};
		parse_var_decl_list(p, &mods, attrs, &vars);
		s->var_decl.vars  = vars.items;
		s->var_decl.count = vars.count;
		return s;
	}

	svsl_ast_stmt_t *s = stmt_new(p, svsl_stmt_expr, loc);
	s->attrs = attrs;
	s->expr  = parse_expr(p);
	expect(p, svsl_tok_semicolon, "';' after expression");
	return s;
}

// --- top-level declarations -----------------------------------------------------------------------------

static svsl_ast_decl_t *decl_new(parse_t *p, svsl_decl_ kind, svsl_loc_t loc) {
	svsl_ast_decl_t *d = svsl_arena_alloc(p->arena, sizeof(svsl_ast_decl_t));
	d->kind = kind;
	d->loc  = loc;
	return d;
}

// struct members / buffer block members
static void parse_member_list(parse_t *p, var_list_t *out) {
	expect(p, svsl_tok_lbrace, "'{'");
	while (!at(p, svsl_tok_rbrace) && !at(p, svsl_tok_eof)) {
		int32_t          before = p->pos;
		svsl_ast_attrs_t attrs  = parse_attrs(p); // [offset(N)], [location(N)]
		decl_mods_t      mods   = parse_decl_mods(p, false);
		reject_pack_mods(p, &mods, cur(p)->loc);
		parse_var_decl_list(p, &mods, attrs, out);
		if (p->pos == before) { stmt_sync(p); if (p->pos == before) advance(p); }
	}
	expect(p, svsl_tok_rbrace, "'}'");
}

static svsl_ast_decl_t *parse_struct(parse_t *p) {
	svsl_ast_decl_t *d = decl_new(p, svsl_decl_struct, cur(p)->loc);
	advance(p); // struct
	d->struct_decl.name = expect_ident(p, "struct name");
	if (at(p, svsl_tok_colon)) { // HLSL struct inheritance: not supported in SVSL
		svsl_diag_add(p->arena, p->diags, svsl_severity_error, cur(p)->loc,
		              "struct inheritance is not supported");
		advance(p);                       // ':'
		expect_ident(p, "base name");     // consume the base so members still parse
	}

	var_list_t members = {0};
	parse_member_list(p, &members);
	expect(p, svsl_tok_semicolon, "';' after struct");

	d->struct_decl.members      = members.items;
	d->struct_decl.member_count = members.count;
	d->struct_decl.loc          = d->loc;
	return d;
}

// enum [Name] [: underlying] { A [= expr], ... } — stops after '}'. Constant names
// are global (C / HLSL-unscoped flavor); the enum names an integer type alias.
static void parse_enum_body(parse_t *p, svsl_ast_enum_t *out) {
	out->loc = cur(p)->loc;
	advance(p); // enum
	if (at(p, svsl_tok_ident))                          // optional name (else anonymous)
		out->name = expect_ident(p, "enum name");
	if (accept(p, svsl_tok_colon))                      // optional underlying integer type
		out->underlying = parse_type(p);

	expect(p, svsl_tok_lbrace, "'{' for enum body");
	svsl_array_t(svsl_ast_enum_item_t) items = {0};
	while (!at(p, svsl_tok_rbrace) && !at(p, svsl_tok_eof)) {
		svsl_ast_enum_item_t it = { .loc = cur(p)->loc };
		it.name = expect_ident(p, "enum constant name");
		if (accept(p, svsl_tok_assign)) it.value = parse_expr_assign(p);
		svsl_array_push(p->arena, &items, it);
		if (!accept(p, svsl_tok_comma)) break;          // trailing comma optional
	}
	expect(p, svsl_tok_rbrace, "'}' after enum body");
	out->items      = items.items;
	out->item_count = items.count;
}

static svsl_ast_decl_t *parse_enum(parse_t *p) {
	svsl_ast_decl_t *d = decl_new(p, svsl_decl_enum, cur(p)->loc);
	parse_enum_body(p, &d->enum_decl);
	d->enum_decl.loc = d->loc;
	return d;
}

static svsl_ast_type_t *make_type_named(parse_t *p, svsl_str_t name, svsl_loc_t loc) {
	svsl_ast_type_t *t = svsl_arena_alloc(p->arena, sizeof(svsl_ast_type_t));
	*t = (svsl_ast_type_t){ .name = name, .loc = loc };
	return t;
}

// access modifiers and layout keywords that used to follow the block keyword and
// now prefix it — reserved words, so none can ever be a valid buffer name
static bool at_buffer_postfix_mod(const parse_t *p) {
	return at_kw(p, svsl_kw_readonly) || at_kw(p, svsl_kw_writeonly) ||
	       at_kw(p, svsl_kw_coherent) || at_kw(p, svsl_kw_volatile)  ||
	       at_kw(p, svsl_kw_pack1)    || at_kw(p, svsl_kw_pack8)     || at_kw(p, svsl_kw_pack16);
}

// cbuffer/uniform/storagebuffer/pushconstant blocks; access modifiers and the
// pack layout arrive as prefix declaration modifiers ('readonly pack16 storagebuffer B')
static svsl_ast_decl_t *parse_buffer_block(parse_t *p, svsl_block_kind_ kind, bool legacy,
                                           svsl_ast_attrs_t attrs, const decl_mods_t *mods) {
	svsl_ast_decl_t *d = decl_new(p, svsl_decl_block, cur(p)->loc);
	advance(p); // block keyword
	d->block.loc    = d->loc;
	d->block.kind   = kind;
	d->block.legacy = legacy;
	d->block.attrs  = attrs;
	d->block.pack   = (svsl_pack_)mods->pack;

	const uint32_t block_flags = svsl_var_flag_readonly | svsl_var_flag_writeonly |
	                             svsl_var_flag_coherent | svsl_var_flag_volatile;
	d->block.flags = mods->flags & block_flags;
	if ((mods->flags & ~block_flags) || mods->interp)
		svsl_diag_add(p->arena, p->diags, svsl_severity_error, d->loc,
		              "only access modifiers and a layout keyword may prefix a buffer block");

	// catch the old postfix order ('storagebuffer readonly Name') so it gives a
	// clear diagnostic and recovers, instead of consuming the keyword as the name
	if (at_buffer_postfix_mod(p)) {
		svsl_diag_add(p->arena, p->diags, svsl_severity_error, cur(p)->loc,
		              "access modifiers and layout keywords now prefix the block keyword, "
		              "e.g. 'readonly storagebuffer Name' instead of 'storagebuffer readonly Name'");
		while (at_buffer_postfix_mod(p)) advance(p);
	}
	d->block.name = expect_ident(p, "buffer name");
	parse_colon_clauses(p, NULL, &d->block.reg);

	var_list_t members = {0};
	parse_member_list(p, &members);
	accept(p, svsl_tok_semicolon); // optional after blocks (HLSL style)

	d->block.members      = members.items;
	d->block.member_count = members.count;
	return d;
}

static svsl_ast_decl_t *parse_function(parse_t *p, svsl_ast_type_t *return_type, svsl_str_t name,
                                       svsl_ast_attrs_t attrs, svsl_loc_t loc) {
	svsl_ast_decl_t *d = decl_new(p, svsl_decl_func, loc);
	d->func.name        = name;
	d->func.loc         = loc;
	d->func.return_type = return_type;
	d->func.attrs       = attrs;

	expect(p, svsl_tok_lparen, "'('");
	var_list_t params = {0};
	if (!at(p, svsl_tok_rparen)) {
		// 'void' as the whole parameter list
		if (!(at(p, svsl_tok_ident) && svsl_str_eq_cstr(cur(p)->text, "void") &&
		      peek(p, 1)->kind == svsl_tok_rparen)) {
			do {
				decl_mods_t      mods  = parse_decl_mods(p, true);
				reject_pack_mods(p, &mods, cur(p)->loc);
				svsl_ast_type_t *ptype = parse_type(p);
				svsl_ast_var_t  *param = parse_declarator(p, ptype, &mods, (svsl_ast_attrs_t){0});
				svsl_array_push(p->arena, &params, param);
			} while (accept(p, svsl_tok_comma));
		} else {
			advance(p); // void
		}
	}
	expect(p, svsl_tok_rparen, "')' after parameters");
	d->func.params      = params.items;
	d->func.param_count = params.count;

	parse_colon_clauses(p, &d->func.return_semantic, NULL);

	if (accept(p, svsl_tok_semicolon)) return d; // forward declaration
	d->func.body = parse_block(p);
	return d;
}

// handles the keyword-introduced declarations; returns NULL for the
// modifier/type-led forms (globals, functions) which parse_decl_into finishes
static svsl_ast_decl_t *parse_decl(parse_t *p, svsl_ast_attrs_t attrs, svsl_loc_t loc,
                                   const decl_mods_t *mods) {
	if (at_kw(p, svsl_kw_struct)) {
		if (mods->any)
			svsl_diag_add(p->arena, p->diags, svsl_severity_error, loc,
			              "modifiers do not apply to struct declarations");
		svsl_ast_decl_t *d = parse_struct(p);
		d->struct_decl.loc = loc;
		return d;
	}
	if (at_kw(p, svsl_kw_cbuffer))
		return parse_buffer_block(p, svsl_block_uniform, true, attrs, mods);
	if (at_kw(p, svsl_kw_storagebuffer))
		return parse_buffer_block(p, svsl_block_storagebuffer, false, attrs, mods);
	if (at_kw(p, svsl_kw_pushconstant))
		return parse_buffer_block(p, svsl_block_pushconstant, false, attrs, mods);
	if (at_kw(p, svsl_kw_include) && peek(p, 1)->kind == svsl_tok_string_lit) {
		if (mods->any)
			svsl_diag_add(p->arena, p->diags, svsl_severity_error, loc,
			              "modifiers do not apply to includes");
		advance(p);
		svsl_ast_decl_t *d = decl_new(p, svsl_decl_include, loc);
		d->include_path = cur(p)->text;
		advance(p);
		accept(p, svsl_tok_semicolon); // optional
		return d;
	}
	return NULL; // global variables and functions are handled by parse_decl_into
}

// emits one svsl_decl_var per parsed declarator; the first takes `loc` (the decl's
// start, before its type), the rest their own declarator location
static void push_var_list_decls(parse_t *p, decl_list_t *out, const var_list_t *vars, svsl_loc_t loc) {
	for (int32_t i = 0; i < vars->count; i++) {
		svsl_ast_decl_t *d = decl_new(p, svsl_decl_var, i == 0 ? loc : vars->items[i]->loc);
		d->var = *vars->items[i];
		svsl_array_push(p->arena, out, d);
	}
}

// global variable declarations become one decl per declarator
static void push_var_decls(parse_t *p, decl_list_t *out, const decl_mods_t *mods,
                           svsl_ast_attrs_t attrs, svsl_loc_t loc) {
	var_list_t vars = {0};
	parse_var_decl_list(p, mods, attrs, &vars);
	push_var_list_decls(p, out, &vars, loc);
}

// parses one top-level construct, pushing the resulting decl(s)
static void parse_decl_into(parse_t *p, decl_list_t *out) {
	svsl_ast_attrs_t attrs = parse_attrs(p);
	svsl_loc_t       loc   = cur(p)->loc;

	if (at_kw(p, svsl_kw_enum)) {
		svsl_ast_decl_t *ed = parse_enum(p);
		svsl_array_push(p->arena, out, ed);
		// optional inline variable declarators sharing the enum's type (C 'enum {} v;')
		if (at(p, svsl_tok_ident)) {
			svsl_ast_type_t *base = ed->enum_decl.name.len
				? make_type_named(p, ed->enum_decl.name, ed->loc)
				: ed->enum_decl.underlying
					? ed->enum_decl.underlying
					: make_type_named(p, svsl_str("int"), ed->loc);
			decl_mods_t mods = {0};
			var_list_t  vars = {0};
			do {
				svsl_array_push(p->arena, &vars, parse_declarator(p, base, &mods, (svsl_ast_attrs_t){0}));
			} while (accept(p, svsl_tok_comma));
			push_var_list_decls(p, out, &vars, vars.items[0]->loc); // ≥1 declarator here
		}
		expect(p, svsl_tok_semicolon, "';' after enum");
		return;
	}

	// modifiers first: they prefix every declaration form (vars, blocks, resources)
	decl_mods_t mods = parse_decl_mods(p, false);

	svsl_ast_decl_t *simple = parse_decl(p, attrs, loc, &mods);
	if (simple) {
		svsl_array_push(p->arena, out, simple);
		return;
	}
	if (at_kw(p, svsl_kw_uniform)) {
		// 'uniform Type name;' (legacy global) vs 'uniform Name { }' (block)
		if (peek(p, 1)->kind == svsl_tok_ident && is_type_name(p, peek(p, 1)->text) &&
		    peek(p, 2)->kind == svsl_tok_ident) {
			advance(p); // 'uniform' as a legacy modifier on a global
			mods.flags |= svsl_var_flag_uniform | svsl_var_flag_legacy_spelling;
			decl_mods_t more = parse_decl_mods(p, false);
			mods.flags  |= more.flags;
			mods.interp |= more.interp;
			if (more.pack) mods.pack = more.pack;
			push_var_decls(p, out, &mods, attrs, loc);
			return;
		}
		svsl_array_push(p->arena, out, parse_buffer_block(p, svsl_block_uniform, false, attrs, &mods));
		return;
	}

	if (!at(p, svsl_tok_ident)) {
		svsl_diag_add(p->arena, p->diags, svsl_severity_error, loc,
		              "expected declaration, got '%.*s'", cur(p)->text.len ? cur(p)->text.len : 5,
		              cur(p)->text.len ? cur(p)->text.ptr : "<eof>");
		return;
	}
	svsl_ast_type_t *type = parse_type(p);

	if (at(p, svsl_tok_ident) && peek(p, 1)->kind == svsl_tok_lparen) {
		svsl_str_t name = cur(p)->text;
		advance(p);
		if (mods.any)
			svsl_diag_add(p->arena, p->diags, svsl_severity_warning, loc, "modifiers on a function are ignored");
		svsl_array_push(p->arena, out, parse_function(p, type, name, attrs, loc));
		return;
	}

	// global variable(s): resources, bare globals, static const, workgroup arrays
	var_list_t vars = {0};
	do {
		svsl_array_push(p->arena, &vars, parse_declarator(p, type, &mods, attrs));
	} while (accept(p, svsl_tok_comma));
	expect(p, svsl_tok_semicolon, "';' after declaration");
	for (int32_t i = 0; i < vars.count; i++) {
		svsl_ast_decl_t *d = decl_new(p, svsl_decl_var, i == 0 ? loc : vars.items[i]->loc);
		d->var = *vars.items[i];
		svsl_array_push(p->arena, out, d);
	}
}

// pass 1: scan tokens for user type names (struct + named enum) so pass 2 can
// classify identifiers as type names in ambiguous 'name name' positions
static void scan_struct_names(parse_t *p) {
	for (int32_t i = 0; i + 1 < p->count; i++) {
		if (p->toks[i].kind == svsl_tok_ident &&
		    (p->toks[i].keyword == svsl_kw_struct || p->toks[i].keyword == svsl_kw_enum) &&
		    p->toks[i + 1].kind == svsl_tok_ident) {
			svsl_array_push(p->arena, &p->struct_names, p->toks[i + 1].text); // 'enum Name'/'enum : t' → skip ':'
		}
	}
}

static void toplevel_sync(parse_t *p) {
	int32_t depth = 0;
	while (!at(p, svsl_tok_eof)) {
		svsl_tok_ k = cur(p)->kind;
		if (k == svsl_tok_lbrace) depth++;
		if (k == svsl_tok_rbrace) { if (depth > 0) depth--; advance(p); if (depth == 0) { accept(p, svsl_tok_semicolon); return; } continue; }
		if (depth == 0 && k == svsl_tok_semicolon) { advance(p); return; }
		advance(p);
	}
}

svsl_ast_t *svsl_parse(svsl_arena_t *arena, const svsl_token_list_t *tokens,
                       svsl_diag_list_t *ref_diags) {
	parse_t p = {
		.arena = arena,
		.toks  = tokens->items,
		.count = tokens->count,
		.diags = ref_diags };
	scan_struct_names(&p);

	decl_list_t decls = {0};
	while (!at(&p, svsl_tok_eof)) {
		int32_t before = p.pos;
		parse_decl_into(&p, &decls);
		if (p.pos == before) toplevel_sync(&p);
		if (p.pos == before) advance(&p);
	}

	svsl_ast_t *ast = svsl_arena_alloc(arena, sizeof(svsl_ast_t));
	ast->decls      = decls.items;
	ast->decl_count = decls.count;
	return ast;
}

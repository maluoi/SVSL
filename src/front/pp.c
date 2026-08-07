#include "pp.h"

#include "../util/array.h"

#include <stdlib.h>
#include <string.h>

typedef svsl_array_t(char) pp_buf_t;

#define PP_MAX_COND_DEPTH    64
#define PP_MAX_INCLUDE_DEPTH 64
#define PP_MAX_EXPAND_DEPTH  64

typedef struct pp_macro_t {
	svsl_str_t  name;
	svsl_str_t  body;
	svsl_str_t *params;      // NULL when object-like
	int32_t     param_count; // -1 when object-like
	bool        alive;
} pp_macro_t;

typedef struct pp_cond_t {
	bool       parent_active;
	bool       taken;    // some branch of this chain has been active
	bool       active;
	bool       has_else;
	svsl_loc_t loc;      // where the #if started, for unterminated-#if errors
} pp_cond_t;

typedef struct pp_t {
	svsl_arena_t            *arena;
	const svsl_pp_options_t *opt;
	svsl_diag_list_t        *diags;

	svsl_array_t(pp_macro_t)     macros;
	svsl_array_t(const char *)   pragma_once;
	pp_buf_t                     out;
	svsl_array_t(svsl_pp_line_t) lines;
	svsl_array_t(svsl_pp_meta_t) metas;

	// reused scratch buffers (reset per line, capacity persists)
	pp_buf_t logical;
	pp_buf_t clean;
	pp_buf_t expanded;

	pp_cond_t  cond_stack[PP_MAX_COND_DEPTH];
	int32_t    cond_depth;
	svsl_str_t expanding[PP_MAX_EXPAND_DEPTH]; // macro names currently being expanded
	int32_t    expand_depth;
	int32_t    include_depth;
	bool       in_block_comment;
} pp_t;

typedef struct pp_file_t {
	svsl_str_t  src;
	int32_t     pos;
	int32_t     line; // 1-based line number of the next physical line
	const char *name; // arena-owned
} pp_file_t;

static void pp_process_file(pp_t *pp, svsl_str_t src, const char *file_name);

// --- character helpers -------------------------------------------------------

static bool is_ident_start(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static bool is_ident_char (char c) { return is_ident_start(c) || (c >= '0' && c <= '9'); }
static bool is_hspace     (char c) { return c == ' ' || c == '\t' || c == '\r'; }
static bool is_digit      (char c) { return c >= '0' && c <= '9'; }

static bool pp_active(const pp_t *pp) {
	return pp->cond_depth == 0 || pp->cond_stack[pp->cond_depth - 1].active;
}

// --- macro table --------------------------------------------------------------

static pp_macro_t *pp_macro_find(pp_t *pp, svsl_str_t name) {
	// runs for every identifier in the source: reject on the first character
	// before anything else so a miss costs one compare per macro
	char c0 = name.len > 0 ? name.ptr[0] : '\0';
	for (int32_t i = pp->macros.count - 1; i >= 0; i--) {
		const pp_macro_t *m = &pp->macros.items[i];
		if (m->alive && m->name.len > 0 && m->name.ptr[0] == c0 && svsl_str_eq(m->name, name))
			return &pp->macros.items[i];
	}
	return NULL;
}

static bool pp_macro_expanding(const pp_t *pp, svsl_str_t name) {
	for (int32_t i = 0; i < pp->expand_depth; i++)
		if (svsl_str_eq(pp->expanding[i], name)) return true;
	return false;
}

static svsl_str_t pp_str_dup(pp_t *pp, svsl_str_t s) {
	return (svsl_str_t){ .ptr = svsl_arena_strndup(pp->arena, s.ptr, (size_t)s.len), .len = s.len };
}

// --- metadata (//--name[:tag] = value) ----------------------------------------

// Matches skshaderc's sksc_meta_find_defaults: a comment whose content (after
// horizontal whitespace) starts with "--". Name/tag/value split on ':' and '='.
static void pp_meta_check(pp_t *pp, svsl_str_t comment, const char *file, int32_t line, int32_t col) {
	if (pp->include_depth > 0) return; // skshaderc only scans the main file

	int32_t start = 0;
	while (start < comment.len && (comment.ptr[start] == ' ' || comment.ptr[start] == '\t')) start++;
	if (comment.len - start < 2 || comment.ptr[start] != '-' || comment.ptr[start + 1] != '-') return;

	svsl_str_t body      = svsl_str_slice(comment, start + 2, comment.len);
	int32_t    value_pos = svsl_str_find_char(body, '=');
	int32_t    tag_pos   = svsl_str_find_char(body, ':');
	if (tag_pos >= 0 && value_pos >= 0 && tag_pos > value_pos) tag_pos = -1; // ':' inside the value

	int32_t    name_end = tag_pos >= 0 ? tag_pos : (value_pos >= 0 ? value_pos : body.len);
	svsl_str_t name     = svsl_str_trim(svsl_str_slice(body, 0, name_end));
	svsl_str_t tag      = {0};
	svsl_str_t value    = {0};
	if (tag_pos   >= 0) tag   = svsl_str_trim(svsl_str_slice(body, tag_pos + 1, value_pos >= 0 ? value_pos : body.len));
	if (value_pos >= 0) value = svsl_str_trim(svsl_str_slice(body, value_pos + 1, body.len));

	svsl_array_push(pp->arena, &pp->metas, (svsl_pp_meta_t){
		.name  = pp_str_dup(pp, name),
		.tag   = pp_str_dup(pp, tag),
		.value = pp_str_dup(pp, value),
		.loc   = { .file = file, .line = line, .col = col + start + 1 } });
}

// --- comment stripping ---------------------------------------------------------

// Strips comments from one logical line into pp->clean (reset first), collecting
// metadata. Block-comment state persists across lines in pp->in_block_comment.
static void pp_strip_comments(pp_t *pp, svsl_str_t line, const char *file, int32_t line_no) {
	pp->clean.count = 0;
	int32_t i = 0;

	if (pp->in_block_comment) {
		int32_t end = line.len;
		bool    closed = false;
		for (int32_t j = 0; j + 1 < line.len; j++) {
			if (line.ptr[j] == '*' && line.ptr[j + 1] == '/') { end = j; closed = true; break; }
		}
		pp_meta_check(pp, svsl_str_slice(line, 0, end), file, line_no, 0);
		if (!closed) return; // whole line is comment
		pp->in_block_comment = false;
		i = end + 2;
		svsl_array_push(pp->arena, &pp->clean, ' ');
	}

	while (i < line.len) {
		char c    = line.ptr[i];
		char next = i + 1 < line.len ? line.ptr[i + 1] : '\0';

		if (c == '/' && next == '/') {
			pp_meta_check(pp, svsl_str_slice(line, i + 2, line.len), file, line_no, i + 2);
			break;
		}
		if (c == '/' && next == '*') {
			int32_t end    = line.len;
			bool    closed = false;
			for (int32_t j = i + 2; j + 1 < line.len; j++) {
				if (line.ptr[j] == '*' && line.ptr[j + 1] == '/') { end = j; closed = true; break; }
			}
			pp_meta_check(pp, svsl_str_slice(line, i + 2, end), file, line_no, i + 2);
			svsl_array_push(pp->arena, &pp->clean, ' ');
			if (!closed) { pp->in_block_comment = true; return; }
			i = end + 2;
			continue;
		}
		if (c == '"') { // copy string literals verbatim, comments inside don't count
			svsl_array_push(pp->arena, &pp->clean, c);
			i++;
			while (i < line.len && line.ptr[i] != '"') {
				if (line.ptr[i] == '\\' && i + 1 < line.len) { svsl_array_push(pp->arena, &pp->clean, line.ptr[i]); i++; }
				svsl_array_push(pp->arena, &pp->clean, line.ptr[i]);
				i++;
			}
			if (i < line.len) { svsl_array_push(pp->arena, &pp->clean, '"'); i++; }
			continue;
		}
		svsl_array_push(pp->arena, &pp->clean, c);
		i++;
	}
}

// --- macro expansion ------------------------------------------------------------

static void pp_expand(pp_t *pp, svsl_str_t text, pp_buf_t *out, svsl_loc_t loc);

static void pp_append_str(pp_t *pp, pp_buf_t *out, svsl_str_t s) {
	for (int32_t i = 0; i < s.len; i++)
		svsl_array_push(pp->arena, out, s.ptr[i]);
}

// Parses a function-like macro's argument list starting at text[*ref_i] == '('.
// Returns false (and doesn't move *ref_i) if the list is unterminated.
static bool pp_parse_args(svsl_str_t text, int32_t *ref_i,
                          svsl_str_t out_args[], int32_t *out_count, int32_t max_args) {
	int32_t i     = *ref_i + 1; // past '('
	int32_t depth = 1;
	int32_t start = i;
	int32_t count = 0;

	while (i < text.len && depth > 0) {
		char c = text.ptr[i];
		if      (c == '(') depth++;
		else if (c == ')') depth--;
		else if (c == ',' && depth == 1) {
			if (count < max_args) out_args[count] = svsl_str_trim(svsl_str_slice(text, start, i));
			count++;
			start = i + 1;
		}
		i++;
	}
	if (depth > 0) return false;

	svsl_str_t last = svsl_str_trim(svsl_str_slice(text, start, i - 1));
	if (count > 0 || last.len > 0) {
		if (count < max_args) out_args[count] = last;
		count++;
	}
	*out_count = count;
	*ref_i     = i;
	return true;
}

#define PP_MAX_ARGS 32

static void pp_expand_macro(pp_t *pp, pp_macro_t *macro, svsl_str_t args[], int32_t arg_count,
                            pp_buf_t *out, svsl_loc_t loc) {
	if (pp->expand_depth >= PP_MAX_EXPAND_DEPTH) {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc,
		              "macro expansion too deep expanding '%.*s'", macro->name.len, macro->name.ptr);
		return;
	}

	svsl_str_t body = macro->body;
	if (macro->param_count >= 0) {
		// Fully expand each argument once, then substitute into the body.
		svsl_str_t expanded_args[PP_MAX_ARGS];
		for (int32_t a = 0; a < arg_count && a < macro->param_count; a++) {
			pp_buf_t buf = {0};
			pp_expand(pp, args[a], &buf, loc);
			expanded_args[a] = (svsl_str_t){ .ptr = buf.items, .len = buf.count };
		}

		pp_buf_t subst = {0};
		int32_t i = 0;
		while (i < body.len) {
			char c = body.ptr[i];
			if (is_ident_start(c)) {
				int32_t start = i;
				while (i < body.len && is_ident_char(body.ptr[i])) i++;
				svsl_str_t ident = svsl_str_slice(body, start, i);
				int32_t    param = -1;
				for (int32_t k = 0; k < macro->param_count; k++)
					if (svsl_str_eq(macro->params[k], ident)) { param = k; break; }
				if (param >= 0 && param < arg_count) pp_append_str(pp, &subst, expanded_args[param]);
				else                                 pp_append_str(pp, &subst, ident);
				continue;
			}
			svsl_array_push(pp->arena, &subst, c);
			i++;
		}
		body = (svsl_str_t){ .ptr = subst.items, .len = subst.count };
	}

	pp->expanding[pp->expand_depth++] = macro->name;
	pp_expand(pp, body, out, loc);
	pp->expand_depth--;
}

static void pp_expand(pp_t *pp, svsl_str_t text, pp_buf_t *out, svsl_loc_t loc) {
	int32_t i = 0;
	while (i < text.len) {
		char c = text.ptr[i];

		if (c == '"') { // don't expand inside string literals
			svsl_array_push(pp->arena, out, c);
			i++;
			while (i < text.len && text.ptr[i] != '"') {
				if (text.ptr[i] == '\\' && i + 1 < text.len) { svsl_array_push(pp->arena, out, text.ptr[i]); i++; }
				svsl_array_push(pp->arena, out, text.ptr[i]);
				i++;
			}
			if (i < text.len) { svsl_array_push(pp->arena, out, '"'); i++; }
			continue;
		}
		if (is_digit(c)) { // numbers can contain ident chars (0xFF, 1e5f) — copy whole
			while (i < text.len && (is_ident_char(text.ptr[i]) || text.ptr[i] == '.')) {
				svsl_array_push(pp->arena, out, text.ptr[i]);
				i++;
			}
			continue;
		}
		if (!is_ident_start(c)) {
			svsl_array_push(pp->arena, out, c);
			i++;
			continue;
		}

		int32_t start = i;
		while (i < text.len && is_ident_char(text.ptr[i])) i++;
		svsl_str_t  ident = svsl_str_slice(text, start, i);
		pp_macro_t *macro = pp_macro_find(pp, ident);

		if (!macro || pp_macro_expanding(pp, ident)) {
			pp_append_str(pp, out, ident);
			continue;
		}
		if (macro->param_count < 0) { // object-like
			pp_expand_macro(pp, macro, NULL, 0, out, loc);
			continue;
		}

		// Function-like: only an invocation if '(' follows on this logical line.
		int32_t peek = i;
		while (peek < text.len && is_hspace(text.ptr[peek])) peek++;
		if (peek >= text.len || text.ptr[peek] != '(') {
			pp_append_str(pp, out, ident);
			continue;
		}

		svsl_str_t args[PP_MAX_ARGS];
		int32_t    arg_count = 0;
		int32_t    args_end  = peek;
		if (!pp_parse_args(text, &args_end, args, &arg_count, PP_MAX_ARGS)) {
			svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc,
			              "unterminated argument list for macro '%.*s'", ident.len, ident.ptr);
			pp_append_str(pp, out, ident);
			continue;
		}
		if (arg_count != macro->param_count && !(macro->param_count == 0 && arg_count == 0)) {
			svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc,
			              "macro '%.*s' expects %d argument(s), got %d",
			              ident.len, ident.ptr, macro->param_count, arg_count);
			i = args_end;
			continue;
		}
		i = args_end;
		pp_expand_macro(pp, macro, args, arg_count, out, loc);
	}
}

// --- #if expression evaluation ---------------------------------------------------

typedef struct pp_expr_t {
	const char *p;
	const char *end;
	pp_t       *pp;
	svsl_loc_t  loc;
	int32_t     depth; // recursion guard for nested/prefix/ternary expressions
	bool        failed;
} pp_expr_t;

#define PP_EXPR_MAX_DEPTH 256

static void expr_error(pp_expr_t *e, const char *msg) {
	if (!e->failed)
		svsl_diag_add(e->pp->arena, e->pp->diags, svsl_severity_error, e->loc, "%s in #if expression", msg);
	e->failed = true;
}

static void expr_skip_ws(pp_expr_t *e) {
	while (e->p < e->end && is_hspace(*e->p)) e->p++;
}

static bool expr_accept(pp_expr_t *e, const char *op) {
	expr_skip_ws(e);
	size_t len = strlen(op);
	if ((size_t)(e->end - e->p) < len || strncmp(e->p, op, len) != 0) return false;
	// don't match "<" when the text is "<<" or "<="
	if (len == 1 && (op[0] == '<' || op[0] == '>' || op[0] == '&' || op[0] == '|' || op[0] == '=') &&
	    e->p + 1 < e->end && (e->p[1] == op[0] || e->p[1] == '=')) return false;
	e->p += len;
	return true;
}

static int64_t expr_ternary(pp_expr_t *e);
static int64_t expr_primary_inner(pp_expr_t *e);

// expr_primary and expr_ternary are the two recursion entry points (prefix/paren and
// ternary arms); guarding both caps total nesting depth. expr_binary sits between them
// and is bounded by the finite precedence ladder.
static int64_t expr_primary(pp_expr_t *e) {
	if (e->depth >= PP_EXPR_MAX_DEPTH) { expr_error(e, "expression nesting too deep"); return 0; }
	e->depth++;
	int64_t v = expr_primary_inner(e);
	e->depth--;
	return v;
}

static int64_t expr_primary_inner(pp_expr_t *e) {
	expr_skip_ws(e);
	if (e->p >= e->end) { expr_error(e, "unexpected end"); return 0; }

	char c = *e->p;
	if (c == '(') {
		e->p++;
		int64_t v = expr_ternary(e);
		expr_skip_ws(e);
		if (e->p < e->end && *e->p == ')') e->p++;
		else expr_error(e, "missing ')'");
		return v;
	}
	if (c == '!') { e->p++; return !expr_primary(e); }
	if (c == '~') { e->p++; return ~expr_primary(e); }
	if (c == '-') { e->p++; return -expr_primary(e); }
	if (c == '+') { e->p++; return  expr_primary(e); }
	if (is_digit(c)) {
		char   *num_end = NULL;
		int64_t v       = (int64_t)strtoll(e->p, &num_end, 0); // dec/hex/octal
		if (num_end > e->p && num_end <= e->end) {
			// skip integer suffixes
			while (num_end < e->end && (*num_end == 'u' || *num_end == 'U' || *num_end == 'l' || *num_end == 'L'))
				num_end++;
			e->p = num_end;
			return v;
		}
		expr_error(e, "bad number");
		return 0;
	}
	if (is_ident_start(c)) { // surviving identifiers evaluate to 0
		while (e->p < e->end && is_ident_char(*e->p)) e->p++;
		return 0;
	}
	expr_error(e, "unexpected character");
	e->p++;
	return 0;
}

// Precedence climbing over C operators.
static int64_t expr_binary(pp_expr_t *e, int32_t min_prec) {
	int64_t lhs = expr_primary(e);

	for (;;) {
		expr_skip_ws(e);
		int32_t     prec = 0;
		const char *op   = NULL;
		static const struct { const char *op; int32_t prec; } ops[] = {
			{ "||", 1 }, { "&&", 2 }, { "|", 3 }, { "^", 4 }, { "&", 5 },
			{ "==", 6 }, { "!=", 6 },
			{ "<=", 7 }, { ">=", 7 }, { "<<", 8 }, { ">>", 8 }, { "<", 7 }, { ">", 7 },
			{ "+", 9 }, { "-", 9 }, { "*", 10 }, { "/", 10 }, { "%", 10 },
		};
		for (int32_t k = 0; k < (int32_t)(sizeof(ops) / sizeof(ops[0])); k++) {
			pp_expr_t save = *e;
			if (ops[k].prec >= min_prec && expr_accept(e, ops[k].op)) { op = ops[k].op; prec = ops[k].prec; break; }
			*e = save;
		}
		if (!op) return lhs;

		int64_t rhs = expr_binary(e, prec + 1);
		if      (strcmp(op, "||") == 0) lhs = lhs || rhs;
		else if (strcmp(op, "&&") == 0) lhs = lhs && rhs;
		else if (strcmp(op, "|")  == 0) lhs = lhs | rhs;
		else if (strcmp(op, "^")  == 0) lhs = lhs ^ rhs;
		else if (strcmp(op, "&")  == 0) lhs = lhs & rhs;
		else if (strcmp(op, "==") == 0) lhs = lhs == rhs;
		else if (strcmp(op, "!=") == 0) lhs = lhs != rhs;
		else if (strcmp(op, "<=") == 0) lhs = lhs <= rhs;
		else if (strcmp(op, ">=") == 0) lhs = lhs >= rhs;
		else if (strcmp(op, "<<") == 0) lhs = lhs << (rhs & 63);
		else if (strcmp(op, ">>") == 0) lhs = lhs >> (rhs & 63);
		else if (strcmp(op, "<")  == 0) lhs = lhs < rhs;
		else if (strcmp(op, ">")  == 0) lhs = lhs > rhs;
		else if (strcmp(op, "+")  == 0) lhs = lhs + rhs;
		else if (strcmp(op, "-")  == 0) lhs = lhs - rhs;
		else if (strcmp(op, "*")  == 0) lhs = lhs * rhs;
		else if (strcmp(op, "%")  == 0 || strcmp(op, "/") == 0) {
			if (rhs == 0) { expr_error(e, "division by zero"); lhs = 0; }
			else lhs = op[0] == '/' ? lhs / rhs : lhs % rhs;
		}
	}
}

static int64_t expr_ternary_inner(pp_expr_t *e);

static int64_t expr_ternary(pp_expr_t *e) {
	if (e->depth >= PP_EXPR_MAX_DEPTH) { expr_error(e, "expression nesting too deep"); return 0; }
	e->depth++;
	int64_t v = expr_ternary_inner(e);
	e->depth--;
	return v;
}

static int64_t expr_ternary_inner(pp_expr_t *e) {
	int64_t cond = expr_binary(e, 1);
	expr_skip_ws(e);
	if (e->p < e->end && *e->p == '?') {
		e->p++;
		int64_t a = expr_ternary(e);
		expr_skip_ws(e);
		if (e->p < e->end && *e->p == ':') e->p++;
		else expr_error(e, "missing ':'");
		int64_t b = expr_ternary(e);
		return cond ? a : b;
	}
	return cond;
}

// Resolves defined(X) / defined X before macro expansion, then expands and evaluates.
static bool pp_eval_condition(pp_t *pp, svsl_str_t text, svsl_loc_t loc) {
	pp_buf_t resolved = {0};
	int32_t i = 0;
	while (i < text.len) {
		char c = text.ptr[i];
		if (!is_ident_start(c)) {
			svsl_array_push(pp->arena, &resolved, c);
			i++;
			continue;
		}
		int32_t start = i;
		while (i < text.len && is_ident_char(text.ptr[i])) i++;
		svsl_str_t ident = svsl_str_slice(text, start, i);
		if (!svsl_str_eq_cstr(ident, "defined")) {
			pp_append_str(pp, &resolved, ident);
			continue;
		}
		while (i < text.len && is_hspace(text.ptr[i])) i++;
		bool parens = i < text.len && text.ptr[i] == '(';
		if (parens) i++;
		while (i < text.len && is_hspace(text.ptr[i])) i++;
		int32_t name_start = i;
		while (i < text.len && is_ident_char(text.ptr[i])) i++;
		svsl_str_t name = svsl_str_slice(text, name_start, i);
		if (parens) {
			while (i < text.len && is_hspace(text.ptr[i])) i++;
			if (i < text.len && text.ptr[i] == ')') i++;
			else svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "missing ')' after defined(");
		}
		if (name.len == 0)
			svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "expected identifier after 'defined'");
		svsl_array_push(pp->arena, &resolved, pp_macro_find(pp, name) ? '1' : '0');
	}

	pp_buf_t expanded = {0};
	pp_expand(pp, (svsl_str_t){ .ptr = resolved.items, .len = resolved.count }, &expanded, loc);

	pp_expr_t e = {
		.p   = expanded.items,
		.end = expanded.items + expanded.count,
		.pp  = pp,
		.loc = loc };
	int64_t v = expr_ternary(&e);
	expr_skip_ws(&e);
	if (!e.failed && e.p < e.end)
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "trailing characters in #if expression");
	return v != 0;
}

// --- directives -------------------------------------------------------------------

static void pp_directive_define(pp_t *pp, svsl_str_t rest, svsl_loc_t loc) {
	rest = svsl_str_trim(rest);
	int32_t i = 0;
	while (i < rest.len && is_ident_char(rest.ptr[i])) i++;
	svsl_str_t name = svsl_str_slice(rest, 0, i);
	if (name.len == 0 || !is_ident_start(name.ptr[0])) {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "expected macro name after #define");
		return;
	}

	svsl_str_t *params      = NULL;
	int32_t     param_count = -1;
	if (i < rest.len && rest.ptr[i] == '(') { // '(' must directly follow the name
		svsl_str_t raw[PP_MAX_ARGS];
		param_count = 0;
		if (!pp_parse_args(rest, &i, raw, &param_count, PP_MAX_ARGS)) {
			svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "unterminated parameter list in #define");
			return;
		}
		if (param_count > PP_MAX_ARGS) {
			svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "too many macro parameters (max %d)", PP_MAX_ARGS);
			return;
		}
		for (int32_t k = 0; k < param_count; k++) {
			if (raw[k].len == 0 || !is_ident_start(raw[k].ptr[0])) {
				svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "bad macro parameter name");
				return;
			}
		}
		params = svsl_arena_alloc(pp->arena, (size_t)(param_count > 0 ? param_count : 1) * sizeof(svsl_str_t));
		for (int32_t k = 0; k < param_count; k++) params[k] = pp_str_dup(pp, raw[k]);
	}

	svsl_str_t body = svsl_str_trim(svsl_str_slice(rest, i, rest.len));
	svsl_array_push(pp->arena, &pp->macros, (pp_macro_t){
		.name        = pp_str_dup(pp, name),
		.body        = pp_str_dup(pp, body),
		.params      = params,
		.param_count = param_count,
		.alive       = true });
}

static void pp_directive_include(pp_t *pp, svsl_str_t rest, const char *requester, svsl_loc_t loc, bool once) {
	rest = svsl_str_trim(rest);
	if (rest.len < 2 || (rest.ptr[0] != '"' && rest.ptr[0] != '<')) {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "expected \"file\" or <file> after #include");
		return;
	}
	char    close = rest.ptr[0] == '"' ? '"' : '>';
	int32_t end   = svsl_str_find_char(svsl_str_slice(rest, 1, rest.len), close);
	if (end < 0) {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "unterminated #include path");
		return;
	}
	svsl_str_t path = svsl_str_slice(rest, 1, 1 + end);

	if (pp->include_depth >= PP_MAX_INCLUDE_DEPTH) {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "#include nesting too deep (cycle?)");
		return;
	}
	if (!pp->opt || !pp->opt->include_cb) {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc,
		              "#include '%.*s': no include callback provided", path.len, path.ptr);
		return;
	}

	char *path_cstr = svsl_arena_strndup(pp->arena, path.ptr, (size_t)path.len);
	svsl_include_src_t src = pp->opt->include_cb(pp->opt->include_user, path_cstr, requester);
	if (!src.content) {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "cannot open include file '%s'", path_cstr);
		return;
	}
	const char *resolved = src.path ? src.path : path_cstr;
	for (int32_t k = 0; k < pp->pragma_once.count; k++)
		if (strcmp(pp->pragma_once.items[k], resolved) == 0) return;
	if (once) { // language-level `include` is idempotent
		const char *pin = svsl_arena_strndup(pp->arena, resolved, strlen(resolved));
		svsl_array_push(pp->arena, &pp->pragma_once, pin);
	}

	int32_t     length    = src.length >= 0 ? src.length : (int32_t)strlen(src.content);
	const char *file_name = svsl_arena_strndup(pp->arena, resolved, strlen(resolved));
	// copy the content so the caller may free it when the callback returns ownership
	const char *content   = svsl_arena_strndup(pp->arena, src.content, (size_t)length);

	pp->include_depth++;
	pp_process_file(pp, (svsl_str_t){ .ptr = content, .len = length }, file_name);
	pp->include_depth--;
}

static void pp_directive(pp_t *pp, svsl_str_t line, const char *file, int32_t line_no) {
	svsl_loc_t loc = { .file = file, .line = line_no, .col = 1 };

	int32_t i = 0;
	while (i < line.len && is_hspace(line.ptr[i])) i++;
	i++; // '#'
	while (i < line.len && is_hspace(line.ptr[i])) i++;
	int32_t start = i;
	while (i < line.len && is_ident_char(line.ptr[i])) i++;
	svsl_str_t name = svsl_str_slice(line, start, i);
	svsl_str_t rest = svsl_str_trim(svsl_str_slice(line, i, line.len));

	bool active = pp_active(pp);

	// Conditionals are tracked even in inactive regions (nesting must balance).
	if (svsl_str_eq_cstr(name, "if") || svsl_str_eq_cstr(name, "ifdef") || svsl_str_eq_cstr(name, "ifndef")) {
		if (pp->cond_depth >= PP_MAX_COND_DEPTH) {
			svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "#if nesting too deep");
			return;
		}
		bool cond = false;
		if (active) {
			if (svsl_str_eq_cstr(name, "if")) {
				if (rest.len == 0) svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "#if with no expression");
				else               cond = pp_eval_condition(pp, rest, loc);
			} else {
				svsl_str_t macro_name = rest;
				int32_t    n          = 0;
				while (n < macro_name.len && is_ident_char(macro_name.ptr[n])) n++;
				macro_name = svsl_str_slice(macro_name, 0, n);
				if (macro_name.len == 0)
					svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "expected identifier after #%.*s", name.len, name.ptr);
				cond = pp_macro_find(pp, macro_name) != NULL;
				if (svsl_str_eq_cstr(name, "ifndef")) cond = !cond;
			}
		}
		pp->cond_stack[pp->cond_depth++] = (pp_cond_t){
			.parent_active = active,
			.taken         = !active || cond, // inactive parent: no branch may activate
			.active        = active && cond,
			.loc           = loc };
		return;
	}
	if (svsl_str_eq_cstr(name, "elif")) {
		if (pp->cond_depth == 0) { svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "#elif without #if"); return; }
		pp_cond_t *top = &pp->cond_stack[pp->cond_depth - 1];
		if (top->has_else) { svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "#elif after #else"); return; }
		if (top->taken) {
			top->active = false;
		} else {
			bool cond   = pp_eval_condition(pp, rest, loc);
			top->active = cond;
			top->taken  = cond;
		}
		return;
	}
	if (svsl_str_eq_cstr(name, "else")) {
		if (pp->cond_depth == 0) { svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "#else without #if"); return; }
		pp_cond_t *top = &pp->cond_stack[pp->cond_depth - 1];
		if (top->has_else) { svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "duplicate #else"); return; }
		top->has_else = true;
		top->active   = !top->taken;
		top->taken    = true;
		return;
	}
	if (svsl_str_eq_cstr(name, "endif")) {
		if (pp->cond_depth == 0) { svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "#endif without #if"); return; }
		pp->cond_depth--;
		return;
	}

	if (!active) return; // all other directives are skipped in inactive regions

	if      (svsl_str_eq_cstr(name, "define"))  pp_directive_define(pp, rest, loc);
	else if (svsl_str_eq_cstr(name, "undef")) {
		int32_t n = 0;
		while (n < rest.len && is_ident_char(rest.ptr[n])) n++;
		// a redefinition pushes a second entry rather than replacing the first
		// (lookup scans in reverse, so the last one wins), so #undef has to clear
		// every entry with the name or an older definition resurfaces
		svsl_str_t target = svsl_str_slice(rest, 0, n);
		for (int32_t i = 0; i < pp->macros.count; i++)
			if (svsl_str_eq(pp->macros.items[i].name, target)) pp->macros.items[i].alive = false;
	}
	else if (svsl_str_eq_cstr(name, "include")) pp_directive_include(pp, rest, file, loc, false);
	else if (svsl_str_eq_cstr(name, "pragma")) {
		if (svsl_str_eq_cstr(svsl_str_trim(rest), "once")) {
			svsl_array_push(pp->arena, &pp->pragma_once, file);
		} else {
			svsl_diag_add(pp->arena, pp->diags, svsl_severity_info, loc, "ignoring '#pragma %.*s'", rest.len, rest.ptr);
		}
	}
	else if (svsl_str_eq_cstr(name, "error")) {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "#error %.*s", rest.len, rest.ptr);
	}
	else if (name.len == 0) { /* '#' alone is a null directive */ }
	else {
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error, loc, "unknown directive '#%.*s'", name.len, name.ptr);
	}
}

// --- file processing -----------------------------------------------------------------

// Assembles the next logical line (backslash-newline spliced) into pp->logical.
// Returns false at end of file.
static bool pp_next_logical_line(pp_t *pp, pp_file_t *f, int32_t *out_line_no) {
	if (f->pos >= f->src.len) return false;
	pp->logical.count = 0;
	*out_line_no      = f->line;

	while (f->pos < f->src.len) {
		int32_t start = f->pos;
		while (f->pos < f->src.len && f->src.ptr[f->pos] != '\n') f->pos++;
		int32_t end = f->pos;
		if (f->pos < f->src.len) f->pos++; // consume '\n'
		f->line++;
		if (end > start && f->src.ptr[end - 1] == '\r') end--;

		bool spliced = end > start && f->src.ptr[end - 1] == '\\';
		if (spliced) end--;
		for (int32_t k = start; k < end; k++)
			svsl_array_push(pp->arena, &pp->logical, f->src.ptr[k]);
		if (!spliced) break;
	}
	return true;
}

static void pp_process_file(pp_t *pp, svsl_str_t src, const char *file_name) {
	// skip a UTF-8 byte-order mark
	if (src.len >= 3 && (uint8_t)src.ptr[0] == 0xEF && (uint8_t)src.ptr[1] == 0xBB && (uint8_t)src.ptr[2] == 0xBF)
		src = svsl_str_slice(src, 3, src.len);

	pp_file_t f = { .src = src, .pos = 0, .line = 1, .name = file_name };

	bool    saved_block  = pp->in_block_comment;
	int32_t base_depth   = pp->cond_depth;
	pp->in_block_comment = false;

	int32_t line_no = 0;
	while (pp_next_logical_line(pp, &f, &line_no)) {
		svsl_str_t logical = { .ptr = pp->logical.items, .len = pp->logical.count };
		bool was_in_comment = pp->in_block_comment;
		pp_strip_comments(pp, logical, file_name, line_no);
		svsl_str_t clean = { .ptr = pp->clean.items, .len = pp->clean.count };

		// a line that was entirely inside a block comment has no code on it
		if (was_in_comment && pp->clean.count == 0) continue;

		int32_t first = 0;
		while (first < clean.len && is_hspace(clean.ptr[first])) first++;

		if (first < clean.len && clean.ptr[first] == '#') {
			pp_directive(pp, clean, file_name, line_no);
			continue;
		}
		if (!pp_active(pp)) continue;

		// language-level `include "file"` — resolved like #include, but idempotent
		if (first + 7 < clean.len && memcmp(clean.ptr + first, "include", 7) == 0) {
			int32_t after = first + 7;
			while (after < clean.len && is_hspace(clean.ptr[after])) after++;
			if (after < clean.len && clean.ptr[after] == '"' && after > first + 7) {
				svsl_str_t rest = svsl_str_trim(svsl_str_slice(clean, after, clean.len));
				if (rest.len > 0 && rest.ptr[rest.len - 1] == ';')
					rest = svsl_str_trim(svsl_str_slice(rest, 0, rest.len - 1));
				pp_directive_include(pp, rest,
				                     file_name, (svsl_loc_t){ .file = file_name, .line = line_no, .col = 1 }, true);
				continue;
			}
		}

		pp->expanded.count = 0;
		svsl_loc_t loc = { .file = file_name, .line = line_no, .col = 1 };
		pp_expand(pp, clean, &pp->expanded, loc);

		for (int32_t k = 0; k < pp->expanded.count; k++)
			svsl_array_push(pp->arena, &pp->out, pp->expanded.items[k]);
		svsl_array_push(pp->arena, &pp->out, '\n');
		svsl_array_push(pp->arena, &pp->lines, (svsl_pp_line_t){ .file = file_name, .line = line_no });
	}

	if (pp->in_block_comment)
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error,
		              (svsl_loc_t){ .file = file_name, .line = line_no, .col = 1 },
		              "unterminated block comment");
	while (pp->cond_depth > base_depth) {
		pp->cond_depth--;
		svsl_diag_add(pp->arena, pp->diags, svsl_severity_error,
		              pp->cond_stack[pp->cond_depth].loc, "unterminated #if");
	}
	pp->in_block_comment = saved_block;
}

// --- entry point --------------------------------------------------------------------

bool svsl_pp_run(svsl_arena_t *arena, const char *source, const char *opt_filename,
                 const svsl_pp_options_t *opt_options, svsl_pp_result_t *out_result,
                 svsl_diag_list_t *ref_diags) {
	pp_t pp = {
		.arena = arena,
		.opt   = opt_options,
		.diags = ref_diags };

	if (opt_options) {
		for (int32_t i = 0; i < opt_options->define_count; i++) {
			const svsl_define_t *def = &opt_options->defines[i];
			svsl_array_push(arena, &pp.macros, (pp_macro_t){
				.name        = pp_str_dup(&pp, svsl_str(def->name)),
				.body        = pp_str_dup(&pp, svsl_str(def->value ? def->value : "1")),
				.param_count = -1,
				.alive       = true });
		}
	}

	const char *file_name = svsl_arena_strndup(arena,
		opt_filename ? opt_filename : "<source>",
		strlen(opt_filename ? opt_filename : "<source>"));
	pp_process_file(&pp, svsl_str(source), file_name);

	svsl_array_push(arena, &pp.out, '\0');
	*out_result = (svsl_pp_result_t){
		.text       = pp.out.items,
		.text_len   = pp.out.count - 1,
		.lines      = pp.lines.items,
		.line_count = pp.lines.count,
		.metas      = pp.metas.items,
		.meta_count = pp.metas.count };
	return ref_diags->error_count == 0;
}

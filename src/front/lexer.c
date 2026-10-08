#include "lexer.h"
#include "chars.h"

#include "../tables/keywords.h"
#include "../util/array.h"

#include <stdlib.h>
#include <string.h>

typedef struct lex_t {
	svsl_arena_t           *arena;
	const svsl_pp_result_t *pp;
	svsl_str_t              src;
	int32_t                 pos;
	int32_t                 out_line;   // index into pp->lines
	int32_t                 line_start; // src offset of the current line's first char
	svsl_token_list_t      *tokens;
	svsl_diag_list_t       *diags;
	svsl_keyword_index_t    keywords;
} lex_t;

// Punctuators by first character, longest first (len derived from the literal)
#define P(text, tok) { text, (int32_t)sizeof(text) - 1, tok }
static const struct { const char *text; int32_t len; svsl_tok_ tok; } punct_table[128][4] = {
	['<'] = { P("<<=", svsl_tok_shl_assign), P("<<", svsl_tok_shl), P("<=", svsl_tok_le), P("<", svsl_tok_lt) },
	['>'] = { P(">>=", svsl_tok_shr_assign), P(">>", svsl_tok_shr), P(">=", svsl_tok_ge), P(">", svsl_tok_gt) },
	['='] = { P("==", svsl_tok_eq), P("=", svsl_tok_assign) },
	['!'] = { P("!=", svsl_tok_neq), P("!", svsl_tok_not) },
	['&'] = { P("&&", svsl_tok_andand), P("&=", svsl_tok_and_assign), P("&", svsl_tok_amp) },
	['|'] = { P("||", svsl_tok_oror), P("|=", svsl_tok_or_assign), P("|", svsl_tok_pipe) },
	['+'] = { P("++", svsl_tok_plusplus), P("+=", svsl_tok_plus_assign), P("+", svsl_tok_plus) },
	['-'] = { P("--", svsl_tok_minusminus), P("-=", svsl_tok_minus_assign), P("-", svsl_tok_minus) },
	['*'] = { P("*=", svsl_tok_star_assign), P("*", svsl_tok_star) },
	['/'] = { P("/=", svsl_tok_slash_assign), P("/", svsl_tok_slash) },
	['%'] = { P("%=", svsl_tok_percent_assign), P("%", svsl_tok_percent) },
	['^'] = { P("^=", svsl_tok_xor_assign), P("^", svsl_tok_caret) },
	[':'] = { P("::", svsl_tok_coloncolon), P(":", svsl_tok_colon) },
	['('] = { P("(", svsl_tok_lparen) },
	[')'] = { P(")", svsl_tok_rparen) },
	['['] = { P("[", svsl_tok_lbracket) },
	[']'] = { P("]", svsl_tok_rbracket) },
	['{'] = { P("{", svsl_tok_lbrace) },
	['}'] = { P("}", svsl_tok_rbrace) },
	[','] = { P(",", svsl_tok_comma) },
	[';'] = { P(";", svsl_tok_semicolon) },
	['$'] = { P("$", svsl_tok_dollar) },
	['.'] = { P(".", svsl_tok_dot) },
	['?'] = { P("?", svsl_tok_question) },
	['~'] = { P("~", svsl_tok_tilde) },
};
#undef P


static svsl_loc_t lex_loc(const lex_t *lex, int32_t pos) {
	svsl_loc_t loc = { .file = NULL, .line = 0, .col = pos - lex->line_start + 1 };
	if (lex->out_line < lex->pp->line_count) {
		loc.file = lex->pp->lines[lex->out_line].file;
		loc.line = lex->pp->lines[lex->out_line].line;
	} else if (lex->pp->line_count > 0) {
		loc.file = lex->pp->lines[lex->pp->line_count - 1].file;
		loc.line = lex->pp->lines[lex->pp->line_count - 1].line;
	}
	return loc;
}

static void lex_push(lex_t *lex, svsl_token_t token) {
	svsl_array_push(lex->arena, lex->tokens, token);
}

// Scans a numeric literal starting at lex->pos. Handles hex/binary/octal/decimal
// integers, decimal floats (including '.5' and '1.'), exponents, and suffixes.
static void lex_number(lex_t *lex) {
	svsl_str_t src   = lex->src;
	int32_t    start = lex->pos;
	int32_t    i     = start;
	svsl_loc_t loc   = lex_loc(lex, start);

	bool is_float = false;
	bool is_hex   = false;
	bool is_bin   = false;

	if (src.ptr[i] == '0' && i + 1 < src.len && (src.ptr[i + 1] == 'x' || src.ptr[i + 1] == 'X')) {
		is_hex = true;
		i += 2;
		while (i < src.len && svsl_is_hex_digit(src.ptr[i])) i++;
	} else if (src.ptr[i] == '0' && i + 1 < src.len && (src.ptr[i + 1] == 'b' || src.ptr[i + 1] == 'B')) {
		is_bin = true;
		i += 2;
		while (i < src.len && (src.ptr[i] == '0' || src.ptr[i] == '1')) i++;
	} else {
		while (i < src.len && svsl_is_digit(src.ptr[i])) i++;
		if (i < src.len && src.ptr[i] == '.') {
			is_float = true;
			i++;
			while (i < src.len && svsl_is_digit(src.ptr[i])) i++;
		}
		if (i < src.len && (src.ptr[i] == 'e' || src.ptr[i] == 'E')) {
			int32_t exp = i + 1;
			if (exp < src.len && (src.ptr[exp] == '+' || src.ptr[exp] == '-')) exp++;
			if (exp < src.len && svsl_is_digit(src.ptr[exp])) {
				is_float = true;
				i = exp;
				while (i < src.len && svsl_is_digit(src.ptr[i])) i++;
			}
		}
	}
	int32_t digits_end = i;

	// suffix: trailing identifier characters
	while (i < src.len && svsl_is_ident_char(src.ptr[i])) i++;
	svsl_str_t suffix_text = svsl_str_slice(src, digits_end, i);
	svsl_str_t token_text  = svsl_str_slice(src, start, i);

	uint8_t suffix = svsl_suffix_none;
	bool    bad    = false;
	if      (suffix_text.len == 0)                       suffix = svsl_suffix_none;
	else if (svsl_str_eq_cstr(suffix_text, "u")  || svsl_str_eq_cstr(suffix_text, "U"))  suffix = svsl_suffix_u;
	else if (svsl_str_eq_cstr(suffix_text, "l")  || svsl_str_eq_cstr(suffix_text, "L"))  suffix = svsl_suffix_l;
	else if (svsl_str_eq_cstr(suffix_text, "ul") || svsl_str_eq_cstr(suffix_text, "uL") ||
	         svsl_str_eq_cstr(suffix_text, "Ul") || svsl_str_eq_cstr(suffix_text, "UL") ||
	         svsl_str_eq_cstr(suffix_text, "lu") || svsl_str_eq_cstr(suffix_text, "lU") ||
	         svsl_str_eq_cstr(suffix_text, "Lu") || svsl_str_eq_cstr(suffix_text, "LU")) suffix = svsl_suffix_ul;
	else if (svsl_str_eq_cstr(suffix_text, "f")  || svsl_str_eq_cstr(suffix_text, "F"))  suffix = svsl_suffix_f;
	else if (svsl_str_eq_cstr(suffix_text, "h")  || svsl_str_eq_cstr(suffix_text, "H"))  suffix = svsl_suffix_h;
	else if (svsl_str_eq_cstr(suffix_text, "lf") || svsl_str_eq_cstr(suffix_text, "LF")) suffix = svsl_suffix_lf;
	else bad = true;

	// float suffixes promote an integer-looking literal to float (HLSL allows 1f)
	if (suffix == svsl_suffix_f || suffix == svsl_suffix_h || suffix == svsl_suffix_lf) {
		if (is_hex || is_bin) bad = true;
		else                  is_float = true;
	}
	if (is_float && (suffix == svsl_suffix_u || suffix == svsl_suffix_l || suffix == svsl_suffix_ul))
		bad = true;

	if (bad) {
		svsl_diag_add(lex->arena, lex->diags, svsl_severity_error, loc,
		              "invalid literal '%.*s'", token_text.len, token_text.ptr);
		suffix = svsl_suffix_none;
	}

	svsl_token_t token = { .kind = svsl_tok_int_lit, .text = token_text, .loc = loc, .suffix = suffix };
	if (is_float) {
		token.kind        = svsl_tok_float_lit;
		token.float_value = strtod(src.ptr + start, NULL);
	} else if (is_hex) {
		token.int_value = strtoull(src.ptr + start + 2, NULL, 16);
	} else if (is_bin) {
		token.int_value = strtoull(src.ptr + start + 2, NULL, 2);
	} else if (token_text.len > 1 && token_text.ptr[0] == '0' && !is_float) {
		// octal (matching C and glslang) - reject a non-octal digit rather than
		// silently truncating at it (strtoull would parse "09" as just 0)
		for (int32_t k = start + 1; k < i; k++) {
			char c = src.ptr[k];
			if (c < '0' || c > '7') {
				if (c >= '8' && c <= '9')
					svsl_diag_add(lex->arena, lex->diags, svsl_severity_error, loc,
					              "invalid digit '%c' in octal literal '%.*s'", c, token_text.len, token_text.ptr);
				break; // '8'/'9' errored above; anything else is the suffix
			}
		}
		token.int_value = strtoull(src.ptr + start, NULL, 8);
	} else {
		token.int_value = strtoull(src.ptr + start, NULL, 10);
	}
	lex->pos = i;
	lex_push(lex, token);
}

bool svsl_lex(svsl_arena_t *arena, const svsl_pp_result_t *pp,
              svsl_token_list_t *out_tokens, svsl_diag_list_t *ref_diags) {
	lex_t lex = {
		.arena  = arena,
		.pp     = pp,
		.src    = { .ptr = pp->text, .len = pp->text_len },
		.tokens = out_tokens,
		.diags  = ref_diags };

	svsl_keyword_index_build(&lex.keywords);
	svsl_array_reserve(arena, out_tokens, out_tokens->count + lex.src.len / 4 + 16); // ~1 token per 4 bytes of source

	int32_t errors_before = ref_diags->error_count;
	while (lex.pos < lex.src.len) {
		char c = lex.src.ptr[lex.pos];

		if (c == '\n') {
			lex.pos++;
			lex.out_line++;
			lex.line_start = lex.pos;
			continue;
		}
		if (c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f') {
			lex.pos++;
			continue;
		}
		if (svsl_is_ident_start(c)) {
			int32_t start = lex.pos;
			while (lex.pos < lex.src.len && svsl_is_ident_char(lex.src.ptr[lex.pos])) lex.pos++;
			svsl_str_t text = svsl_str_slice(lex.src, start, lex.pos);
			lex_push(&lex, (svsl_token_t){
				.kind    = svsl_tok_ident,
				.keyword = (int16_t)svsl_keyword_find(&lex.keywords, text),
				.text    = text,
				.loc     = lex_loc(&lex, start) });
			continue;
		}
		if (svsl_is_digit(c) || (c == '.' && lex.pos + 1 < lex.src.len && svsl_is_digit(lex.src.ptr[lex.pos + 1]))) {
			lex_number(&lex);
			continue;
		}
		if (c == '"') {
			int32_t start = lex.pos;
			lex.pos++;
			int32_t content_start = lex.pos;
			while (lex.pos < lex.src.len && lex.src.ptr[lex.pos] != '"' && lex.src.ptr[lex.pos] != '\n') {
				if (lex.src.ptr[lex.pos] == '\\') lex.pos++;
				lex.pos++;
			}
			if (lex.pos >= lex.src.len || lex.src.ptr[lex.pos] != '"') {
				svsl_diag_add(arena, ref_diags, svsl_severity_error, lex_loc(&lex, start), "unterminated string literal");
				continue;
			}
			lex_push(&lex, (svsl_token_t){
				.kind = svsl_tok_string_lit,
				.text = svsl_str_slice(lex.src, content_start, lex.pos),
				.loc  = lex_loc(&lex, start) });
			lex.pos++; // closing quote
			continue;
		}

		bool matched = false;
		for (int32_t k = 0; k < 4 && (uint8_t)c < 128 && !matched; k++) {
			const char *text = punct_table[(uint8_t)c][k].text;
			int32_t     len  = punct_table[(uint8_t)c][k].len;
			if (len == 0) break;
			bool hit = lex.pos + len <= lex.src.len;
			for (int32_t j = 1; j < len && hit; j++) hit = lex.src.ptr[lex.pos + j] == text[j];
			if (hit) {
				lex_push(&lex, (svsl_token_t){
					.kind = punct_table[(uint8_t)c][k].tok,
					.text = svsl_str_slice(lex.src, lex.pos, lex.pos + len),
					.loc  = lex_loc(&lex, lex.pos) });
				lex.pos += len;
				matched = true;
			}
		}
		if (!matched) {
			svsl_diag_add(arena, ref_diags, svsl_severity_error, lex_loc(&lex, lex.pos),
			              "unexpected character '%c' (0x%02x)", c >= 32 && c < 127 ? c : '?', (uint8_t)c);
			lex.pos++;
		}
	}

	lex_push(&lex, (svsl_token_t){ .kind = svsl_tok_eof, .loc = lex_loc(&lex, lex.pos) });
	return ref_diags->error_count == errors_before;
}

#include "lexer.h"

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
} lex_t;

// Longest-match-first punctuator table.
static const struct { const char *text; svsl_tok_ tok; } punct_table[] = {
	{ "<<=", svsl_tok_shl_assign     },
	{ ">>=", svsl_tok_shr_assign     },
	{ "<<",  svsl_tok_shl            },
	{ ">>",  svsl_tok_shr            },
	{ "<=",  svsl_tok_le             },
	{ ">=",  svsl_tok_ge             },
	{ "==",  svsl_tok_eq             },
	{ "!=",  svsl_tok_neq            },
	{ "&&",  svsl_tok_andand         },
	{ "||",  svsl_tok_oror           },
	{ "++",  svsl_tok_plusplus       },
	{ "--",  svsl_tok_minusminus     },
	{ "+=",  svsl_tok_plus_assign    },
	{ "-=",  svsl_tok_minus_assign   },
	{ "*=",  svsl_tok_star_assign    },
	{ "/=",  svsl_tok_slash_assign   },
	{ "%=",  svsl_tok_percent_assign },
	{ "&=",  svsl_tok_and_assign     },
	{ "|=",  svsl_tok_or_assign      },
	{ "^=",  svsl_tok_xor_assign     },
	{ "::",  svsl_tok_coloncolon     },
	{ "(",   svsl_tok_lparen         },
	{ ")",   svsl_tok_rparen         },
	{ "[",   svsl_tok_lbracket       },
	{ "]",   svsl_tok_rbracket       },
	{ "{",   svsl_tok_lbrace         },
	{ "}",   svsl_tok_rbrace         },
	{ ",",   svsl_tok_comma          },
	{ ";",   svsl_tok_semicolon      },
	{ "$",   svsl_tok_dollar         },
	{ ".",   svsl_tok_dot            },
	{ "?",   svsl_tok_question       },
	{ ":",   svsl_tok_colon          },
	{ "+",   svsl_tok_plus           },
	{ "-",   svsl_tok_minus          },
	{ "*",   svsl_tok_star           },
	{ "/",   svsl_tok_slash          },
	{ "%",   svsl_tok_percent        },
	{ "=",   svsl_tok_assign         },
	{ "<",   svsl_tok_lt             },
	{ ">",   svsl_tok_gt             },
	{ "!",   svsl_tok_not            },
	{ "&",   svsl_tok_amp            },
	{ "|",   svsl_tok_pipe           },
	{ "^",   svsl_tok_caret          },
	{ "~",   svsl_tok_tilde          },
};

static bool is_ident_start(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static bool is_ident_char (char c) { return is_ident_start(c) || (c >= '0' && c <= '9'); }
static bool is_digit      (char c) { return c >= '0' && c <= '9'; }
static bool is_hex_digit  (char c) { return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

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
		while (i < src.len && is_hex_digit(src.ptr[i])) i++;
	} else if (src.ptr[i] == '0' && i + 1 < src.len && (src.ptr[i + 1] == 'b' || src.ptr[i + 1] == 'B')) {
		is_bin = true;
		i += 2;
		while (i < src.len && (src.ptr[i] == '0' || src.ptr[i] == '1')) i++;
	} else {
		while (i < src.len && is_digit(src.ptr[i])) i++;
		if (i < src.len && src.ptr[i] == '.') {
			is_float = true;
			i++;
			while (i < src.len && is_digit(src.ptr[i])) i++;
		}
		if (i < src.len && (src.ptr[i] == 'e' || src.ptr[i] == 'E')) {
			int32_t exp = i + 1;
			if (exp < src.len && (src.ptr[exp] == '+' || src.ptr[exp] == '-')) exp++;
			if (exp < src.len && is_digit(src.ptr[exp])) {
				is_float = true;
				i = exp;
				while (i < src.len && is_digit(src.ptr[i])) i++;
			}
		}
	}
	int32_t digits_end = i;

	// suffix: trailing identifier characters
	while (i < src.len && is_ident_char(src.ptr[i])) i++;
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
		// octal (matching C and glslang) — reject a non-octal digit rather than
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
		if (is_ident_start(c)) {
			int32_t start = lex.pos;
			while (lex.pos < lex.src.len && is_ident_char(lex.src.ptr[lex.pos])) lex.pos++;
			svsl_str_t text = svsl_str_slice(lex.src, start, lex.pos);
			lex_push(&lex, (svsl_token_t){
				.kind    = svsl_tok_ident,
				.keyword = (int16_t)svsl_keyword_lookup(text),
				.text    = text,
				.loc     = lex_loc(&lex, start) });
			continue;
		}
		if (is_digit(c) || (c == '.' && lex.pos + 1 < lex.src.len && is_digit(lex.src.ptr[lex.pos + 1]))) {
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
		for (int32_t k = 0; k < (int32_t)(sizeof(punct_table) / sizeof(punct_table[0])); k++) {
			int32_t len = (int32_t)strlen(punct_table[k].text);
			if (lex.pos + len <= lex.src.len && strncmp(lex.src.ptr + lex.pos, punct_table[k].text, (size_t)len) == 0) {
				lex_push(&lex, (svsl_token_t){
					.kind = punct_table[k].tok,
					.text = svsl_str_slice(lex.src, lex.pos, lex.pos + len),
					.loc  = lex_loc(&lex, lex.pos) });
				lex.pos += len;
				matched = true;
				break;
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

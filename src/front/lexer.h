// Lexer: preprocessed text → flat token array (one pass, one growing allocation).

#pragma once

#include "pp.h"
#include "../diag.h"
#include "../util/str.h"

#include <stdbool.h>
#include <stdint.h>

typedef enum svsl_tok_ {
	svsl_tok_eof = 0,
	svsl_tok_ident,     // keyword id carried alongside (svsl_kw_)
	svsl_tok_int_lit,
	svsl_tok_float_lit,
	svsl_tok_string_lit,

	svsl_tok_lparen, svsl_tok_rparen,
	svsl_tok_lbracket, svsl_tok_rbracket,
	svsl_tok_lbrace, svsl_tok_rbrace,
	svsl_tok_comma, svsl_tok_semicolon, svsl_tok_dot, svsl_tok_question,
	svsl_tok_colon, svsl_tok_coloncolon,
	svsl_tok_dollar, // '$' — value/type interpolation inside spirv_asm blocks

	svsl_tok_plus, svsl_tok_minus, svsl_tok_star, svsl_tok_slash, svsl_tok_percent,
	svsl_tok_plusplus, svsl_tok_minusminus,

	svsl_tok_assign,
	svsl_tok_plus_assign, svsl_tok_minus_assign, svsl_tok_star_assign,
	svsl_tok_slash_assign, svsl_tok_percent_assign,
	svsl_tok_and_assign, svsl_tok_or_assign, svsl_tok_xor_assign,
	svsl_tok_shl_assign, svsl_tok_shr_assign,

	svsl_tok_eq, svsl_tok_neq, svsl_tok_lt, svsl_tok_gt, svsl_tok_le, svsl_tok_ge,
	svsl_tok_andand, svsl_tok_oror, svsl_tok_not,
	svsl_tok_amp, svsl_tok_pipe, svsl_tok_caret, svsl_tok_tilde,
	svsl_tok_shl, svsl_tok_shr,
} svsl_tok_;

// Literal suffixes; sema uses these to pick the literal's type.
typedef enum svsl_suffix_ {
	svsl_suffix_none = 0,
	svsl_suffix_u,   // 42u        → uint32
	svsl_suffix_l,   // 42L        → int64
	svsl_suffix_ul,  // 42uL       → uint64
	svsl_suffix_f,   // 3.14f      → float32
	svsl_suffix_h,   // 3.14h      → half
	svsl_suffix_lf,  // 3.14lf     → float64
} svsl_suffix_;

typedef struct svsl_token_t {
	svsl_tok_    kind;
	int16_t      keyword;     // svsl_kw_, svsl_kw_none unless kind == ident
	uint8_t      suffix;      // svsl_suffix_
	svsl_str_t   text;        // slice into the preprocessed text (string_lit: without quotes)
	svsl_loc_t   loc;
	uint64_t     int_value;   // int_lit
	double       float_value; // float_lit
} svsl_token_t;

typedef struct svsl_token_list_t {
	svsl_token_t *items;
	int32_t       count;
	int32_t       capacity;
} svsl_token_list_t;

// Tokenizes pp->text. Always terminates the list with an eof token.
// Returns false if any error diagnostic was emitted.
bool svsl_lex(svsl_arena_t *arena, const svsl_pp_result_t *pp,
              svsl_token_list_t *out_tokens, svsl_diag_list_t *ref_diags);

#include "test.h"

#include "front/lexer.h"
#include "front/pp.h"
#include "tables/keywords.h"
#include "util/arena.h"

#include <string.h>

typedef struct lex_run_t {
	svsl_token_list_t tokens;
	svsl_diag_list_t  diags;
	bool              ok;
} lex_run_t;

static lex_run_t run_lex(svsl_arena_t *arena, const char *src) {
	lex_run_t r = {0};
	svsl_pp_result_t pp;
	svsl_pp_run(arena, src, "lex.hlsl", NULL, &pp, &r.diags);
	r.ok = svsl_lex(arena, &pp, &r.tokens, &r.diags);
	return r;
}

static void test_lex_idents_keywords(void) {
	svsl_arena_t arena = {0};

	lex_run_t r = run_lex(&arena, "float4 color = diffuse;\nif (x) return;\n");
	TEST_CHECK(r.ok);
	// float4 color = diffuse ; if ( x ) return ; eof
	TEST_CHECK(r.tokens.count == 12);
	TEST_CHECK(r.tokens.items[0].kind == svsl_tok_ident && svsl_str_eq_cstr(r.tokens.items[0].text, "float4"));
	TEST_CHECK(r.tokens.items[0].keyword == svsl_kw_none); // type names aren't keywords
	TEST_CHECK(r.tokens.items[2].kind == svsl_tok_assign);
	TEST_CHECK(r.tokens.items[5].keyword == svsl_kw_if);
	TEST_CHECK(r.tokens.items[9].keyword == svsl_kw_return);
	TEST_CHECK(r.tokens.items[11].kind == svsl_tok_eof);

	// context-sensitive words carry a keyword id but stay identifiers
	r = run_lex(&arena, "sample register uniform pack16 min16float");
	TEST_CHECK(r.tokens.items[0].keyword == svsl_kw_sample);
	TEST_CHECK(r.tokens.items[1].keyword == svsl_kw_register);
	TEST_CHECK(r.tokens.items[2].keyword == svsl_kw_uniform);
	TEST_CHECK(r.tokens.items[3].keyword == svsl_kw_pack16);
	TEST_CHECK(r.tokens.items[4].keyword == svsl_kw_none); // min16float is a type alias, not a keyword

	svsl_arena_free(&arena);
}

static void test_lex_int_literals(void) {
	svsl_arena_t arena = {0};

	lex_run_t r = run_lex(&arena, "42 42u 42L 42uL 0xFF 0xff 0b101 010 0");
	TEST_CHECK(r.ok);
	TEST_CHECK(r.tokens.items[0].kind == svsl_tok_int_lit && r.tokens.items[0].int_value == 42);
	TEST_CHECK(r.tokens.items[0].suffix == svsl_suffix_none);
	TEST_CHECK(r.tokens.items[1].int_value == 42  && r.tokens.items[1].suffix == svsl_suffix_u);
	TEST_CHECK(r.tokens.items[2].int_value == 42  && r.tokens.items[2].suffix == svsl_suffix_l);
	TEST_CHECK(r.tokens.items[3].int_value == 42  && r.tokens.items[3].suffix == svsl_suffix_ul);
	TEST_CHECK(r.tokens.items[4].int_value == 255 && r.tokens.items[4].suffix == svsl_suffix_none);
	TEST_CHECK(r.tokens.items[5].int_value == 255);
	TEST_CHECK(r.tokens.items[6].int_value == 5);
	TEST_CHECK(r.tokens.items[7].int_value == 8); // octal, matching C
	TEST_CHECK(r.tokens.items[8].int_value == 0);

	svsl_arena_free(&arena);
}

static void test_lex_float_literals(void) {
	svsl_arena_t arena = {0};

	lex_run_t r = run_lex(&arena, "3.14 3.14f .5 1. 2.5e3 1e-4 0.5h 0.0069h 3.14lf 1f");
	TEST_CHECK(r.ok);
	for (int32_t i = 0; i < 10; i++)
		TEST_CHECK(r.tokens.items[i].kind == svsl_tok_float_lit);
	TEST_CHECK(r.tokens.items[0].float_value == 3.14   && r.tokens.items[0].suffix == svsl_suffix_none);
	TEST_CHECK(r.tokens.items[1].float_value == 3.14   && r.tokens.items[1].suffix == svsl_suffix_f);
	TEST_CHECK(r.tokens.items[2].float_value == 0.5);
	TEST_CHECK(r.tokens.items[3].float_value == 1.0);
	TEST_CHECK(r.tokens.items[4].float_value == 2500.0);
	TEST_CHECK(r.tokens.items[5].float_value == 1e-4);
	TEST_CHECK(r.tokens.items[6].float_value == 0.5    && r.tokens.items[6].suffix == svsl_suffix_h);
	TEST_CHECK(r.tokens.items[7].float_value == 0.0069 && r.tokens.items[7].suffix == svsl_suffix_h);
	TEST_CHECK(r.tokens.items[8].float_value == 3.14   && r.tokens.items[8].suffix == svsl_suffix_lf);
	TEST_CHECK(r.tokens.items[9].float_value == 1.0    && r.tokens.items[9].suffix == svsl_suffix_f);

	// bad suffixes are errors
	r = run_lex(&arena, "3.14q");
	TEST_CHECK(!r.ok);
	r = run_lex(&arena, "0xFFh");
	TEST_CHECK(!r.ok);
	r = run_lex(&arena, "1.5u");
	TEST_CHECK(!r.ok);

	svsl_arena_free(&arena);
}

static void test_lex_operators(void) {
	svsl_arena_t arena = {0};

	lex_run_t r = run_lex(&arena, "a <<= b >> c <= d << e < f");
	TEST_CHECK(r.ok);
	TEST_CHECK(r.tokens.items[1].kind == svsl_tok_shl_assign);
	TEST_CHECK(r.tokens.items[3].kind == svsl_tok_shr);
	TEST_CHECK(r.tokens.items[5].kind == svsl_tok_le);
	TEST_CHECK(r.tokens.items[7].kind == svsl_tok_shl);
	TEST_CHECK(r.tokens.items[9].kind == svsl_tok_lt);

	r = run_lex(&arena, "x++ + ++y && a || b == c != d");
	TEST_CHECK(r.tokens.items[1].kind == svsl_tok_plusplus);
	TEST_CHECK(r.tokens.items[2].kind == svsl_tok_plus);
	TEST_CHECK(r.tokens.items[3].kind == svsl_tok_plusplus);
	TEST_CHECK(r.tokens.items[5].kind == svsl_tok_andand);
	TEST_CHECK(r.tokens.items[7].kind == svsl_tok_oror);
	TEST_CHECK(r.tokens.items[9].kind == svsl_tok_eq);
	TEST_CHECK(r.tokens.items[11].kind == svsl_tok_neq);

	// [[vk::binding(0, 1)]] — attribute punctuation incl. ::
	r = run_lex(&arena, "[[vk::binding(0, 1)]]");
	TEST_CHECK(r.ok);
	TEST_CHECK(r.tokens.items[0].kind == svsl_tok_lbracket);
	TEST_CHECK(r.tokens.items[1].kind == svsl_tok_lbracket);
	TEST_CHECK(r.tokens.items[2].kind == svsl_tok_ident && svsl_str_eq_cstr(r.tokens.items[2].text, "vk"));
	TEST_CHECK(r.tokens.items[3].kind == svsl_tok_coloncolon);
	TEST_CHECK(r.tokens.items[10].kind == svsl_tok_rbracket);
	TEST_CHECK(r.tokens.items[11].kind == svsl_tok_rbracket);

	// swizzles and matrix element access are dot + ident
	r = run_lex(&arena, "v.xyz m._m00 m._11");
	TEST_CHECK(r.tokens.items[1].kind == svsl_tok_dot);
	TEST_CHECK(svsl_str_eq_cstr(r.tokens.items[2].text, "xyz"));
	TEST_CHECK(r.tokens.items[4].kind == svsl_tok_dot);
	TEST_CHECK(svsl_str_eq_cstr(r.tokens.items[5].text, "_m00"));
	TEST_CHECK(r.tokens.items[7].kind == svsl_tok_dot);
	TEST_CHECK(svsl_str_eq_cstr(r.tokens.items[8].text, "_11"));

	svsl_arena_free(&arena);
}

static void test_lex_strings_and_locs(void) {
	svsl_arena_t arena = {0};

	lex_run_t r = run_lex(&arena, "[[vk::image_format(\"rgba8\")]]");
	TEST_CHECK(r.ok);
	TEST_CHECK(r.tokens.items[6].kind == svsl_tok_string_lit);
	TEST_CHECK(svsl_str_eq_cstr(r.tokens.items[6].text, "rgba8"));

	r = run_lex(&arena, "\"unterminated\n");
	TEST_CHECK(!r.ok);

	// locations: line comes from the pp line map, column from the output line
	r = run_lex(&arena, "one\n  two three\n");
	TEST_CHECK(r.tokens.items[0].loc.line == 1 && r.tokens.items[0].loc.col == 1);
	TEST_CHECK(r.tokens.items[1].loc.line == 2 && r.tokens.items[1].loc.col == 3);
	TEST_CHECK(r.tokens.items[2].loc.line == 2 && r.tokens.items[2].loc.col == 7);
	TEST_CHECK(strcmp(r.tokens.items[0].loc.file, "lex.hlsl") == 0);

	// unexpected characters are errors but lexing continues
	r = run_lex(&arena, "a @ b\n");
	TEST_CHECK(!r.ok);
	TEST_CHECK(r.tokens.count == 3); // a b eof

	svsl_arena_free(&arena);
}

void test_lexer(void) {
	test_lex_idents_keywords();
	test_lex_int_literals();
	test_lex_float_literals();
	test_lex_operators();
	test_lex_strings_and_locs();
}

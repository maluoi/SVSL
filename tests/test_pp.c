#include "test.h"

#include "diag.h"
#include "front/pp.h"
#include "util/arena.h"

#include <string.h>

typedef struct test_inc_t {
	const char *name;
	const char *content;
} test_inc_t;

static svsl_include_src_t test_include(void *user, const char *path, const char *requester) {
	(void)requester;
	const test_inc_t *table = user;
	for (int32_t i = 0; table[i].name; i++)
		if (strcmp(table[i].name, path) == 0)
			return (svsl_include_src_t){ .content = table[i].content, .length = -1 };
	return (svsl_include_src_t){0};
}

typedef struct pp_run_t {
	svsl_pp_result_t result;
	svsl_diag_list_t diags;
	bool             ok;
} pp_run_t;

static pp_run_t run_pp(svsl_arena_t *arena, const char *src, const test_inc_t *opt_incs) {
	pp_run_t r = {0};
	svsl_pp_options_t opt = {
		.include_cb   = opt_incs ? test_include : NULL,
		.include_user = (void *)opt_incs };
	r.ok = svsl_pp_run(arena, src, "test.hlsl", &opt, &r.result, &r.diags);
	return r;
}

static bool text_is(const pp_run_t *r, const char *expected) {
	if (strcmp(r->result.text, expected) == 0) return true;
	printf("  pp output mismatch:\n  --- got ---\n%s  --- expected ---\n%s  ---\n", r->result.text, expected);
	return false;
}

static void test_pp_basics(void) {
	svsl_arena_t arena = {0};

	// passthrough + comment stripping
	pp_run_t r = run_pp(&arena, "int a; // trailing\nfloat b/*mid*/c;\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "int a; \nfloat b c;\n"));

	// object macros, nesting, #undef
	r = run_pp(&arena,
		"#define A B\n"
		"#define B 7\n"
		"x = A;\n"
		"#undef B\n"
		"y = A;\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "x = 7;\ny = B;\n"));

	// self-referential macro doesn't loop
	r = run_pp(&arena, "#define R R+1\nx = R;\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "x = R+1;\n"));

	// empty #define works with #ifdef and expands to nothing
	r = run_pp(&arena, "#define FLAG\n#ifdef FLAG\nFLAG on;\n#endif\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, " on;\n"));

	// macros don't expand inside identifiers or numbers
	r = run_pp(&arena, "#define e 9\nx = 1e5 + e + he;\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "x = 1e5 + 9 + he;\n"));

	// backslash continuation forms one logical line
	r = run_pp(&arena, "a\\\nb\nc\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "ab\nc\n"));
	TEST_CHECK(r.result.line_count == 2);
	TEST_CHECK(r.result.lines[0].line == 1);
	TEST_CHECK(r.result.lines[1].line == 3);

	svsl_arena_free(&arena);
}

static void test_pp_function_macros(void) {
	svsl_arena_t arena = {0};

	pp_run_t r = run_pp(&arena,
		"#define MUL(a, b) ((a) * (b))\n"
		"x = MUL(1 + 2, f(3, 4));\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "x = ((1 + 2) * (f(3, 4)));\n"));

	// arguments expand before substitution
	r = run_pp(&arena,
		"#define VAL 5\n"
		"#define ID(x) x\n"
		"y = ID(VAL);\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "y = 5;\n"));

	// function-like macro without parens is not an invocation
	r = run_pp(&arena, "#define F(x) x\nptr = F;\nz = F(1);\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "ptr = F;\nz = 1;\n"));

	// nested invocations
	r = run_pp(&arena,
		"#define ADD(a, b) (a + b)\n"
		"w = ADD(ADD(1, 2), 3);\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "w = ((1 + 2) + 3);\n"));

	// wrong argument count is an error
	r = run_pp(&arena, "#define TWO(a, b) a b\nx = TWO(1);\n", NULL);
	TEST_CHECK(!r.ok);
	TEST_CHECK(r.diags.error_count == 1);

	svsl_arena_free(&arena);
}

static void test_pp_conditionals(void) {
	svsl_arena_t arena = {0};

	pp_run_t r = run_pp(&arena,
		"#define FOO 2\n"
		"#if FOO == 2\na\n"
		"#elif FOO == 3\nb\n"
		"#else\nc\n"
		"#endif\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "a\n"));

	r = run_pp(&arena,
		"#define FOO 3\n"
		"#if FOO == 2\na\n"
		"#elif FOO == 3\nb\n"
		"#else\nc\n"
		"#endif\n", NULL);
	TEST_CHECK(text_is(&r, "b\n"));

	r = run_pp(&arena,
		"#if 0\na\n"
		"#elif 0\nb\n"
		"#else\nc\n"
		"#endif\n", NULL);
	TEST_CHECK(text_is(&r, "c\n"));

	// defined() in both spellings, logic operators
	r = run_pp(&arena,
		"#define X\n"
		"#if defined(X) && !defined Y\nyes\n#endif\n"
		"#if defined(Y) || 0\nno\n#endif\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "yes\n"));

	// macros expand in #if expressions; arithmetic and comparison
	r = run_pp(&arena,
		"#define N 4\n"
		"#if N * 2 + 1 == 9 && (N >> 1) == 2\nok\n#endif\n", NULL);
	TEST_CHECK(text_is(&r, "ok\n"));

	// ternary
	r = run_pp(&arena, "#if (1 ? 5 : 6) == 5\nt\n#endif\n", NULL);
	TEST_CHECK(text_is(&r, "t\n"));

	// nested conditionals inside inactive regions stay balanced
	r = run_pp(&arena,
		"#if 0\n"
		"#ifdef NEVER\nx\n#else\ny\n#endif\n"
		"#endif\n"
		"after\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "after\n"));

	// inactive regions don't define macros or include files
	r = run_pp(&arena,
		"#if 0\n"
		"#define HIDDEN 1\n"
		"#include \"nonexistent.h\"\n"
		"#endif\n"
		"#ifdef HIDDEN\nbad\n#else\ngood\n#endif\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "good\n"));

	// ifndef
	r = run_pp(&arena, "#ifndef NOPE\nu\n#endif\n", NULL);
	TEST_CHECK(text_is(&r, "u\n"));

	svsl_arena_free(&arena);
}

static void test_pp_errors(void) {
	svsl_arena_t arena = {0};

	pp_run_t r = run_pp(&arena, "#if 1\nx\n", NULL);        // unterminated #if
	TEST_CHECK(!r.ok);
	r = run_pp(&arena, "#endif\n", NULL);                    // unmatched #endif
	TEST_CHECK(!r.ok);
	r = run_pp(&arena, "#if 1\n#else\n#elif 2\n#endif\n", NULL); // #elif after #else
	TEST_CHECK(!r.ok);
	r = run_pp(&arena, "#bogus\n", NULL);                    // unknown directive
	TEST_CHECK(!r.ok);
	r = run_pp(&arena, "#if 1 / 0\n#endif\n", NULL);         // division by zero
	TEST_CHECK(!r.ok);
	r = run_pp(&arena, "#error custom text\n", NULL);        // #error
	TEST_CHECK(!r.ok && strstr(r.diags.items[0].message, "custom text") != NULL);
	r = run_pp(&arena, "#include \"missing.h\"\n", NULL);    // missing include
	TEST_CHECK(!r.ok);
	r = run_pp(&arena, "/* never closed\n", NULL);           // unterminated block comment
	TEST_CHECK(!r.ok);

	// errors carry locations
	r = run_pp(&arena, "ok line\n#bogus\n", NULL);
	TEST_CHECK(r.diags.count == 1);
	TEST_CHECK(r.diags.items[0].loc.line == 2);
	TEST_CHECK(strcmp(r.diags.items[0].loc.file, "test.hlsl") == 0);

	svsl_arena_free(&arena);
}

static void test_pp_includes(void) {
	svsl_arena_t arena = {0};

	const test_inc_t incs[] = {
		{ "inc.h",     "int from_inc;\n" },
		{ "guarded.h", "#ifndef GUARD_H\n#define GUARD_H\nint guarded;\n#endif\n" },
		{ "once.h",    "#pragma once\nint once;\n" },
		{ "outer.h",   "#include \"inner.h\"\nint outer;\n" },
		{ "inner.h",   "int inner;\n" },
		{ "meta.h",    "//--sneaky = value\nint has_meta;\n" },
		{ NULL, NULL },
	};

	// quotes and angle brackets resolve identically
	pp_run_t r = run_pp(&arena, "#include \"inc.h\"\na;\n#include <inc.h>\n", incs);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "int from_inc;\na;\nint from_inc;\n"));

	// line map spans files correctly
	TEST_CHECK(r.result.line_count == 3);
	TEST_CHECK(strcmp(r.result.lines[0].file, "inc.h") == 0 && r.result.lines[0].line == 1);
	TEST_CHECK(strcmp(r.result.lines[1].file, "test.hlsl") == 0 && r.result.lines[1].line == 2);
	TEST_CHECK(strcmp(r.result.lines[2].file, "inc.h") == 0 && r.result.lines[2].line == 1);

	// include guards
	r = run_pp(&arena, "#include \"guarded.h\"\n#include \"guarded.h\"\n", incs);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "int guarded;\n"));

	// #pragma once
	r = run_pp(&arena, "#include \"once.h\"\n#include \"once.h\"\n", incs);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "int once;\n"));

	// nested includes
	r = run_pp(&arena, "#include \"outer.h\"\n", incs);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "int inner;\nint outer;\n"));

	// metadata in included files is ignored (matches skshaderc)
	r = run_pp(&arena, "//--mine = 1\n#include \"meta.h\"\n", incs);
	TEST_CHECK(r.ok);
	TEST_CHECK(r.result.meta_count == 1);
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[0].name, "mine"));

	// defines cross include boundaries (stereokit.hlsli pattern)
	const test_inc_t def_incs[] = {
		{ "defs.h", "#define SK_MAX_VIEWS 6\n" },
		{ NULL, NULL },
	};
	r = run_pp(&arena, "#include \"defs.h\"\nfloat4 v[SK_MAX_VIEWS];\n", def_incs);
	TEST_CHECK(r.ok);
	TEST_CHECK(text_is(&r, "float4 v[6];\n"));

	svsl_arena_free(&arena);
}

static void test_pp_metadata(void) {
	svsl_arena_t arena = {0};

	pp_run_t r = run_pp(&arena,
		"//--name = sk/unlit\n"
		"//--color:color = 1,1,1,1\n"
		"//--uv_scale: range(0, 2) = 0.5\n"
		"//--diffuse = white\n"
		"//--flag\n"
		"/*--block = inside*/\n"
		"/*\n"
		"--multi = line\n"
		"*/\n"
		"// -- spaced = ok\n"
		"//not_meta = 1\n"
		"code; //--trailing = 3\n", NULL);
	TEST_CHECK(r.ok);
	TEST_CHECK(r.result.meta_count == 9);

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[0].name,  "name"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[0].value, "sk/unlit"));
	TEST_CHECK(r.result.metas[0].tag.len == 0);
	TEST_CHECK(r.result.metas[0].loc.line == 1);

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[1].name,  "color"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[1].tag,   "color"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[1].value, "1,1,1,1"));

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[2].name,  "uv_scale"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[2].tag,   "range(0, 2)"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[2].value, "0.5"));

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[3].name,  "diffuse"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[3].value, "white"));

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[4].name, "flag"));
	TEST_CHECK(r.result.metas[4].value.len == 0);

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[5].name,  "block"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[5].value, "inside"));

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[6].name,  "multi"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[6].value, "line"));
	TEST_CHECK(r.result.metas[6].loc.line == 8);

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[7].name,  "spaced")); // whitespace before -- is fine
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[7].value, "ok"));

	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[8].name,  "trailing"));
	TEST_CHECK(svsl_str_eq_cstr(r.result.metas[8].value, "3"));

	svsl_arena_free(&arena);
}

static void test_pp_predefines(void) {
	svsl_arena_t arena = {0};

	const svsl_define_t defines[] = {
		{ "SK_OPENGL", NULL },
		{ "LEVEL",     "3"  },
	};
	svsl_pp_options_t opt = { .defines = defines, .define_count = 2 };
	svsl_pp_result_t  result;
	svsl_diag_list_t  diags = {0};
	bool ok = svsl_pp_run(&arena, "#ifdef SK_OPENGL\ngl\n#endif\nx = LEVEL;\n", "t", &opt, &result, &diags);
	TEST_CHECK(ok);
	TEST_CHECK(strcmp(result.text, "gl\nx = 3;\n") == 0);

	svsl_arena_free(&arena);
}

void test_pp(void) {
	test_pp_basics();
	test_pp_function_macros();
	test_pp_conditionals();
	test_pp_errors();
	test_pp_includes();
	test_pp_metadata();
	test_pp_predefines();
}

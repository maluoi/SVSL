#include "test.h"

#include "front/ast.h"
#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "util/arena.h"

#include <string.h>

typedef struct parse_run_t {
	svsl_ast_t      *ast;
	svsl_diag_list_t diags;
	bool             ok;
} parse_run_t;

static parse_run_t run_parse(svsl_arena_t *arena, const char *src) {
	parse_run_t r = {0};
	svsl_pp_result_t  pp;
	svsl_token_list_t tokens = {0};
	svsl_pp_run(arena, src, "parse.hlsl", NULL, &pp, &r.diags);
	svsl_lex(arena, &pp, &tokens, &r.diags);
	r.ast = svsl_parse(arena, &tokens, &r.diags);
	r.ok  = r.diags.error_count == 0;
	return r;
}

// parse + dump, compare against expected snapshot
static bool dump_is(svsl_arena_t *arena, const char *src, const char *expected) {
	parse_run_t r = run_parse(arena, src);
	if (!r.ok) {
		for (int32_t i = 0; i < r.diags.count; i++)
			if (r.diags.items[i].severity == svsl_severity_error)
				printf("  parse error %s:%d: %s\n", r.diags.items[i].loc.file,
				       r.diags.items[i].loc.line, r.diags.items[i].message);
		return false;
	}
	const char *dump = svsl_ast_dump(arena, r.ast);
	if (strcmp(dump, expected) == 0) return true;
	printf("  dump mismatch:\n  --- got ---\n%s  --- expected ---\n%s  ---\n", dump, expected);
	return false;
}

static void test_parse_exprs(void) {
	svsl_arena_t arena = {0};

	// precedence and associativity
	TEST_CHECK(dump_is(&arena,
		"float f() { return 1 + 2 * 3; }",
		"(func f (ret float) (block (return (+ 1 (* 2 3)))))\n"));
	TEST_CHECK(dump_is(&arena,
		"float f() { return a = b = c; }",
		"(func f (ret float) (block (return (= a (= b c)))))\n"));
	TEST_CHECK(dump_is(&arena,
		"float f() { return a < b && c || d; }",
		"(func f (ret float) (block (return (|| (&& (< a b) c) d))))\n"));
	TEST_CHECK(dump_is(&arena,
		"float f() { return a ? b : c ? d : e; }",
		"(func f (ret float) (block (return (?: a b (?: c d e)))))\n"));
	TEST_CHECK(dump_is(&arena,
		"float f() { return -a * !b; }",
		"(func f (ret float) (block (return (* (- a) (! b)))))\n"));

	// postfix chains: swizzles, indexing, methods, calls
	TEST_CHECK(dump_is(&arena,
		"float f() { return sk_inst[ids.inst].world; }",
		"(func f (ret float) (block (return (. ([] sk_inst (. ids inst)) world))))\n"));
	TEST_CHECK(dump_is(&arena,
		"float f() { return diffuse.Sample(diffuse_s, input.uv); }",
		"(func f (ret float) (block (return (call (. diffuse Sample) diffuse_s (. input uv)))))\n"));
	TEST_CHECK(dump_is(&arena,
		"float f() { return mul(float4(input.pos.xyz, 1), m); }",
		"(func f (ret float) (block (return (call mul (ctor float4 (. (. input pos) xyz) 1) m))))\n"));

	// casts vs parens
	TEST_CHECK(dump_is(&arena,
		"float f() { return (float3x3)world * v; }",
		"(func f (ret float) (block (return (* (cast float3x3 world) v))))\n"));
	TEST_CHECK(dump_is(&arena,
		"float f() { return (a) * v; }",
		"(func f (ret float) (block (return (* a v))))\n"));

	// increment / compound assign
	TEST_CHECK(dump_is(&arena,
		"void f() { i++; ++i; x += 2; }",
		"(func f (ret void) (block (expr (post++ i)) (expr (++ i)) (expr (+= x 2))))\n"));

	svsl_arena_free(&arena);
}

static void test_parse_decls(void) {
	svsl_arena_t arena = {0};

	// struct with semantics
	TEST_CHECK(dump_is(&arena,
		"struct psIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };",
		"(struct psIn (member pos float4 :SV_POSITION) (member uv float2 :TEXCOORD0))\n"));

	// cbuffer with register and array member
	TEST_CHECK(dump_is(&arena,
		"cbuffer stereokit_buffer : register(b1) { float4x4 sk_view[6]; uint sk_view_count; };",
		"(cbuffer stereokit_buffer :register(b1,space0) (member sk_view float4x4[6]) (member sk_view_count uint))\n"));

	// resources: pairing conventions, templates, registers
	TEST_CHECK(dump_is(&arena,
		"Texture2D diffuse : register(t0);\nSamplerState diffuse_s : register(s0);",
		"(global diffuse Texture2D :register(t0,space0))\n(global diffuse_s SamplerState :register(s0,space0))\n"));
	TEST_CHECK(dump_is(&arena,
		"StructuredBuffer<inst_t> sk_inst : register(t12);",
		"(global sk_inst StructuredBuffer<inst_t> :register(t12,space0))\n"));

	// bare globals with initializers ($Globals material params)
	TEST_CHECK(dump_is(&arena,
		"float4 color = {1,1,1,1};\nfloat4 tex_trans = float4(0,0,1,1);",
		"(global color float4 = (init 1 1 1 1))\n(global tex_trans float4 = (ctor float4 0 0 1 1))\n"));

	// multiple declarators split into separate globals
	TEST_CHECK(dump_is(&arena,
		"static const float a = 1, b = 2;",
		"(global a static const float = 1)\n(global b static const float = 2)\n"));

	// groupshared multi-dim array
	TEST_CHECK(dump_is(&arena,
		"groupshared float3 tile[9][64];",
		"(global tile workgroup float3[9][64])\n"));

	// SVSL-native forms
	TEST_CHECK(dump_is(&arena,
		"uniform SceneData : register(b0, space0) { float4x4 viewProj; };",
		"(uniform SceneData :register(b0,space0) (member viewProj float4x4))\n"));
	TEST_CHECK(dump_is(&arena,
		"storagebuffer readonly Vertices : register(t0, space1) { Vertex vertices[]; };",
		"(storagebuffer Vertices readonly :register(t0,space1) (member vertices Vertex[]))\n"));
	TEST_CHECK(dump_is(&arena,
		"pushconstant Draw { float4x4 model; uint32 materialId; };",
		"(pushconstant Draw (member model float4x4) (member materialId uint32))\n"));
	TEST_CHECK(dump_is(&arena,
		"storagebuffer pack16 Data { float4 v[]; };",
		"(storagebuffer Data pack16 (member v float4[]))\n"));
	TEST_CHECK(dump_is(&arena,
		"specialization const uint32 TILE = 16;",
		"(global TILE spec const uint32 = 16)\n"));

	// struct inheritance is a clear error, and parsing still recovers
	{
		parse_run_t r = run_parse(&arena, "struct psIn : base_t { float4 pos : SV_POSITION; };");
		TEST_CHECK(!r.ok);
		TEST_CHECK(r.diags.error_count == 1);
		TEST_CHECK(strstr(r.diags.items[0].message, "inheritance") != NULL);
		TEST_CHECK(r.ast->decl_count == 1 && r.ast->decls[0]->struct_decl.member_count == 1);
	}
	TEST_CHECK(dump_is(&arena,
		"Texture2D tex : register(0, 1);",
		"(global tex Texture2D :register(0,1))\n"));

	// image types with format, subpass inputs
	TEST_CHECK(dump_is(&arena,
		"Image2D<float4, rgba8> img;",
		"(global img Image2D<float4,rgba8>)\n"));
	TEST_CHECK(dump_is(&arena,
		"SubpassInput<float4, 1> src;",
		"(global src SubpassInput<float4,1>)\n"));
	TEST_CHECK(dump_is(&arena,
		"RWTexture2D<float4> dest : register(u0);",
		"(global dest RWTexture2D<float4> :register(u0,space0))\n"));

	svsl_arena_free(&arena);
}

static void test_parse_functions(void) {
	svsl_arena_t arena = {0};

	// entry points with attributes, params with semantics
	TEST_CHECK(dump_is(&arena,
		"[numthreads(8, 8, 1)]\nvoid cs(uint3 id : SV_DispatchThreadID) { }",
		"(func cs [numthreads 8 8 1](ret void) (param id uint3 :SV_DispatchThreadID) (block))\n"));
	TEST_CHECK(dump_is(&arena,
		"[vertex] VSOut my_vs(VSIn input, sk_ids_t ids) { VSOut o; return o; }",
		"(func my_vs [vertex](ret VSOut) (param input VSIn) (param ids sk_ids_t) "
		"(block (decl (var o VSOut)) (return o)))\n"));
	TEST_CHECK(dump_is(&arena,
		"min16float4 ps(psIn input) : SV_TARGET { return input.color; }",
		"(func ps (ret min16float4 :SV_TARGET) (param input psIn) (block (return (. input color))))\n"));

	// inout params, resource params, defaults
	TEST_CHECK(dump_is(&arena,
		"void f(inout float3 accum, Texture2D t, Sampler s) { }",
		"(func f (ret void) (param accum inout float3) (param t Texture2D) (param s Sampler) (block))\n"));

	// [[vk::*]] attributes
	TEST_CHECK(dump_is(&arena,
		"[[vk::image_format(\"rgba8\")]] RWTexture2D<float4> tex : register(u0);",
		"(global tex [vk::image_format \"rgba8\"]RWTexture2D<float4> :register(u0,space0))\n"));

	// wave_size + compute
	TEST_CHECK(dump_is(&arena,
		"[wave_size(32)] [compute(64, 1, 1)] void my_cs() { }",
		"(func my_cs [wave_size 32][compute 64 1 1](ret void) (block))\n"));

	svsl_arena_free(&arena);
}

static void test_parse_stmts(void) {
	svsl_arena_t arena = {0};

	TEST_CHECK(dump_is(&arena,
		"void f() { if (x > 0) y = 1; else y = 2; }",
		"(func f (ret void) (block (if (> x 0) (expr (= y 1)) (expr (= y 2)))))\n"));
	TEST_CHECK(dump_is(&arena,
		"void f() { [unroll] for (int i = 0; i < 4; i++) total += i; }",
		"(func f (ret void) (block [unroll](for (decl (var i int = 0)) (< i 4) (post++ i) (expr (+= total i)))))\n"));
	TEST_CHECK(dump_is(&arena,
		"void f() { while (count < 5) count = count + 1; }",
		"(func f (ret void) (block (while (< count 5) (expr (= count (+ count 1))))))\n"));
	TEST_CHECK(dump_is(&arena,
		"void f() { do { x--; } while (x > 0); }",
		"(func f (ret void) (block (do (block (expr (post-- x))) (> x 0))))\n"));
	TEST_CHECK(dump_is(&arena,
		"void f() { switch (mode) { case 0: a = 1; break; case 1: case 2: b = 2; break; default: break; } }",
		"(func f (ret void) (block (switch mode"
		" (case 0 (expr (= a 1)) (break)) (case 1) (case 2 (expr (= b 2)) (break)) (default (break)))))\n"));
	TEST_CHECK(dump_is(&arena,
		"void f() { if (alpha < 0.5) discard; demote; }",
		"(func f (ret void) (block (if (< alpha 0.5) (discard)) (demote)))\n"));
	TEST_CHECK(dump_is(&arena,
		"void f() { for (i = 0, j = 8; i < j; i++, j--) { } }",
		"(func f (ret void) (block (for (expr (, (= i 0) (= j 8))) (< i j) (, (post++ i) (post-- j)) (block))))\n"));

	svsl_arena_free(&arena);
}

static void test_parse_recovery(void) {
	svsl_arena_t arena = {0};

	// an error in one function doesn't hide later declarations
	parse_run_t r = run_parse(&arena, "void broken() { x = ; }\nfloat4 fine() { return 1; }");
	TEST_CHECK(!r.ok);
	TEST_CHECK(r.diags.error_count >= 1);
	TEST_CHECK(r.ast->decl_count == 2);
	TEST_CHECK(r.ast->decls[1]->kind == svsl_decl_func);
	TEST_CHECK(svsl_str_eq_cstr(r.ast->decls[1]->func.name, "fine"));

	// garbage at top level recovers to the next declaration
	r = run_parse(&arena, "$$$;\nstruct S { float x; };");
	TEST_CHECK(!r.ok);
	bool found_struct = false;
	for (int32_t i = 0; i < r.ast->decl_count; i++)
		if (r.ast->decls[i]->kind == svsl_decl_struct) found_struct = true;
	TEST_CHECK(found_struct);

	// forward reference to a struct defined later (two-pass scan)
	r = run_parse(&arena, "psIn helper() { psIn o; return o; }\nstruct psIn { float4 pos; };");
	TEST_CHECK(r.ok);

	svsl_arena_free(&arena);
}

static void test_parse_reference_shader(void) {
	svsl_arena_t arena = {0};

	// the StereoKit reference shader shape, minus the include
	parse_run_t r = run_parse(&arena,
		"//--name = sk/unlit\n"
		"float4 color = {1,1,1,1};\n"
		"float4 tex_trans = {0,0,1,1};\n"
		"Texture2D diffuse : register(t0);\n"
		"SamplerState diffuse_s : register(s0);\n"
		"struct vsIn { float4 pos:SV_Position; float3 norm:NORMAL0; float2 uv:TEXCOORD0; float4 col:COLOR0; };\n"
		"struct psIn { float4 pos:SV_POSITION; float2 uv:TEXCOORD0; min16float4 color:COLOR0; };\n"
		"struct sk_ids_t { uint inst : SV_InstanceID; uint view : SV_ViewID; };\n"
		"struct inst_t { float4x4 world; float4 color; };\n"
		"StructuredBuffer<inst_t> sk_inst : register(t12);\n"
		"cbuffer stereokit_buffer : register(b1) { float4x4 sk_viewproj[6]; };\n"
		"psIn vs(vsIn input, sk_ids_t ids) {\n"
		"	psIn o;\n"
		"	float4 world = mul(float4(input.pos.xyz, 1), sk_inst[ids.inst].world);\n"
		"	o.pos = mul(world, sk_viewproj[ids.view]);\n"
		"	o.uv = (input.uv * tex_trans.zw) + tex_trans.xy;\n"
		"	o.color = input.col * color * sk_inst[ids.inst].color;\n"
		"	return o;\n"
		"}\n"
		"min16float4 ps(psIn input) : SV_TARGET {\n"
		"	return diffuse.Sample(diffuse_s, input.uv) * input.color;\n"
		"}\n");
	TEST_CHECK(r.ok);
	TEST_CHECK(r.ast->decl_count == 12);

	svsl_arena_free(&arena);
}

static void test_parse_depth_limits(void) {
	svsl_arena_t arena = {0};
	char         src[64 * 1024];

	// pathological nesting must produce an error, not a stack overflow
	// (parser recursion for parens/blocks; sema's tree walk for long chains)
	int32_t n   = 2000;
	int32_t len = snprintf(src, sizeof(src), "float f() { return ");
	for (int32_t i = 0; i < n; i++) src[len++] = '(';
	src[len++] = '1';
	for (int32_t i = 0; i < n; i++) src[len++] = ')';
	len += snprintf(src + len, sizeof(src) - (size_t)len, "; }");
	parse_run_t r = run_parse(&arena, src);
	TEST_CHECK(!r.ok);

	len = snprintf(src, sizeof(src), "float f() { float x = 1");
	for (int32_t i = 0; i < 2000; i++) { src[len++] = '+'; src[len++] = '1'; }
	len += snprintf(src + len, sizeof(src) - (size_t)len, "; return x; }");
	r = run_parse(&arena, src);
	TEST_CHECK(!r.ok);

	len = snprintf(src, sizeof(src), "float f() { ");
	for (int32_t i = 0; i < 2000; i++) src[len++] = '{';
	for (int32_t i = 0; i < 2000; i++) src[len++] = '}';
	len += snprintf(src + len, sizeof(src) - (size_t)len, " return 1; }");
	r = run_parse(&arena, src);
	TEST_CHECK(!r.ok);

	// well under the limits stays clean
	r = run_parse(&arena, "float f() { return ((((((((1)))))))) + 1 + 2 + 3; }");
	TEST_CHECK(r.ok);

	svsl_arena_free(&arena);
}

void test_parser(void) {
	test_parse_exprs();
	test_parse_decls();
	test_parse_functions();
	test_parse_stmts();
	test_parse_recovery();
	test_parse_reference_shader();
	test_parse_depth_limits();
}

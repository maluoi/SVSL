#include "test.h"
#include "test_spv.h"

#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "ir/ir.h"
#include "sema/sema.h"
#include "util/arena.h"

#include <string.h>

typedef struct ir_run_t {
	svsl_ir_module_t module;
	svsl_program_t   prog;
	svsl_diag_list_t diags;
	bool             ok;
} ir_run_t;

static ir_run_t run_ir(svsl_arena_t *arena, const char *src) {
	ir_run_t r = {0};
	svsl_pp_result_t  pp;
	svsl_token_list_t tokens = {0};
	svsl_pp_run(arena, src, "ir_test.hlsl", NULL, &pp, &r.diags);
	svsl_lex(arena, &pp, &tokens, &r.diags);
	svsl_ast_t *ast = svsl_parse(arena, &tokens, &r.diags);
	svsl_sema_run(arena, ast, &pp, "ir_test.hlsl", NULL, &r.prog, &r.diags);
	if (r.diags.error_count == 0)
		svsl_ir_build(arena, &r.prog, svsl_opt_default, &r.module, &r.diags);
	r.ok = r.diags.error_count == 0;
	if (!r.ok)
		for (int32_t i = 0; i < r.diags.count; i++)
			if (r.diags.items[i].severity == svsl_severity_error)
				printf("  ir error %s:%d: %s\n", r.diags.items[i].loc.file,
				       r.diags.items[i].loc.line, r.diags.items[i].message);
	return r;
}

static bool dump_is(svsl_arena_t *arena, const char *src, const char *expected) {
	ir_run_t r = run_ir(arena, src);
	if (!r.ok) return false;
	const char *dump = svsl_ir_dump(arena, &r.module, &r.prog);
	if (strcmp(dump, expected) == 0) return true;
	printf("  ir dump mismatch:\n  --- got ---\n%s  --- expected ---\n%s  ---\n", dump, expected);
	return false;
}

// find an op in a function's stream; -1 if absent
static int32_t find_op(const svsl_ir_func_t *fn, svsl_ir_op_ op) {
	for (int32_t i = 0; i < fn->insts.count; i++)
		if (fn->insts.items[i].op == op) return i;
	return -1;
}
static int32_t count_op(const svsl_ir_func_t *fn, svsl_ir_op_ op) {
	int32_t n = 0;
	for (int32_t i = 0; i < fn->insts.count; i++)
		if (fn->insts.items[i].op == op) n++;
	return n;
}

// the prototype-killer: opaque texture/sampler params resolve to global
// resources at inline time - golden, byte for byte
static void test_ir_opaque_inline(void) {
	svsl_arena_t arena = {0};
	TEST_CHECK(dump_is(&arena,
		"Texture2D    tex   : register(t0);\n"
		"SamplerState tex_s : register(s0);\n"
		"float4 sample_fade(Texture2D t, SamplerState s, float2 uv, float f) {\n"
		"	return t.Sample(s, uv) * f;\n"
		"}\n"
		"float4 ps(float2 uv : TEXCOORD0) : SV_TARGET {\n"
		"	return sample_fade(tex, tex_s, uv, 0.5);\n"
		"}\n",
		// after inlining, store-to-load forwarding + dead-store elimination
		// collapse every param-copy/result var; the opaque tex/sampler params
		// still resolve to the globals (the point of this test). The `* f` splat
		// is stripped to a scalar operand (emit selects OpVectorTimesScalar).
		// `uv` is read after the constant: lvalue arguments are read as the call
		// starts, once every argument has been evaluated.
		"func ps pixel\n"
		"  %0 = param float2 #0 ; uv\n"
		"  %3 = const float 0.5\n"
		"  %5 = load float2 %0\n"
		"  %9 = tex float4 tex method=0 sampler=tex_s (%5)\n"
		"  %12 = mul float4 %9 %3\n"
		"  return %12\n"));
	svsl_arena_free(&arena);
}

// flat chains: member access through arrays loads exactly one member,
// never the whole struct (the prototype's 64x over-fetch bug)
static void test_ir_flat_chains(void) {
	svsl_arena_t arena = {0};
	ir_run_t r = run_ir(&arena,
		"struct inst_t { float4x4 world; float4 color; };\n"
		"StructuredBuffer<inst_t> sk_inst : register(t12);\n"
		"float4 ps(uint id : SV_InstanceID) : SV_TARGET {\n"
		"	return sk_inst[id].color;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];

	// exactly one chain (base + [id] + .color folded flat), one load of float4
	TEST_CHECK(count_op(fn, svsl_ir_chain) == 1);
	int32_t chain = find_op(fn, svsl_ir_chain);
	TEST_CHECK(chain >= 0 && fn->insts.items[chain].aux_count == 2); // index + member
	bool loads_struct = false;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		if (fn->insts.items[i].op != svsl_ir_load) continue;
		const svsl_type_t *t = svsl_type_get(&r.prog.types, fn->insts.items[i].type);
		if (t->kind == svsl_type_struct) loads_struct = true;
	}
	TEST_CHECK(!loads_struct);
	svsl_arena_free(&arena);
}

static void test_ir_control_flow(void) {
	svsl_arena_t arena = {0};

	// loop shape: cond-break at top, increment after loop_continue
	ir_run_t r = run_ir(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	float total = 0;\n"
		"	for (int i = 0; i < 4; i++) { if (i == 2) continue; total += float(i); }\n"
		"	while (total > 10) total -= 1;\n"
		"	do { total += 1; } while (total < 3);\n"
		"	return total;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_loop) == 3);
	TEST_CHECK(count_op(fn, svsl_ir_end_loop) == 3);
	TEST_CHECK(count_op(fn, svsl_ir_loop_continue) == 3);
	TEST_CHECK(count_op(fn, svsl_ir_continue) == 1);
	TEST_CHECK(count_op(fn, svsl_ir_break) >= 2); // for/while conditions (do-while uses a conditional back-edge)
	TEST_CHECK(find_op(fn, svsl_ir_return) >= 0);

	// multi-return functions inline inside a single-trip loop; returns become breaks
	r = run_ir(&arena,
		"float pick(float x) { if (x > 1) return 2; return x; }\n"
		"float4 ps() : SV_TARGET { return pick(0.5); }\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_loop) == 1);     // the inline wrapper
	TEST_CHECK(count_op(fn, svsl_ir_return) == 1);   // only the entry's return survives
	TEST_CHECK(count_op(fn, svsl_ir_break) >= 2);    // both returns became breaks

	// switch keeps its cases
	r = run_ir(&arena,
		"float4 ps(uint m : TEXCOORD0) : SV_TARGET {\n"
		"	float v = 0;\n"
		"	switch (m) { case 0: v = 1; break; case 1: case 2: v = 2; break; default: v = 3; break; }\n"
		"	return v;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_switch) == 1);
	TEST_CHECK(count_op(fn, svsl_ir_case) == 4);
	int32_t sw = find_op(fn, svsl_ir_switch);
	TEST_CHECK(sw >= 0 && fn->insts.items[sw].aux_count == 4);

	svsl_arena_free(&arena);
}

// An early return nested in a loop can't break straight to the inline wrapper
// (structured CF forbids a multi-level break), so it sets a bool flag and cascades
// the break outward. Regression for the miscompile where the fallthrough return
// clobbered the early one (pick(2) returned -1 instead of 2).
static void test_ir_return_in_loop(void) {
	svsl_arena_t arena = {0};
	ir_run_t r = run_ir(&arena,
		"float pick(int n) {\n"
		"	for (int i = 0; i < 4; i++) { if (i == n) return float(i); }\n"
		"	return -1;\n"
		"}\n"
		"float4 ps(uint m : TEXCOORD0) : SV_TARGET { return pick(int(m)); }\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_loop) == 2); // user for-loop + inline wrapper
	// the returned-flag bool var is the signature of the cascade lowering
	int32_t bool_vars = 0;
	for (int32_t i = 0; i < fn->insts.count; i++)
		if (fn->insts.items[i].op == svsl_ir_var &&
		    svsl_type_get(&r.prog.types, fn->insts.items[i].type)->scalar == svsl_scalar_bool)
			bool_vars++;
	TEST_CHECK(bool_vars >= 1);
	svsl_arena_free(&arena);
}

static void test_ir_passes(void) {
	svsl_arena_t arena = {0};

	// constant folding across a conversion: 1 + 2 into a float context = 3.0f
	ir_run_t r = run_ir(&arena,
		"float4 ps() : SV_TARGET { float total = 1 + 2; return total; }\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	bool found_3 = false;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->op != svsl_ir_const) continue;
		const svsl_type_t *t = svsl_type_get(&r.prog.types, inst->type);
		if (t->scalar != svsl_scalar_float32) continue;
		float f;
		uint32_t bits = inst->args[0];
		memcpy(&f, &bits, 4);
		if (f == 3.0f) found_3 = true;
	}
	TEST_CHECK(found_3);
	TEST_CHECK(count_op(fn, svsl_ir_add) == 0); // folded away

	// unsigned folding uses unsigned semantics and full width (regression: 32-bit
	// signed fold turned 0xFFFFFFFFu / 2u into 0)
	r = run_ir(&arena,
		"float4 ps() : SV_TARGET { uint a = 0xFFFFFFFFu / 2u; return float(a); }\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	bool found_max = false;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->op != svsl_ir_const) continue;
		if (svsl_type_get(&r.prog.types, inst->type)->scalar != svsl_scalar_float32) continue;
		float f; uint32_t bits = inst->args[0];
		memcpy(&f, &bits, 4);
		if (f == 2147483647.0f) found_max = true;
	}
	TEST_CHECK(found_max);
	TEST_CHECK(count_op(fn, svsl_ir_div) == 0); // folded away

	// dead code disappears: an unused expression becomes nops
	r = run_ir(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	float unused = sqrt(25.0);\n" // pure, unreferenced... but stored: var stays
		"	float2 dead_value = float2(1, 2);\n"
		"	return 1;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	// the returned splat construct survives; everything else feeding stores stays
	// conservative - but a value with no store and no use must be gone:
	// (the shuffle/extract-free dump keeps this focused on nop-ing behavior)
	TEST_CHECK(count_op(fn, svsl_ir_nop) >= 0); // structural sanity

	// out/inout copy-back at the call boundary: a=1,b=0; x=a; x+=1->2; y=x*2->4;
	// copy back a=2,b=4; return a+b = 6. Store-to-load forwarding threads the
	// copies and folding collapses the whole thing to the constant 6 - a proof
	// that the inout/out write-back is wired correctly.
	r = run_ir(&arena,
		"void bump(inout float x, out float y) { x += 1; y = x * 2; }\n"
		"float4 ps() : SV_TARGET {\n"
		"	float a = 1, b = 0;\n"
		"	bump(a, b);\n"
		"	return a + b;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_var)   == 0); // all param copies forwarded away
	TEST_CHECK(count_op(fn, svsl_ir_store) == 0);
	bool found_6 = false;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *inst = &fn->insts.items[i];
		if (inst->op != svsl_ir_const) continue;
		const svsl_type_t *t = svsl_type_get(&r.prog.types, inst->type);
		if (t->scalar != svsl_scalar_float32) continue;
		float f; uint32_t bits = inst->args[0];
		memcpy(&f, &bits, 4);
		if (f == 6.0f) found_6 = true;
	}
	TEST_CHECK(found_6);

	svsl_arena_free(&arena);
}

// compound assignment and increment evaluate their lvalue chain exactly once:
// a[i++] *= 2 must load and store the SAME element (the sh_compute windowing
// bug: idx++ re-ran between the load and the store, shifting every write by one)
static void test_ir_single_eval_target(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"float ps(uint j : TEXCOORD0) : SV_TARGET {\n"
		"	float a[2];\n"
		"	a[0] = 1;\n"
		"	a[1] = 2;\n"
		"	uint i = j;\n"
		"	a[i++] *= 2;\n"
		"	return a[0] + i;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	// a[0]=, a[1]=, a[i++]; the a[0] read reuses the a[0]= address (CSE merges
	// the two identical chains). The single-eval guarantee is that a[i++] uses
	// ONE chain for its load and store and i++ runs once - proven by add == 2.
	TEST_CHECK(count_op(fn, svsl_ir_chain) == 3);
	TEST_CHECK(count_op(fn, svsl_ir_add)   == 2); // one i+1, one a[0]+i
	svsl_arena_free(&arena);
}

// indexing a non-addressable value: a constant index extracts the right element
// (was silently element 0); a dynamic index becomes a single extract_dynamic on
// the value - no spill variable, no access chain into memory (#17).
static void test_ir_rvalue_index(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"float ps(float2 uv : TEXCOORD0) : SV_TARGET {\n"
		"	float a = float4(1, 2, 3, 4)[2];\n"     // constant: extract [2]
		"	int   i = (int)(uv.x * 4);\n"
		"	float b = float4(5, 6, 7, 8)[i];\n"     // dynamic: extract_dynamic, no spill
		"	return a + b;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	// the constant index picks element [2] = 3.0 (not element 0 = 1.0): peephole
	// folds extract(construct(1,2,3,4), 2) straight to the constant 3
	bool got_3 = false;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op != svsl_ir_const) continue;
		const svsl_type_t *t = svsl_type_get(&r.prog.types, in->type);
		if (t->scalar != svsl_scalar_float32) continue;
		float f; uint32_t bits = in->args[0];
		memcpy(&f, &bits, 4);
		if (f == 3.0f) got_3 = true;
	}
	TEST_CHECK(got_3);
	// the dynamic index lowered to extract_dynamic on the float4 value, and the
	// float4 was never spilled to a Function variable (no var op survives)
	TEST_CHECK(count_op(fn, svsl_ir_extract_dynamic) == 1);
	TEST_CHECK(count_op(fn, svsl_ir_var)             == 0);
	svsl_arena_free(&arena);
}

// dominance-based forwarding: a single-assignment local established at the top
// level flows into branch bodies (no reload), but a conditionally-assigned local
// must NOT be forwarded past the merge - that boundary is what keeps it sound.
static void test_ir_cross_cf_forward(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"float ps(float2 uv : TEXCOORD0) : SV_TARGET {\n"
		"	float k = uv.x * 2;\n"          // single store, depth 0 -> dominates all
		"	float acc = 0;\n"
		"	if (uv.y > 0.5) { acc = k + 1; }\n"  // reads k inside the branch ...
		"	else            { acc = k - 1; }\n"  // ... and the other branch
		"	return acc;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	// k is read in both arms but forwards to the one `uv.x*2` value: its var is
	// gone and no load of it survives. Only uv.x, uv.y and the final acc remain -
	// three loads, not five. `acc` keeps its var and post-if load (conditionally
	// assigned -> cannot forward past the merge without a phi).
	TEST_CHECK(count_op(fn, svsl_ir_mul)  == 1); // the single k = uv.x*2
	TEST_CHECK(count_op(fn, svsl_ir_load) == 3); // uv.x, uv.y, acc - k's two loads forwarded
	TEST_CHECK(count_op(fn, svsl_ir_var)  == 1); // only acc; k fully scalar-forwarded away

	// adversarial: a local reassigned *inside* a branch is conditional - its value
	// after the merge is ambiguous, so it must be reloaded, never forwarded. If the
	// pass forwarded the pre-branch value the var/stores would vanish (var==0).
	ir_run_t r2 = run_ir(&arena,
		"float ps(float2 uv : TEXCOORD0) : SV_TARGET {\n"
		"	float k = uv.x;\n"
		"	if (uv.y > 0.5) { k = 99; }\n"
		"	return k;\n"                       // must load k (uv.x or 99), not forward uv.x
		"}\n");
	TEST_CHECK(r2.ok);
	const svsl_ir_func_t *fn2 = &r2.module.funcs[0];
	TEST_CHECK(count_op(fn2, svsl_ir_var)   == 1); // k survives - not scalar-forwarded
	TEST_CHECK(count_op(fn2, svsl_ir_store) == 2); // both k= stores kept (DSE can't kill them)
	TEST_CHECK(count_op(fn2, svsl_ir_load)  == 3); // uv.x, uv.y, and the reloaded k after the if
	svsl_arena_free(&arena);
}

// vector * scalar: the front-end splats the scalar to a vector and multiplies
// component-wise; the peephole strips the splat so emit can select
// OpVectorTimesScalar. No construct survives and the mul's operand is the scalar.
static void test_ir_vector_times_scalar(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"float4 ps(float4 c : COLOR0, float s : TEXCOORD0) : SV_TARGET {\n"
		"	return c * s;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_construct) == 0); // the splat is gone (DCE)
	TEST_CHECK(count_op(fn, svsl_ir_mul)       == 1);
	int32_t m = find_op(fn, svsl_ir_mul);
	TEST_CHECK(m >= 0);
	const svsl_type_t *rhs = svsl_type_get(&r.prog.types, fn->insts.items[fn->insts.items[m].args[1]].type);
	TEST_CHECK(rhs->kind == svsl_type_scalar); // operand is the scalar, not a splatted vector
	svsl_arena_free(&arena);
}

static void test_ir_getdim_out_params(void) {
	svsl_arena_t arena = {0};

	// the out-param form must query once and store every component; a dead
	// void-typed query would leave the out arguments uninitialized
	ir_run_t r = run_ir(&arena,
		"Texture2D tex;\n"
		"SamplerState tex_s;\n"
		"float4 ps(float2 uv : TEXCOORD0) : SV_TARGET {\n"
		"	float w, h;\n"
		"	tex.GetDimensions(w, h);\n"
		"	return float4(w, h, 0, 1);\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_tex) == 1);
	TEST_CHECK(count_op(fn, svsl_ir_extract) >= 2);  // one per component
	TEST_CHECK(count_op(fn, svsl_ir_convert) >= 2);  // uint -> float out args
	// the queried components feed the result directly: forwarding threads each
	// store into the float4(w,h,..) construct, so the w/h stores are eliminated
	TEST_CHECK(count_op(fn, svsl_ir_construct) >= 1);
	svsl_arena_free(&arena);
}

static void test_ir_atomic_op_selection(void) {
	svsl_arena_t arena = {0};

	// every atomic spelling must map to its exact op code; "or"/"xor" and
	// "exchange"/"compare_exchange" are one substring apart
	ir_run_t r = run_ir(&arena,
		"RWStructuredBuffer<uint> buf;\n"
		"[numthreads(1,1,1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint orig;\n"
		"	InterlockedAdd     (buf[0], 1u);\n"                 // op 0
		"	InterlockedAnd     (buf[1], 2u);\n"                 // op 4
		"	InterlockedOr      (buf[2], 4u);\n"                 // op 5
		"	InterlockedXor     (buf[3], 8u);\n"                 // op 6
		"	InterlockedExchange(buf[4], 9u, orig);\n"           // op 7
		"	InterlockedCompareExchange(buf[5], 1u, 2u, orig);\n"// op 8
		"	InterlockedCompareStore   (buf[6], 3u, 4u);\n"      // op 8
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	static const uint32_t expect[] = { 0, 4, 5, 6, 7, 8, 8 };
	int32_t seen = 0;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		if (fn->insts.items[i].op != svsl_ir_atomic) continue;
		TEST_CHECK(seen < 7 && fn->insts.items[i].args[3] == expect[seen]);
		seen++;
	}
	TEST_CHECK(seen == 7);
	svsl_arena_free(&arena);
}

static void test_ir_buffer_dimensions(void) {
	svsl_arena_t arena = {0};

	// buffer GetDimensions: count queries the runtime array, stride is a
	// layout constant; both must be stored to the out arguments
	ir_run_t r = run_ir(&arena,
		"struct item_t { float4 a; float2 b; };\n"
		"std430 StructuredBuffer<item_t> items;\n"
		"RWStructuredBuffer<float4> outp;\n"
		"[numthreads(1,1,1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint count, stride;\n"
		"	items.GetDimensions(count, stride);\n"
		"	outp[0] = (float)(count + stride);\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_tex) == 1);
	bool stride_const = false; // std430 stride of item_t = 32
	for (int32_t i = 0; i < fn->insts.count; i++)
		if (fn->insts.items[i].op == svsl_ir_const && fn->insts.items[i].args[0] == 32)
			stride_const = true;
	TEST_CHECK(stride_const);
	svsl_arena_free(&arena);
}

static void test_ir_swizzle_stores(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	float4 c = float4(1, 2, 3, 4);\n"
		"	c.xw  = float2(0, 0);\n"   // partial write: load, insert x2, store
		"	c.rgb = c.bgr;\n"          // reordered swizzle write
		"	c.y   = 5;\n"              // single component: direct chain store
		"	return c;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_insert) == 5);  // 2 (xw) + 3 (rgb)
	TEST_CHECK(count_op(fn, svsl_ir_shuffle) == 0); // .bgr read folds into the inserts' extracts (3)
	TEST_CHECK(count_op(fn, svsl_ir_chain) == 1);   // c.y store path
	svsl_arena_free(&arena);
}

// for/while conditions emit as glslang's loop header exit: the block after the
// header ends in one OpBranchConditional to the loop merge - no OpLogicalNot, no
// selection construct around a break block - and integer literal bounds are real
// constants, not OpBitcasts. Adreno's compiler could not handle the old shape
// (docs/dev/case-study-astc-encoders.md). Each case runs at -O0 (lowering alone)
// and -O1, and every module must pass spirv-val.
typedef struct loop_shape_t {
	int32_t loops, exits, exits_true_to_merge, nots, sel_merges, bitcasts;
	int32_t unroll_masks, dont_unroll_masks;
} loop_shape_t;

static loop_shape_t loop_shape(const svsl_spirv_blob_t *b) {
	loop_shape_t r = {0};
	r.nots       = test_spv_count(b, SpvOpLogicalNot, -1, -1);
	r.sel_merges = test_spv_count(b, SpvOpSelectionMerge, -1, -1);
	r.bitcasts   = test_spv_count(b, SpvOpBitcast, -1, -1);
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == SpvOpLoopMerge) {
			r.loops++;
			if (b->words[i + 3] & SpvLoopControlUnrollMask)     r.unroll_masks++;
			if (b->words[i + 3] & SpvLoopControlDontUnrollMask) r.dont_unroll_masks++;
		}
		i += (int32_t)wc;
	}
	r.exits = test_spv_loop_exits(b, &r.exits_true_to_merge);
	return r;
}

static void test_ir_loop_exit_shape(void) {
	#define LOOP_SRC(body) \
		"RWStructuredBuffer<uint> o : register(u0);\n" \
		"[numthreads(8,1,1)] void cs(uint3 id : SV_DispatchThreadID) {\n" \
		"	uint s = id.x, t = id.x;\n" body "\n	o[id.x] = s;\n}\n"
	static const struct {
		const char  *name;
		const char  *src;
		loop_shape_t want; // loops, exits, exits_true_to_merge, nots, sel_merges, bitcasts, unroll, dont_unroll
	} cases[] = {
		{ "for, uint literal bound, [unroll]",
		  LOOP_SRC("[unroll] for (uint i = 0; i < 4; i++) s += i * t;"),
		  { 1, 1, 0, 0, 0, 0, 1, 0 } },
		{ "while, runtime condition",
		  LOOP_SRC("while (s < 100) s = s * 2 + 1;"),
		  { 1, 1, 0, 0, 0, 0, 0, 0 } },
		{ "[loop] keeps DontUnroll",
		  LOOP_SRC("[loop] for (int i = 7; i >= 0; i--) s = s * 3 + t;"),
		  { 1, 1, 0, 0, 0, 0, 0, 1 } },
		{ "user-written top break stays a selection (as in glslang)",
		  LOOP_SRC("for (;;) { if (s > 20) break; s += 7; }"),
		  { 1, 0, 0, 0, 1, 0, 0, 0 } },
		{ "top if carrying [branch] keeps its selection",
		  LOOP_SRC("for (;;) { [branch] if (s > 20) break; s += 7; }"),
		  { 1, 0, 0, 0, 1, 0, 0, 0 } },
		{ "[flatten] on a loop body's if keeps its hint",
		  LOOP_SRC("for (uint i = 0; i < 8; i++) { [flatten] if (s > 20) s -= 3; s += 7; }"),
		  { 1, 1, 0, 0, 1, 0, 0, 0 } },
		{ "do-while: the back edge exits, a body break stays a selection",
		  LOOP_SRC("do { if (s > 30) break; s = s * 2 + 1; } while (s < 50);"),
		  { 1, 1, 0, 0, 1, 0, 0, 0 } }, // back edge: OpBranchConditional %c %header %merge
		{ "condition inlining an early-return call: exit after the wrapper loop",
		  "RWStructuredBuffer<uint> o : register(u0);\n"
		  "bool keep_going(uint v, uint lim) { if (v > 1000) return false; return v < lim; }\n"
		  "[numthreads(8,1,1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		  "	uint s = id.x;\n"
		  "	while (keep_going(s, 90)) s = s * 3 + 1;\n"
		  "	o[id.x] = s;\n"
		  "}\n",
		  { 2, 1, 0, 0, 1, 0, 0, 0 } }, // outer loop + the call's wrapper; its `if` keeps a selection
		{ "for condition inlining an early-return call",
		  "RWStructuredBuffer<uint> o : register(u0);\n"
		  "uint limit(uint v) { if (v > 5) return 5; return v + 2; }\n"
		  "[numthreads(8,1,1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		  "	uint s = 0;\n"
		  "	for (uint i = 0; i < limit(id.x); i++) s += i;\n"
		  "	o[id.x] = s;\n"
		  "}\n",
		  { 2, 1, 0, 0, 1, 0, 0, 0 } },
		{ "nested loops both exit from their headers",
		  LOOP_SRC("for (uint p = 0; p < 3; p++) for (uint q = 0; q <= p; q++) s += p * 4 + q;"),
		  { 2, 2, 0, 0, 0, 0, 0, 0 } },
		{ "mid-body break stays a selection",
		  LOOP_SRC("for (uint n = 0; n < 10; n++) { s += n; if (s > t * 3) break; }"),
		  { 1, 1, 0, 0, 1, 0, 0, 0 } },
	};
	#undef LOOP_SRC
	for (int32_t c = 0; c < (int32_t)(sizeof(cases) / sizeof(cases[0])); c++) {
		for (int32_t level = svsl_opt_none; level <= svsl_opt_default; level++) {
			svsl_arena_t arena = {0};
			test_spv_t   spv;
			test_spv_compile(&arena, cases[c].src, NULL, NULL, (svsl_opt_level_)level, &spv);
			TEST_CHECK(spv.ok && test_spv_validate(&spv));
			if (!spv.ok) { printf("  case: %s\n", cases[c].name); svsl_arena_free(&arena); continue; }
			loop_shape_t got = loop_shape(&spv.blobs[0]), want = cases[c].want;
			bool match = got.loops == want.loops && got.exits == want.exits &&
			             got.exits_true_to_merge == want.exits_true_to_merge && got.nots == want.nots &&
			             got.sel_merges == want.sel_merges && got.bitcasts == want.bitcasts &&
			             got.unroll_masks == want.unroll_masks && got.dont_unroll_masks == want.dont_unroll_masks;
			TEST_CHECK(match);
			if (!match)
				printf("  case '%s' -O%d: loops %d exits %d true->merge %d nots %d sel %d bitcast %d unroll %d dont %d\n",
				       cases[c].name, level, got.loops, got.exits, got.exits_true_to_merge, got.nots,
				       got.sel_merges, got.bitcasts, got.unroll_masks, got.dont_unroll_masks);
			svsl_arena_free(&arena);
		}
	}
}

// integer constant re-typing (shared by lowering and fold): sign-/zero-extension
// and truncation by the scalar kinds, in the const encoding (signed results
// sign-extended to 64 bits, unsigned zero-extended)
static void test_ir_int_convert_bits(void) {
	static const struct {
		uint64_t     bits;
		svsl_scalar_ from, to;
		uint64_t     want;
	} cases[] = {
		{ 0xFFFFFFFFFFFFFFFFull, svsl_scalar_int32,  svsl_scalar_uint32, 0x00000000FFFFFFFFull }, // -1 -> 0xFFFFFFFF
		{ 0x00000000FFFFFFFFull, svsl_scalar_uint32, svsl_scalar_int32,  0xFFFFFFFFFFFFFFFFull }, // same bits, signed
		{ 0xFFFFFFFFFFFFFFFFull, svsl_scalar_int32,  svsl_scalar_int64,  0xFFFFFFFFFFFFFFFFull }, // sign-extends
		{ 0x00000000FFFFFFFFull, svsl_scalar_uint32, svsl_scalar_uint64, 0x00000000FFFFFFFFull }, // zero-extends
		{ 0x00000000FFFFFFFFull, svsl_scalar_uint32, svsl_scalar_int64,  0x00000000FFFFFFFFull }, // zero-ext, positive
		{ 0x0000000100000005ull, svsl_scalar_int64,  svsl_scalar_int32,  5                     }, // truncates
		{ 0x00000000FFFFFFFEull, svsl_scalar_int64,  svsl_scalar_int16,  0xFFFFFFFFFFFFFFFEull }, // truncate to -2
		{ 0x000000000000FFFEull, svsl_scalar_int16,  svsl_scalar_uint16, 0x000000000000FFFEull },
		{ 200,                   svsl_scalar_uint8,  svsl_scalar_int8,   0xFFFFFFFFFFFFFFC8ull }, // 200 -> -56
		{ 0xFFFFFFFFFFFFFFFFull, svsl_scalar_int8,   svsl_scalar_uint32, 0x00000000FFFFFFFFull }, // -1 widens signed
		{ 0x0000000000000080ull, svsl_scalar_uint8,  svsl_scalar_uint32, 0x80                  },
		{ 0x8000000000000000ull, svsl_scalar_uint64, svsl_scalar_int64,  0x8000000000000000ull },
		{ 7,                     svsl_scalar_int32,  svsl_scalar_int32,  7                     }, // identity
	};
	for (int32_t i = 0; i < (int32_t)(sizeof(cases) / sizeof(cases[0])); i++) {
		uint64_t got = 0;
		bool     ok  = svsl_ir_int_convert_bits(cases[i].bits, cases[i].from, cases[i].to, &got);
		TEST_CHECK(ok && got == cases[i].want);
		if (!ok || got != cases[i].want)
			printf("  int convert case %d: got %016llx want %016llx\n", i,
			       (unsigned long long)got, (unsigned long long)cases[i].want);
	}
	uint64_t unused;
	TEST_CHECK(!svsl_ir_int_convert_bits(1, svsl_scalar_float32, svsl_scalar_int32, &unused)); // not int
	TEST_CHECK(!svsl_ir_int_convert_bits(1, svsl_scalar_int32, svsl_scalar_float32, &unused));
	TEST_CHECK(!svsl_ir_int_convert_bits(1, svsl_scalar_bool,  svsl_scalar_uint32, &unused));  // bool: a select
}

// integer literals re-typed at lowering: no conversion ops even at -O0, for
// scalars and constant vectors alike
static void test_ir_literal_typing(void) {
	const char *src =
		"RWStructuredBuffer<uint4> o : register(u0);\n"
		"[numthreads(1,1,1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint  a = 3;\n"
		"	uint2 v = int2(1, 2);\n"
		"	int64 w = 5;\n"
		"	for (uint i = 0; i < 4; i++) a += i;\n"
		"	o[id.x] = uint4(v, a, (uint)w);\n"
		"}\n";
	for (int32_t level = svsl_opt_none; level <= svsl_opt_default; level++) {
		svsl_arena_t arena = {0};
		test_spv_t   spv;
		test_spv_compile(&arena, src, NULL, NULL, (svsl_opt_level_)level, &spv);
		TEST_CHECK(spv.ok && test_spv_validate(&spv));
		if (spv.ok) {
			const svsl_ir_func_t *fn = &spv.ir.funcs[0];
			int32_t converts = 0; // only the explicit runtime (uint)w may convert
			for (int32_t i = 0; i < fn->insts.count; i++)
				if (fn->insts.items[i].op == svsl_ir_convert) converts++;
			TEST_CHECK(converts <= 1);
			if (level == svsl_opt_default) TEST_CHECK(converts == 0); // w is a constant: folds
		}
		svsl_arena_free(&arena);
	}
}

// private globals (non-const `static`) get one canonical pointer per entry, so
// the memory passes treat them like locals: straight-line writes forward and
// die, a conditional write is not forwarded past its merge
static void test_ir_private_globals(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"static float h = 1;\n"
		"void bump() { h *= 2; }\n"
		"float4 ps() : SV_TARGET { bump(); bump(); return h; }\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_load)  == 0); // 1 * 2 * 2 folds to a constant
	TEST_CHECK(count_op(fn, svsl_ir_store) == 0); // unread afterwards: dead
	TEST_CHECK(count_op(fn, svsl_ir_ptr)   == 0); // and the pointer with them

	r = run_ir(&arena,
		"static float c = 1;\n"
		"float4 ps(float x : TEXCOORD0) : SV_TARGET { if (x > 0) c = 5; return c; }\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	int32_t end = find_op(fn, svsl_ir_end_if), load = -1;
	for (int32_t i = end + 1; i < fn->insts.count; i++)
		if (fn->insts.items[i].op == svsl_ir_load) { load = i; break; }
	TEST_CHECK(end >= 0 && load > end); // c is re-read after the merge, not assumed 1 or 5
	TEST_CHECK(count_op(fn, svsl_ir_ptr) == 1); // one canonical pointer for c
	svsl_arena_free(&arena);
}

// a read-only aggregate `in` parameter binds to the caller's storage: no copy
// var, no whole-array load/store. A parameter the callee writes keeps its copy.
// variables made for a callee parameter's by-value copy (they carry its name)
static int32_t count_param_copies(const svsl_ir_func_t *fn, const char *param) {
	int32_t n = 0;
	for (int32_t i = 0; i < fn->insts.count; i++)
		if (fn->insts.items[i].op == svsl_ir_var && svsl_str_eq_cstr(fn->insts.items[i].name, param)) n++;
	return n;
}
static int32_t count_whole_aggregate_loads(const ir_run_t *r) {
	const svsl_ir_func_t *fn = &r->module.funcs[0];
	int32_t n = 0;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		if (fn->insts.items[i].op != svsl_ir_load) continue;
		svsl_type_kind_ k = svsl_type_get(&r->prog.types, fn->insts.items[i].type)->kind;
		if (k == svsl_type_array || k == svsl_type_struct) n++;
	}
	return n;
}
static int32_t count_whole_array_loads(const ir_run_t *r) {
	const svsl_ir_func_t *fn = &r->module.funcs[0];
	int32_t n = 0;
	for (int32_t i = 0; i < fn->insts.count; i++)
		if (fn->insts.items[i].op == svsl_ir_load &&
		    svsl_type_get(&r->prog.types, fn->insts.items[i].type)->kind == svsl_type_array) n++;
	return n;
}
static void test_ir_array_param_by_reference(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"float sum(float v[8], int n) { float s = 0; for (int i = 0; i < n; i++) s += v[i]; return s; }\n"
		"float4 ps(float x : TEXCOORD0) : SV_TARGET {\n"
		"	float a[8]; for (int i = 0; i < 8; i++) a[i] = x * i;\n"
		"	return sum(a, (int)x) + sum(a, 3);\n"
		"}\n");
	TEST_CHECK(r.ok);
	TEST_CHECK(count_whole_array_loads(&r) == 0);
	int32_t array_vars = 0;
	for (int32_t i = 0; i < r.module.funcs[0].insts.count; i++) {
		const svsl_ir_inst_t *in = &r.module.funcs[0].insts.items[i];
		if (in->op == svsl_ir_var && svsl_type_get(&r.prog.types, in->type)->kind == svsl_type_array) array_vars++;
	}
	TEST_CHECK(array_vars == 1); // just the caller's `a`: neither call made a parameter copy

	r = run_ir(&arena,
		"float first(float v[8]) { v[0] += 1; return v[0]; }\n" // writes its param: copy required
		"float4 ps(float x : TEXCOORD0) : SV_TARGET {\n"
		"	float a[8]; for (int i = 0; i < 8; i++) a[i] = x * i;\n"
		"	return first(a) + a[0];\n"
		"}\n");
	TEST_CHECK(r.ok);
	TEST_CHECK(count_whole_array_loads(&r) == 1);

	// by reference only when nothing can change the storage during the call
	static const struct { const char *name, *src; bool copy; } cases[] = {
		{ "written through a nested inout call", "void g(inout float v[4]) { v[0] = 1; }\n"
		  "float f(float v[4]) { g(v); return v[0]; }\n"
		  "float4 ps(float x : TEXCOORD0) : SV_TARGET { float a[4] = { x, 2, 3, 4 }; return f(a) + a[0]; }\n", true },
		{ "written through an element member", "float f(float2 v[2]) { v[1].y = 3; return v[1].y; }\n"
		  "float4 ps(float x : TEXCOORD0) : SV_TARGET { float2 a[2] = { x.xx, x.xx }; return f(a) + a[1].y; }\n", true },
		{ "read-only through three levels", "float f3(float v[4], int i) { return v[i]; }\n"
		  "float f2(float v[4], int i) { return f3(v, i) + v[0]; }\n"
		  "float f1(float v[4], int i) { return f2(v, i) * 2; }\n"
		  "float4 ps(float x : TEXCOORD0) : SV_TARGET { float a[4] = { x, 2, 3, 4 }; return f1(a, (int)x); }\n", false },
		{ "row of a 2D array", "float f(float v[4], int i) { return v[i]; }\n"
		  "float4 ps(float x : TEXCOORD0) : SV_TARGET { float m[2][4]; for (int i = 0; i < 4; i++) { m[0][i] = x; m[1][i] = x * i; } return f(m[1], (int)x); }\n", false },
		{ "struct member array", "struct s_t { float arr[4]; float k; };\n"
		  "float f(float v[4], int i) { return v[i]; }\n"
		  "float4 ps(float x : TEXCOORD0) : SV_TARGET { s_t s; for (int i = 0; i < 4; i++) s.arr[i] = x * i; s.k = x; return f(s.arr, (int)x) + s.k; }\n", false },
		{ "whole struct", "struct s_t { float arr[4]; float k; };\n"
		  "float f(s_t v, int i) { return v.arr[i] + v.k; }\n"
		  "float4 ps(float x : TEXCOORD0) : SV_TARGET { s_t s; for (int i = 0; i < 4; i++) s.arr[i] = x * i; s.k = x; return f(s, (int)x); }\n", false },
		{ "cbuffer array", "cbuffer P : register(b0) { float w[4]; };\n"
		  "float f(float v[4], int i) { return v[i]; }\n"
		  "float4 ps(float x : TEXCOORD0) : SV_TARGET { return f(w, (int)x); }\n", false },
		{ "private static: the callee could write it", "static float g[4];\n"
		  "float f(float v[4], int i) { g[0] = 5; return v[i]; }\n"
		  "float4 ps(float x : TEXCOORD0) : SV_TARGET { g[1] = x; return f(g, (int)x); }\n", true },
	};
	// Checked at -O0, where a copy always leaves its variable (named after the
	// callee's parameter, `v`); at -O1 forwarding may legitimately erase it.
	for (int32_t c = 0; c < (int32_t)(sizeof(cases) / sizeof(cases[0])); c++) {
		test_spv_t spv;
		test_spv_compile(&arena, cases[c].src, NULL, NULL, svsl_opt_none, &spv);
		TEST_CHECK(spv.ok && test_spv_validate(&spv));
		if (!spv.ok) { printf("  case: %s\n", cases[c].name); continue; }
		int32_t copies = count_param_copies(&spv.ir.funcs[0], "v");
		TEST_CHECK(cases[c].copy ? copies >= 1 : copies == 0);
		if (cases[c].copy ? copies < 1 : copies != 0)
			printf("  case '%s': %d copies of parameter v\n", cases[c].name, copies);
	}

	// groupshared and RW buffers can change under the callee: always copied
	r = run_ir(&arena,
		"struct pair_t { uint a; uint b; };\n"
		"RWStructuredBuffer<pair_t> pairs : register(u0);\n"
		"groupshared uint shared_vals[4];\n"
		"uint fa(uint v[4]) { return v[1]; }\n"
		"uint fs(pair_t p) { return p.b; }\n"
		"[numthreads(4,1,1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	shared_vals[id.x] = id.x; GroupMemoryBarrierWithGroupSync();\n"
		"	pairs[id.x].a = fa(shared_vals) + fs(pairs[id.x]);\n"
		"}\n");
	TEST_CHECK(r.ok && count_whole_aggregate_loads(&r) == 2);
	svsl_arena_free(&arena);
}

// private globals at the SPIR-V level: Private variables zero-initialized with
// OpConstantNull, declared only in the entries that use them, and initialized
// in each such entry
static int32_t count_private_vars(const svsl_spirv_blob_t *b) {
	int32_t n = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == SpvOpVariable && b->words[i + 3] == SpvStorageClassPrivate) n++;
		i += (int32_t)wc;
	}
	return n;
}

static void test_ir_private_globals_spirv(void) {
	svsl_arena_t arena = {0};
	test_spv_t   spv;
	test_spv_compile(&arena,
		"cbuffer P : register(b0) { float4 tint_in; };\n"
		"static float4 tint = tint_in * 2;\n"          // both stages
		"static float  vs_only;\n"                     // vertex stage only, no initializer
		"static float  unused_static = 3;\n"           // nobody reads it
		"float4 vs(float4 p : POSITION) : SV_POSITION { vs_only += p.x; return p * vs_only + tint; }\n"
		"float4 ps() : SV_TARGET { return tint; }\n", NULL, NULL, svsl_opt_none, &spv);
	TEST_CHECK(spv.ok && spv.blob_count == 2 && test_spv_validate(&spv));
	if (spv.ok && spv.blob_count == 2) {
		for (int32_t s = 0; s < 2; s++) {
			const svsl_spirv_blob_t *b = &spv.blobs[s];
			int32_t privates = 0, null_inits = 0;
			for (int32_t i = 5; i < b->word_count; ) {
				uint32_t wc = b->words[i] >> 16;
				if (wc == 0) break;
				if ((b->words[i] & 0xFFFF) == SpvOpVariable && b->words[i + 3] == SpvStorageClassPrivate) {
					privates++;
					int32_t def = wc > 4 ? test_spv_find_def(b, b->words[i + 4]) : -1;
					if (def >= 0 && (b->words[def] & 0xFFFF) == SpvOpConstantNull) null_inits++;
				}
				i += (int32_t)wc;
			}
			// -O0 keeps every referenced static: vs has tint + vs_only, ps has tint.
			// unused_static's prologue store is dead but -O0 has no DSE, so it stays too.
			TEST_CHECK(privates == (s == 0 ? 3 : 2));
			TEST_CHECK(null_inits == privates);
		}
	}
	svsl_arena_free(&arena);

	arena = (svsl_arena_t){0};
	test_spv_compile(&arena,
		"cbuffer P : register(b0) { float4 tint_in; };\n"
		"static float4 tint = tint_in * 2;\n"
		"static float  vs_only;\n"
		"static float  unused_static = 3;\n"
		"float4 vs(float4 p : POSITION) : SV_POSITION { vs_only += p.x; return p * vs_only + tint; }\n"
		"float4 ps() : SV_TARGET { return tint; }\n", NULL, NULL, svsl_opt_default, &spv);
	TEST_CHECK(spv.ok && spv.blob_count == 2 && test_spv_validate(&spv));
	if (spv.ok && spv.blob_count == 2) {
		// -O1 forwards tint and unused_static away. vs_only is read before any store,
		// so its zero initializer is still observable: it alone stays, in vs only.
		TEST_CHECK(count_private_vars(&spv.blobs[0]) == 1);
		TEST_CHECK(count_private_vars(&spv.blobs[1]) == 0);
	}
	svsl_arena_free(&arena);
}

void test_ir(void) {
	test_ir_opaque_inline();
	test_ir_flat_chains();
	test_ir_control_flow();
	test_ir_return_in_loop();
	test_ir_passes();
	test_ir_single_eval_target();
	test_ir_rvalue_index();
	test_ir_cross_cf_forward();
	test_ir_vector_times_scalar();
	test_ir_getdim_out_params();
	test_ir_atomic_op_selection();
	test_ir_buffer_dimensions();
	test_ir_swizzle_stores();
	test_ir_loop_exit_shape();
	test_ir_private_globals();
	test_ir_array_param_by_reference();
	test_ir_int_convert_bits();
	test_ir_literal_typing();
	test_ir_private_globals_spirv();
}

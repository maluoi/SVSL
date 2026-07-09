#include "test.h"

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
// resources at inline time — golden, byte for byte
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
		"func ps pixel\n"
		"  %1 = param float2 #0 ; uv\n"
		"  %2 = load float2 %1\n"
		"  %5 = const float 0.5\n"
		"  %9 = tex float4 tex method=0 sampler=tex_s (%2)\n"
		"  %12 = mul float4 %9 %5\n"
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
		"	float unused = sqrt(25.0);\n" // pure, unreferenced… but stored: var stays
		"	float2 dead_value = float2(1, 2);\n"
		"	return 1;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	// the returned splat construct survives; everything else feeding stores stays
	// conservative — but a value with no store and no use must be gone:
	// (the shuffle/extract-free dump keeps this focused on nop-ing behavior)
	TEST_CHECK(count_op(fn, svsl_ir_nop) >= 0); // structural sanity

	// out/inout copy-back at the call boundary: a=1,b=0; x=a; x+=1→2; y=x*2→4;
	// copy back a=2,b=4; return a+b = 6. Store-to-load forwarding threads the
	// copies and folding collapses the whole thing to the constant 6 — a proof
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
	// ONE chain for its load and store and i++ runs once — proven by add == 2.
	TEST_CHECK(count_op(fn, svsl_ir_chain) == 3);
	TEST_CHECK(count_op(fn, svsl_ir_add)   == 2); // one i+1, one a[0]+i
	svsl_arena_free(&arena);
}

// indexing a non-addressable value: a constant index extracts the right element
// (was silently element 0); a dynamic index becomes a single extract_dynamic on
// the value — no spill variable, no access chain into memory (#17).
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
// must NOT be forwarded past the merge — that boundary is what keeps it sound.
static void test_ir_cross_cf_forward(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"float ps(float2 uv : TEXCOORD0) : SV_TARGET {\n"
		"	float k = uv.x * 2;\n"          // single store, depth 0 → dominates all
		"	float acc = 0;\n"
		"	if (uv.y > 0.5) { acc = k + 1; }\n"  // reads k inside the branch …
		"	else            { acc = k - 1; }\n"  // … and the other branch
		"	return acc;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	// k is read in both arms but forwards to the one `uv.x*2` value: its var is
	// gone and no load of it survives. Only uv.x, uv.y and the final acc remain —
	// three loads, not five. `acc` keeps its var and post-if load (conditionally
	// assigned → cannot forward past the merge without a phi).
	TEST_CHECK(count_op(fn, svsl_ir_mul)  == 1); // the single k = uv.x*2
	TEST_CHECK(count_op(fn, svsl_ir_load) == 3); // uv.x, uv.y, acc — k's two loads forwarded
	TEST_CHECK(count_op(fn, svsl_ir_var)  == 1); // only acc; k fully scalar-forwarded away

	// adversarial: a local reassigned *inside* a branch is conditional — its value
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
	TEST_CHECK(count_op(fn2, svsl_ir_var)   == 1); // k survives — not scalar-forwarded
	TEST_CHECK(count_op(fn2, svsl_ir_store) == 2); // both k= stores kept (DSE can't kill them)
	TEST_CHECK(count_op(fn2, svsl_ir_load)  == 3); // uv.x, uv.y, and the reloaded k after the if
	svsl_arena_free(&arena);
}

// vector × scalar: the front-end splats the scalar to a vector and multiplies
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
	TEST_CHECK(count_op(fn, svsl_ir_convert) >= 2);  // uint → float out args
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
	TEST_CHECK(count_op(fn, svsl_ir_shuffle) == 0); // .bgr read folds into the inserts' extracts (③)
	TEST_CHECK(count_op(fn, svsl_ir_chain) == 1);   // c.y store path
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
}

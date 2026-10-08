#include "test.h"
#include "test_spv.h"

#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "ir/ir.h"
#include "ir/ir_const.h"
#include "ir/ir_verify.h"
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

// count_op, leaving out the stage interface: its io pointers and what reads or
// writes through them (every entry stores its outputs; inputs are loads)
static int32_t count_body_op(const svsl_ir_func_t *fn, svsl_ir_op_ op) {
	int32_t n = 0;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op != op) continue;
		uint32_t p = in->op == svsl_ir_ptr ? (uint32_t)i : (in->op == svsl_ir_load || in->op == svsl_ir_store ||
		             in->op == svsl_ir_chain) ? in->args[0] : SVSL_IR_NONE;
		while (p != SVSL_IR_NONE && fn->insts.items[p].op == svsl_ir_chain) p = fn->insts.items[p].args[0];
		if (p != SVSL_IR_NONE && fn->insts.items[p].op == svsl_ir_ptr && fn->insts.items[p].args[0] == svsl_ref_stage_io)
			continue;
		n++;
	}
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
		// The entry reads its input slot and stores its output slot.
		"func ps pixel\n"
		"  %0 = ptr float2 io 0 0 ; uv\n"
		"  %1 = ptr float4 io 1 0 ; ps\n"
		"  %2 = const float 0.5\n"
		"  %3 = load float2 %0\n"
		"  %4 = tex float4 tex method=0 sampler=tex_s (%3)\n"
		"  %5 = mul float4 %4 %2\n"
		"  store %1 %5\n"
		"  return\n"));
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
	// (a runtime argument: with a constant one the early return folds away, and
	// the cfg pass then flattens the wrapper - see test_ir_cfg)
	r = run_ir(&arena,
		"float pick(float x) { if (x > 1) return 2; return x; }\n"
		"float4 ps(float x : TEXCOORD0) : SV_TARGET { return pick(x); }\n");
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

	// dead code disappears: unread locals and their values are removed, and the
	// committed function is dense (no nop placeholders survive optimization)
	r = run_ir(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	float unused = sqrt(25.0);\n"
		"	float2 dead_value = float2(1, 2);\n"
		"	return 1;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_nop)       == 0);
	TEST_CHECK(count_op(fn, svsl_ir_var)       == 0); // both locals dead-stored away
	TEST_CHECK(count_op(fn, svsl_ir_intrinsic) == 0); // the pure sqrt went with them

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
	TEST_CHECK(count_body_op(fn, svsl_ir_store) == 0);
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
// `[branch]` keeps those branches (if-conversion would otherwise turn them into
// selects - test_ir_if_convert covers that).
static void test_ir_cross_cf_forward(void) {
	svsl_arena_t arena = {0};

	ir_run_t r = run_ir(&arena,
		"float ps(float2 uv : TEXCOORD0) : SV_TARGET {\n"
		"	float k = uv.x * 2;\n"          // single store, depth 0 -> dominates all
		"	float acc = 0;\n"
		"	[branch] if (uv.y > 0.5) { acc = k + 1; }\n"  // reads k inside the branch ...
		"	else                     { acc = k - 1; }\n"  // ... and the other branch
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
		"	[branch] if (uv.y > 0.5) { k = 99; }\n"
		"	return k;\n"                       // must load k (uv.x or 99), not forward uv.x
		"}\n");
	TEST_CHECK(r2.ok);
	const svsl_ir_func_t *fn2 = &r2.module.funcs[0];
	TEST_CHECK(count_op(fn2, svsl_ir_var)   == 1); // k survives - not scalar-forwarded
	TEST_CHECK(count_body_op(fn2, svsl_ir_store) == 2); // both k= stores kept (DSE can't kill them)
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
		int32_t      min_level; // svsl_opt_ below which the case doesn't apply (an optimization's shape)
	} cases[] = {
		{ "for, uint literal bound, [unroll]",
		  LOOP_SRC("[unroll] for (uint i = 0; i < 4; i++) s += i * t;"),
		  { 1, 1, 0, 0, 0, 0, 1, 0 }, svsl_opt_none },
		{ "while, runtime condition",
		  LOOP_SRC("while (s < 100) s = s * 2 + 1;"),
		  { 1, 1, 0, 0, 0, 0, 0, 0 }, svsl_opt_none },
		{ "[loop] keeps DontUnroll",
		  LOOP_SRC("[loop] for (int i = 7; i >= 0; i--) s = s * 3 + t;"),
		  { 1, 1, 0, 0, 0, 0, 0, 1 }, svsl_opt_none },
		{ "user-written top break stays a selection (as in glslang)",
		  LOOP_SRC("for (;;) { if (s > 20) break; s += 7; }"),
		  { 1, 0, 0, 0, 1, 0, 0, 0 }, svsl_opt_none },
		{ "top if carrying [branch] keeps its selection",
		  LOOP_SRC("for (;;) { [branch] if (s > 20) break; s += 7; }"),
		  { 1, 0, 0, 0, 1, 0, 0, 0 }, svsl_opt_none },
		{ "[flatten] on a loop body's if flattens it to a select",
		  LOOP_SRC("for (uint i = 0; i < 8; i++) { [flatten] if (s > 20) s -= 3; s += 7; }"),
		  { 1, 1, 0, 0, 0, 0, 0, 0 }, svsl_opt_default }, // if-conversion (-O1)
		{ "[flatten] that can't flatten (a buffer store) keeps its hint",
		  LOOP_SRC("for (uint i = 0; i < 8; i++) { [flatten] if (s > 20) o[i] = s; s += 7; }"),
		  { 1, 1, 0, 0, 1, 0, 0, 0 }, svsl_opt_none },
		{ "do-while: the back edge exits, a body break stays a selection",
		  LOOP_SRC("do { if (s > 30) break; s = s * 2 + 1; } while (s < 50);"),
		  { 1, 1, 0, 0, 1, 0, 0, 0 }, svsl_opt_none }, // back edge: OpBranchConditional %c %header %merge
		{ "condition inlining an early-return call: exit after the wrapper loop",
		  "RWStructuredBuffer<uint> o : register(u0);\n"
		  "bool keep_going(uint v, uint lim) { if (v > 1000) return false; return v < lim; }\n"
		  "[numthreads(8,1,1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		  "	uint s = id.x;\n"
		  "	while (keep_going(s, 90)) s = s * 3 + 1;\n"
		  "	o[id.x] = s;\n"
		  "}\n",
		  { 2, 1, 0, 0, 1, 0, 0, 0 }, svsl_opt_none }, // outer loop + the call's wrapper; its `if` keeps a selection
		{ "for condition inlining an early-return call",
		  "RWStructuredBuffer<uint> o : register(u0);\n"
		  "uint limit(uint v) { if (v > 5) return 5; return v + 2; }\n"
		  "[numthreads(8,1,1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		  "	uint s = 0;\n"
		  "	for (uint i = 0; i < limit(id.x); i++) s += i;\n"
		  "	o[id.x] = s;\n"
		  "}\n",
		  { 2, 1, 0, 0, 1, 0, 0, 0 }, svsl_opt_none },
		{ "nested loops both exit from their headers",
		  LOOP_SRC("for (uint p = 0; p < 3; p++) for (uint q = 0; q <= p; q++) s += p * 4 + q;"),
		  { 2, 2, 0, 0, 0, 0, 0, 0 }, svsl_opt_none },
		{ "mid-body break stays a selection",
		  LOOP_SRC("for (uint n = 0; n < 10; n++) { s += n; if (s > t * 3) break; }"),
		  { 1, 1, 0, 0, 1, 0, 0, 0 }, svsl_opt_none },
	};
	#undef LOOP_SRC
	for (int32_t c = 0; c < (int32_t)(sizeof(cases) / sizeof(cases[0])); c++) {
		for (int32_t level = cases[c].min_level; level <= svsl_opt_default; level++) {
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
	TEST_CHECK(count_body_op(fn, svsl_ir_store) == 0); // unread afterwards: dead
	TEST_CHECK(count_body_op(fn, svsl_ir_ptr)   == 0); // and the pointer with them

	r = run_ir(&arena,
		"static float c = 1;\n"
		"float4 ps(float x : TEXCOORD0) : SV_TARGET { [branch] if (x > 0) c = 5; return c; }\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	int32_t end = find_op(fn, svsl_ir_end_if), load = -1;
	for (int32_t i = end + 1; i < fn->insts.count; i++)
		if (fn->insts.items[i].op == svsl_ir_load) { load = i; break; }
	TEST_CHECK(end >= 0 && load > end); // c is re-read after the merge, not assumed 1 or 5
	TEST_CHECK(count_body_op(fn, svsl_ir_ptr) == 1); // one canonical pointer for c
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

// a scalar constant of `scalar` type with exactly `bits` (svsl_ir_const encoding)
static bool has_const(const svsl_ir_func_t *fn, const svsl_program_t *prog, svsl_scalar_ scalar, uint64_t bits) {
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op != svsl_ir_const) continue;
		const svsl_type_t *t = svsl_type_get(&prog->types, in->type);
		if (t->kind == svsl_type_scalar && t->scalar == scalar &&
		    ((uint64_t)in->args[0] | ((uint64_t)in->args[1] << 32)) == bits) return true;
	}
	return false;
}

// Item 1 (docs/PLAN_optimizer_llvm.md): lane-wise constant folding of compares,
// bit ops, shifts, selects, vectors and exact intrinsics - and the cases whose
// result SPIR-V leaves open, which must stay runtime ops.
static void test_ir_fold_lanes(void) {
	svsl_arena_t arena = {0};

	// sk_texenc's bit writer with constant pos/count: the word/shift/mask math and
	// the whole select ladder fold, leaving only the value's own masking
	ir_run_t r = run_ir(&arena,
		"RWStructuredBuffer<uint4> o : register(u0);\n"
		"void write_bits(inout uint4 block, uint pos, uint count, uint value) {\n"
		"	uint word  = pos >> 5;\n"
		"	uint shift = pos & 31u;\n"
		"	uint mask  = (count >= 32u) ? 0xFFFFFFFFu : ((1u << count) - 1u);\n"
		"	value &= mask;\n"
		"	uint lo = value << shift;\n"
		"	uint hi = (shift + count > 32u) ? value >> (32u - shift) : 0u;\n"
		"	block |= uint4(word == 0u ? lo : 0u,\n"
		"	               word == 1u ? lo : (word == 0u ? hi : 0u),\n"
		"	               word == 2u ? lo : (word == 1u ? hi : 0u),\n"
		"	               word == 3u ? lo : (word == 2u ? hi : 0u));\n"
		"}\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint4 b = 0;\n"
		"	write_bits(b, 0, 11, id.x);\n"
		"	write_bits(b, 40, 7, id.y);\n"
		"	o[0] = b;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_select) == 0);
	TEST_CHECK(count_op(fn, svsl_ir_eq)     == 0);
	TEST_CHECK(count_op(fn, svsl_ir_ge)     == 0);
	TEST_CHECK(count_op(fn, svsl_ir_gt)     == 0);
	TEST_CHECK(count_op(fn, svsl_ir_shr)    == 0);
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_uint32, 0x7FF)); // (1 << 11) - 1
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_uint32, 0x7F));

	// shifts, signed math, conversions and compares: one constant, no runtime op
	r = run_ir(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	uint x = (0xF0u >> 4) + (1u << 31) + uint(-3 * -4) + (7u % 4u);\n"
		"	bool k = (x != 7u) && !(x == 3u) && (-1 < 0);\n"
		"	return k ? float(x) : 0.0;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_ne) == 0 && count_op(fn, svsl_ir_eq) == 0 && count_op(fn, svsl_ir_lt) == 0);
	TEST_CHECK(count_op(fn, svsl_ir_select) == 0);
	// float compares fold with emit's semantics: != unordered (NaN -> true), the
	// rest ordered (NaN -> false)
	uint64_t nan = 0x7FC00000u, one = 0x3F800000u, out = 2;
	TEST_CHECK(svsl_ir_eval_binary(svsl_ir_ne, svsl_scalar_float32, nan, nan, &out) && out == 1);
	TEST_CHECK(svsl_ir_eval_binary(svsl_ir_eq, svsl_scalar_float32, nan, nan, &out) && out == 0);
	TEST_CHECK(svsl_ir_eval_binary(svsl_ir_lt, svsl_scalar_float32, nan, one, &out) && out == 0);
	TEST_CHECK(svsl_ir_eval_binary(svsl_ir_ge, svsl_scalar_float32, nan, one, &out) && out == 0);
	float want = (float)(15u + 0x80000000u + 12u + 3u);
	uint32_t want_bits;
	memcpy(&want_bits, &want, 4);
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_float32, want_bits));

	// left alone: a shift by the full width and INT_MIN / -1 (SPIR-V leaves the
	// result open) - folding would pick an answer the GPU needn't
	r = run_ir(&arena,
		"float4 ps(float2 uv : TEXCOORD0) : SV_TARGET {\n"
		"	int   s = 32;\n"
		"	uint  a = 5u << s;\n"
		"	int   m = (-2147483647 - 1) / -1;\n"
		"	return float4(a, m, 0, 0) + uv.x;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_shl) == 1);
	TEST_CHECK(count_op(fn, svsl_ir_div) == 1);

	// a vector result folds to a constant composite: emit writes one
	// OpConstantComposite, never a runtime OpCompositeConstruct or OpIMul
	test_spv_t spv;
	test_spv_compile(&arena,
		"RWStructuredBuffer<uint4> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs() { o[0] = uint4(1, 2, 3, 4) * 2u + uint4(1, 1, 1, 1); }\n",
		NULL, NULL, svsl_opt_default, &spv);
	TEST_CHECK(spv.ok && spv.blob_count == 1 && test_spv_validate(&spv));
	if (spv.ok && spv.blob_count == 1) {
		TEST_CHECK(test_spv_count(&spv.blobs[0], SpvOpConstantComposite, -1, -1) >= 1);
		TEST_CHECK(test_spv_count(&spv.blobs[0], SpvOpCompositeConstruct, -1, -1) == 0);
		TEST_CHECK(test_spv_count(&spv.blobs[0], SpvOpIMul, -1, -1) == 0);
	}
	svsl_arena_free(&arena);
}

// Item 2 (docs/PLAN_optimizer_llvm.md): structured CFG simplification.
static void test_ir_cfg(void) {
	svsl_arena_t arena = {0};

	// a constant argument folds the early return; the inline wrapper loop is then
	// run-once and flattens: no loop, no break, a constant result
	ir_run_t r = run_ir(&arena,
		"float pick(float x) { if (x > 1) return 2; return x; }\n"
		"float4 ps() : SV_TARGET { return pick(0.5); }\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_loop)  == 0);
	TEST_CHECK(count_op(fn, svsl_ir_break) == 0);
	TEST_CHECK(count_op(fn, svsl_ir_if)    == 0);
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_float32, 0x3F000000)); // 0.5

	// while (false) runs zero times; while (true) { ...; break; } runs once
	r = run_ir(&arena,
		"RWStructuredBuffer<uint> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	while (false) { o[0] = 1; }\n"
		"	while (true)  { o[1] = id.x; break; }\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_loop)  == 0);
	TEST_CHECK(count_op(fn, svsl_ir_if)    == 0);
	TEST_CHECK(count_op(fn, svsl_ir_store) == 1); // only o[1] = id.x

	// a branch on a runtime value keeps its markers, and code after a return is gone
	r = run_ir(&arena,
		"RWStructuredBuffer<uint> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	float f = asfloat(id.x);\n"
		"	if (f > 0.5) { } else { o[0] = 2; }\n"
		"	if (id.y == 3) { o[1] = 1; return; o[2] = 9; }\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_if)    == 2);
	TEST_CHECK(count_op(fn, svsl_ir_store) == 2); // o[0], o[1]; the o[2] after return is gone

	// a for-loop's exit test is never inverted or folded away while it is live
	r = run_ir(&arena,
		"RWStructuredBuffer<uint> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	for (uint i = 0; i < id.x; i++) o[i] = i;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_loop) == 1);
	int32_t exit_if = find_op(fn, svsl_ir_if);
	TEST_CHECK(exit_if >= 0 && (fn->insts.items[exit_if].flags & svsl_ir_flag_loop_exit));
	// combine leaves the exit's `!(i < n)` alone (not `i >= n`): emit folds the
	// negation into the exit branch, the shape Adreno's loop analysis needs
	TEST_CHECK(exit_if >= 0 && fn->insts.items[fn->insts.items[exit_if].args[0]].op == svsl_ir_log_not);
	svsl_arena_free(&arena);
}

// Item 5 (docs/PLAN_optimizer_llvm.md): instruction combining - canonical
// order, negation and select rewrites, integer reassociation and identities,
// and composite rewrites.
static void test_ir_combine(void) {
	svsl_arena_t arena = {0};
	ir_run_t r = run_ir(&arena,
		"RWStructuredBuffer<uint4> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint i = id.x, j = id.y;\n"
		"	uint a = i * j + 3;\n"
		"	uint b = j * i + 3;\n"                 // CSE meets i*j in either order
		"	bool p = i < j, q = j > i;\n"          // one compare, mirrored
		"	uint k = ((i + 1) + 2) - 5;\n"         // one sub: i - 2
		"	uint s = (j << 2) << 3;\n"             // one shift by 5
		"	uint n = !(i < j) ? 7 : 9;\n"          // i >= j, select arms kept
		"	uint4 v = id.xyzx;\n"
		"	v = (v | 0) * 1;\n"                   // identities on vectors
		"	o[0] = uint4(a - b, (p && q) ? k : s, n, v.w);\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_mul)     == 1); // i*j and j*i: one value (so a and b are too)
	TEST_CHECK(count_op(fn, svsl_ir_sub)     == 2); // a - b, and k = i - 2
	TEST_CHECK(count_op(fn, svsl_ir_add)     == 1); // a (= b)
	TEST_CHECK(count_op(fn, svsl_ir_shl)     == 1);
	TEST_CHECK(count_op(fn, svsl_ir_lt) + count_op(fn, svsl_ir_gt) == 1);
	TEST_CHECK(count_op(fn, svsl_ir_ge)      == 1);
	TEST_CHECK(count_op(fn, svsl_ir_log_not) == 0);
	TEST_CHECK(count_op(fn, svsl_ir_bit_or)  == 0);
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_uint32, 2));
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_uint32, 5));

	// a float !(a < b) is not a >= b (NaN): the negation stays; a reversed
	// component construct is one shuffle
	r = run_ir(&arena,
		"float4 ps(float4 t : TEXCOORD0) : SV_TARGET {\n"
		"	float4 v = t * t;\n"                          // a value: components are extracts
		"	float4 w = float4(v.w, v.z, v.y, v.x);\n"
		"	return !(v.x < v.y) ? w : v;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_construct) == 0);
	TEST_CHECK(count_op(fn, svsl_ir_shuffle)   == 1);
	TEST_CHECK(count_op(fn, svsl_ir_ge)        == 0);
	TEST_CHECK(count_op(fn, svsl_ir_log_not)   == 0); // the select swapped its arms instead
	TEST_CHECK(count_op(fn, svsl_ir_lt)        == 1);
	svsl_arena_free(&arena);
}

// loads whose pointer is rooted at a local var (what forwarding failed to remove)
static int32_t count_var_loads(const svsl_ir_func_t *fn) {
	int32_t n = 0;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		if (fn->insts.items[i].op != svsl_ir_load) continue;
		uint32_t p = fn->insts.items[i].args[0];
		while (fn->insts.items[p].op == svsl_ir_chain) p = fn->insts.items[p].args[0];
		if (fn->insts.items[p].op == svsl_ir_var) n++;
	}
	return n;
}

// A local nothing has stored to yet is undef - except inside a loop that
// repeats, where its (hoisted) storage still holds the previous trip's value, as
// glslang and DXC keep it. The run-once wrapper loop of an inlined early return
// doesn't count: there the unwritten member still reads as undef (and DSE drops
// the store of it).
static void test_ir_uninitialized_locals(void) {
	svsl_arena_t arena = {0};
	ir_run_t r = run_ir(&arena,
		"RWStructuredBuffer<float> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	float sum = 0;\n"
		"	for (int i = 0; i < (int)id.x; i++) { float prev; if (i > 0) sum += prev; prev = i; }\n"
		"	o[0] = sum;\n"
		"}\n");
	TEST_CHECK(r.ok);
	TEST_CHECK(count_op(&r.module.funcs[0], svsl_ir_undef) == 0);

	r = run_ir(&arena,
		"struct S { float a; float b; };\n"
		"S make(float x) { S s; s.a = x; if (x > 1) return s; s.b = x; return s; }\n"
		"RWStructuredBuffer<float> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) { S s = make(id.x); o[0] = s.b; }\n");
	TEST_CHECK(r.ok);
	// the early return copies s.a only: s.b read undef there, and its store went
	// (then-arm a, fall-through a and b, o[0] - not a copy of the unwritten s.b)
	TEST_CHECK(count_op(&r.module.funcs[0], svsl_ir_store) == 4);
	svsl_arena_free(&arena);
}

// Item 3 (docs/PLAN_optimizer_llvm.md): CSE and forwarding scoped by arm, like
// EarlyCSE over the dominator tree.
static void test_ir_scoped(void) {
	svsl_arena_t arena = {0};

	// a value from before an if is reused in both arms and after the merge; a
	// value computed in the then arm is not visible in the else arm
	ir_run_t r = run_ir(&arena,
		"RWStructuredBuffer<float> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	float x = asfloat(id.x), y = asfloat(id.y), z = asfloat(id.z);\n"
		"	o[0] = x * y;\n"
		"	if (id.x > 3) { o[1] = x * y; o[2] = y * z; }\n"
		"	else          { o[3] = x * y; o[4] = y * z; }\n"
		"	o[5] = y * x;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_mul) == 3); // x*y once; y*z once per arm

	// forwarding inside a loop body reaches into nested arms; across the back-edge
	// it must not: acc's value at the top of the body is a merge (0 or acc + i)
	r = run_ir(&arena,
		"RWStructuredBuffer<uint> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint acc = 0;\n"
		"	for (uint i = 0; i < id.x; i++) {\n"
		"		uint t = i * 3;\n"
		"		if (i > 2) o[i] = t;\n"
		"		o[i + 64] = acc;\n"
		"		acc += i;\n"
		"	}\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	int32_t vars = count_op(fn, svsl_ir_var);
	TEST_CHECK(vars == 2);                 // i and acc; t forwarded into the if
	TEST_CHECK(count_var_loads(fn) >= 2);  // i and acc reloaded each trip

	// an else arm sees memory as before the if: k forwards to id.x there, then
	// reloads after the merge (either arm's value)
	r = run_ir(&arena,
		"RWStructuredBuffer<uint> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint k = id.x;\n"
		"	if (id.y > 3) { k = 7; o[0] = k; } else { o[1] = k; }\n"
		"	o[2] = k;\n"
		"}\n");
	TEST_CHECK(r.ok);
	fn = &r.module.funcs[0];
	TEST_CHECK(count_var_loads(fn) == 1); // only the post-merge read of k
	svsl_arena_free(&arena);
}

// Item 4 (docs/PLAN_optimizer_llvm.md): if-conversion of local-only branches.
static void test_ir_if_convert(void) {
	svsl_arena_t arena = {0};

	// both arms, one arm, and nested diamonds become selects; no var survives
	ir_run_t r = run_ir(&arena,
		"float4 ps(float4 uv : TEXCOORD0) : SV_TARGET {\n"
		"	float k = uv.x * 2, acc = 0, m = uv.z;\n"
		"	if (uv.y > 0.5) { acc = k + 1; } else { acc = k - 1; }\n"
		"	if (uv.w > 0.5) m = 99;\n"
		"	float n = 0;\n"
		"	if (uv.x > 0) { if (uv.y > 0) n = 1; else n = 2; }\n"
		"	return float4(acc, m, n, 0);\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_if)     == 0);
	TEST_CHECK(count_op(fn, svsl_ir_var)    == 0);
	TEST_CHECK(count_op(fn, svsl_ir_select) == 4); // acc, m, inner n, outer n

	// kept as branches: an arm that could fault, read out of bounds, or see other
	// invocations if run unconditionally, or that overlaps a location
	static const char *keep[] = {
		"	if (x > 0) a[i] = 1;\n",                       // dynamic index
		"	if (x > 0) s = WaveActiveSum(s);\n",            // subgroup op
		"	if (x > 0) s = s / i;\n",                      // integer division by a variable
		"	if (x > 0) f = ddx(f);\n",                     // derivative
		"	if (x > 0) f = t.Sample(ss, uv).x;\n",         // implicit-LOD sample
		"	if (x > 0) o[0] = s;\n",                       // a global store
		"	if (x > 0) v = float4(1, 2, 3, 4); else v.x = 2;\n", // whole vs member
		"	[branch] if (x > 0) s = 1;\n",                 // the author's hint
	};
	for (size_t c = 0; c < sizeof(keep) / sizeof(keep[0]); c++) {
		char src[1024];
		snprintf(src, sizeof(src),
			"Texture2D t : register(t0); SamplerState ss : register(s0);\n"
			"RWStructuredBuffer<uint> o : register(u0);\n"
			"float4 ps(float4 uv : TEXCOORD0, uint i : TEXCOORD1) : SV_TARGET {\n"
			"	float x = uv.x, f = uv.y; uint s = i; float a[4] = { 0, 0, 0, 0 }; float4 v = uv;\n"
			"%s"
			"	return float4(f + a[i & 3] + v.x, s, x, 0);\n"
			"}\n", keep[c]);
		r = run_ir(&arena, src);
		TEST_CHECK(r.ok);
		if (!r.ok) continue;
		bool kept = count_op(&r.module.funcs[0], svsl_ir_if) == 1;
		TEST_CHECK(kept);
		if (!kept) printf("  if-converted but must not be: %s", keep[c]);
	}
	svsl_arena_free(&arena);
}

// Item 6 (docs/PLAN_optimizer_llvm.md): loads from a constant table through
// constant indices read its initializer; a dynamic index stays a load.
static void test_ir_const_tables(void) {
	svsl_arena_t arena = {0};
	ir_run_t r = run_ir(&arena,
		"static const uint   T[4] = { 3u, 5u, 7u, 9u };\n"
		"static const float3 V[2] = { float3(1, 2, 3), { 4, 5.5, 6 } };\n"
		"float4 ps(uint i : TEXCOORD0) : SV_TARGET { return float4(T[2], V[1].y, V[0].z, T[i & 3]); }\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	int32_t table_loads = 0;
	for (int32_t i = 0; i < fn->insts.count; i++) {
		const svsl_ir_inst_t *in = &fn->insts.items[i];
		if (in->op != svsl_ir_load) continue;
		uint32_t p = in->args[0];
		while (fn->insts.items[p].op == svsl_ir_chain) p = fn->insts.items[p].args[0];
		if (fn->insts.items[p].op == svsl_ir_ptr && fn->insts.items[p].args[0] == svsl_ref_const_global) table_loads++;
	}
	TEST_CHECK(table_loads == 1); // only T[i & 3]
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_float32, 0x40E00000)); // T[2] = 7, as float
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_float32, 0x40B00000)); // 5.5
	TEST_CHECK(has_const(fn, &r.prog, svsl_scalar_float32, 0x40400000)); // 3.0
	svsl_arena_free(&arena);
}

// Item 7 (docs/PLAN_optimizer_llvm.md): a function-local constant table is a
// constant global, not a per-invocation variable re-stored on every call.
static void test_ir_local_const_tables(void) {
	svsl_arena_t arena = {0};
	ir_run_t r = run_ir(&arena,
		"float pick(uint i) { static const float T[3] = { 1.5, 2.5, 3.5 }; return T[i % 3]; }\n"
		"float pick(float v) { const float T[2] = { 4, 5 }; return T[uint(v) & 1]; }\n"
		"float4 ps(float4 uv : TEXCOORD0, uint k : TEXCOORD1) : SV_TARGET {\n"
		"	return float4(pick(k), pick(uv.x), pick(k + 1u), 0);\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_var) == 0); // no per-invocation copy of either table
	int32_t tables = 0;
	for (int32_t i = 0; i < r.prog.const_globals.count; i++)
		if (svsl_str_eq_cstr(r.prog.const_globals.items[i].name, "pick.T") ||
		    svsl_str_eq_cstr(r.prog.const_globals.items[i].name, "pick.T.2")) tables++;
	TEST_CHECK(tables == 2); // one per declaration (overloads numbered), not per call
	svsl_arena_free(&arena);
}

// Items 8 and 9 (docs/PLAN_optimizer_llvm.md): an [unroll] loop whose counter
// indexes a large local array unrolls, and the array splits into values; loops
// that don't serve such an array keep their shape.
static void test_ir_unroll_sroa(void) {
	svsl_arena_t arena = {0};
	#define UNROLL_SRC(decl, loop) \
		"RWStructuredBuffer<float> o : register(u0);\n" \
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n" \
		"	" decl "\n" \
		"	" loop "\n" \
		"	o[id.x] = a[id.y & 1] + a[0];\n" \
		"}\n"
	// fills and reads of a 32-element array through a counter (any step): both
	// loops unroll, and the reads after them hit constant elements
	ir_run_t r = run_ir(&arena,
		"RWStructuredBuffer<float> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	float a[32];\n"
		"	[unroll] for (int i = 31; i >= 0; i -= 1) a[i] = float(id.x) * i;\n"
		"	float s = 0;\n"
		"	[unroll] for (uint k = 0; k < 32; k += 2) s += a[k] - a[k + 1];\n"
		"	o[id.x] = s;\n"
		"}\n");
	TEST_CHECK(r.ok);
	const svsl_ir_func_t *fn = &r.module.funcs[0];
	TEST_CHECK(count_op(fn, svsl_ir_loop) == 0);
	TEST_CHECK(count_op(fn, svsl_ir_var)  == 0); // a split into values, counters forwarded away

	// a nested array counts every level (8x8 = 64 elements), and every array-level
	// index's loop is needed: the inner and outer loops both unroll
	r = run_ir(&arena,
		"RWStructuredBuffer<float> o : register(u0);\n"
		"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	float g[8][8];\n"
		"	[unroll] for (int y = 0; y < 8; y++) [unroll] for (int x = 0; x < 8; x++) g[y][x] = float(id.x * y + x);\n"
		"	float s = 0;\n"
		"	[unroll] for (int d = 0; d < 8; d++) s += g[d][7 - d];\n"
		"	o[id.x] = s;\n"
		"}\n");
	TEST_CHECK(r.ok && count_op(&r.module.funcs[0], svsl_ir_loop) == 0 && count_op(&r.module.funcs[0], svsl_ir_var) == 0);

	// more counted loops than one 64-bit mask: every one is tracked (the 6x6
	// encoder has 70), so the array they all index still splits
	{
		char src[16384];
		int32_t len = snprintf(src, sizeof(src),
			"RWStructuredBuffer<uint> o : register(u0);\n"
			"[numthreads(1, 1, 1)] void cs(uint3 id : SV_DispatchThreadID) {\n"
			"	uint a[32];\n"
			"	[unroll] for (uint i = 0; i < 32; i++) a[i] = id.x + i;\n");
		for (int32_t l = 0; l < 70; l++)
			len += snprintf(src + len, sizeof(src) - (size_t)len,
				"	[unroll] for (uint i%d = 0; i%d < 32; i%d += 8) a[i%d] += %du;\n", l, l, l, l, l);
		snprintf(src + len, sizeof(src) - (size_t)len, "	o[id.x] = a[0] + a[8] + a[16] + a[24];\n}\n");
		r = run_ir(&arena, src);
		TEST_CHECK(r.ok && count_op(&r.module.funcs[0], svsl_ir_loop) == 0 && count_op(&r.module.funcs[0], svsl_ir_var) == 0);
	}

	// kept rolled: no [unroll], a small array, a data-dependent index anywhere,
	// a break, and a counter the body writes
	static const struct { const char *why, *decl, *loop; } keep[] = {
		{ "no [unroll]",   "float a[32];", "for (uint i = 0; i < 32; i++) a[i] = i;" },
		{ "[loop]",        "float a[32];", "[loop] for (uint i = 0; i < 32; i++) a[i] = i;" },
		{ "small array",   "float a[8];",  "[unroll] for (uint i = 0; i < 8; i++) a[i] = i;" },
		{ "runtime index", "float a[32];", "[unroll] for (uint i = 0; i < 32; i++) a[(i * id.z) & 31] = i;" },
		{ "break",         "float a[32];", "[unroll] for (uint i = 0; i < 32; i++) { if (i == id.z) break; a[i] = i; }" },
		{ "counter write", "float a[32];", "[unroll] for (uint i = 0; i < 32; i++) { a[i] = i; i += id.z & 1; }" },
	};
	for (size_t c = 0; c < sizeof(keep) / sizeof(keep[0]); c++) {
		char src[1024];
		snprintf(src, sizeof(src), UNROLL_SRC("%s", "%s"), keep[c].decl, keep[c].loop);
		r = run_ir(&arena, src);
		TEST_CHECK(r.ok);
		bool rolled = r.ok && count_op(&r.module.funcs[0], svsl_ir_loop) == 1;
		TEST_CHECK(rolled);
		if (!rolled) printf("  unrolled but must not be (%s)\n", keep[c].why);
	}
	#undef UNROLL_SRC
	svsl_arena_free(&arena);
}

// The verifier (ir_verify.h) accepts optimized output and rejects each kind of
// broken IR - built by hand, since no pass should ever produce one.
static void test_ir_verify(void) {
	svsl_arena_t arena = {0};
	ir_run_t r = run_ir(&arena,
		"float4 ps(float4 uv : TEXCOORD0) : SV_TARGET { float k = 0; if (uv.x > 0) k = uv.y * 2; return k; }\n");
	TEST_CHECK(r.ok);
	svsl_ir_verify_error_t err;
	TEST_CHECK(svsl_ir_verify(&arena, &r.module.funcs[0], &err));

	svsl_type_id_t f32  = svsl_type_scalar_id(&r.prog.types, svsl_scalar_float32);
	svsl_type_id_t bool_ = svsl_type_scalar_id(&r.prog.types, svsl_scalar_bool);
	#define I(o, t, a0, a1) ((svsl_ir_inst_t){ .op = (o), .type = (t), .args = { (a0), (a1), 0, SVSL_IR_NONE } })
	#define N SVSL_TYPE_NONE
	static const struct { const char *what; svsl_ir_inst_t code[8]; int32_t count; } bad[] = {
		{ "escapes", { I(svsl_ir_const, 0, 0, 0), I(svsl_ir_const, 1, 1, 0), I(svsl_ir_if, N, 1, 0),
		               I(svsl_ir_add, 0, 0, 0), I(svsl_ir_end_if, N, 0, 0), I(svsl_ir_neg, 0, 3, 0),
		               I(svsl_ir_return, N, SVSL_IR_NONE, 0) }, 7 },
		{ "earlier",  { I(svsl_ir_const, 0, 0, 0), I(svsl_ir_add, 0, 0, 2), I(svsl_ir_const, 0, 0, 0),
		               I(svsl_ir_return, N, 1, 0) }, 4 },
		{ "killed",   { I(svsl_ir_const, 0, 0, 0), I(svsl_ir_nop, N, 0, 0), I(svsl_ir_add, 0, 0, 1),
		               I(svsl_ir_return, N, 2, 0) }, 4 },
		{ "outside",  { I(svsl_ir_const, 1, 1, 0), I(svsl_ir_if, N, 0, 0), I(svsl_ir_end_loop, N, SVSL_IR_NONE, 0),
		               I(svsl_ir_return, N, SVSL_IR_NONE, 0) }, 4 },
		{ "outside",  { I(svsl_ir_break, N, 0, 0), I(svsl_ir_return, N, SVSL_IR_NONE, 0) }, 2 },
		{ "unclosed", { I(svsl_ir_const, 1, 1, 0), I(svsl_ir_if, N, 0, 0), I(svsl_ir_return, N, SVSL_IR_NONE, 0) }, 3 },
	};
	#undef I
	#undef N
	for (size_t c = 0; c < sizeof(bad) / sizeof(bad[0]); c++) {
		svsl_ir_func_t fn = {0};
		for (int32_t i = 0; i < bad[c].count; i++) {
			svsl_ir_inst_t in = bad[c].code[i];
			if (in.type == 0) in.type = f32;          // 0 = float, 1 = bool in the table above
			else if (in.type == 1) in.type = bool_;
			svsl_array_push(&arena, &fn.insts, in);
		}
		bool ok = svsl_ir_verify(&arena, &fn, &err);
		TEST_CHECK(!ok && strstr(err.what, bad[c].what));
		if (ok || !strstr(err.what, bad[c].what)) printf("  verify case %zu: expected '%s', got '%s'\n", c, bad[c].what, ok ? "ok" : err.what);
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
	test_ir_fold_lanes();
	test_ir_cfg();
	test_ir_combine();
	test_ir_scoped();
	test_ir_uninitialized_locals();
	test_ir_if_convert();
	test_ir_const_tables();
	test_ir_local_const_tables();
	test_ir_unroll_sroa();
	test_ir_verify();
}

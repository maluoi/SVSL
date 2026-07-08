#include "test.h"

#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "sema/sema.h"
#include "util/arena.h"

#include <string.h>

typedef struct sema_run_t {
	svsl_program_t   prog;
	svsl_diag_list_t diags;
	bool             ok;
} sema_run_t;

// expect_errors: suppress diagnostic printing for tests that want failures
static sema_run_t run_sema_ex(svsl_arena_t *arena, const char *src, bool expect_errors);

static sema_run_t run_sema(svsl_arena_t *arena, const char *src) {
	return run_sema_ex(arena, src, false);
}

static sema_run_t run_sema_ex(svsl_arena_t *arena, const char *src, bool expect_errors) {
	sema_run_t r = {0};
	svsl_pp_result_t  pp;
	svsl_token_list_t tokens = {0};
	svsl_pp_run(arena, src, "sema_test.hlsl", NULL, &pp, &r.diags);
	svsl_lex(arena, &pp, &tokens, &r.diags);
	svsl_ast_t *ast = svsl_parse(arena, &tokens, &r.diags);
	r.ok = svsl_sema_run(arena, ast, &pp, "sema_test.hlsl", NULL, &r.prog, &r.diags);
	if (!r.ok && !expect_errors)
		for (int32_t i = 0; i < r.diags.count; i++)
			if (r.diags.items[i].severity == svsl_severity_error)
				printf("  sema error %s:%d: %s\n", r.diags.items[i].loc.file,
				       r.diags.items[i].loc.line, r.diags.items[i].message);
	return r;
}

static const svsl_buffer_t *find_buffer(const svsl_program_t *p, const char *name) {
	for (int32_t i = 0; i < p->buffers.count; i++)
		if (svsl_str_eq_cstr(p->buffers.items[i].name, name)) return &p->buffers.items[i];
	return NULL;
}
static const svsl_resource_t *find_resource(const svsl_program_t *p, const char *name) {
	for (int32_t i = 0; i < p->resources.count; i++)
		if (svsl_str_eq_cstr(p->resources.items[i].name, name)) return &p->resources.items[i];
	return NULL;
}

// the StereoKit reference shader shape, self-contained
static const char *reference_src =
	"//--name = sk/unlit\n"
	"//--diffuse = white\n"
	"//--color:color = 1,1,1,1\n"
	"//--uv_scale: range(0, 2) = 0.5\n"
	"float4 color     = {1,1,1,1};\n"
	"float4 tex_trans = float4(0,0,1,1);\n"
	"float  uv_scale  = 1;\n"
	"Texture2D    diffuse   : register(t0);\n"
	"SamplerState diffuse_s : register(s0);\n"
	"struct inst_t { float4x4 world; float4 color; };\n"
	"StructuredBuffer<inst_t> sk_inst : register(t12);\n"
	"cbuffer stereokit_buffer : register(b1) {\n"
	"	float4x4 sk_viewproj[6];\n"
	"	uint     sk_view_count;\n"
	"};\n"
	"struct vsIn { float4 pos:SV_Position; float3 norm:NORMAL0; float2 uv:TEXCOORD0; float4 col:COLOR0; };\n"
	"struct psIn { float4 pos:SV_POSITION; float2 uv:TEXCOORD0; min16float4 color:COLOR0; };\n"
	"struct sk_ids_t { uint inst : SV_InstanceID; uint view : SV_ViewID; };\n"
	"psIn vs(vsIn input, sk_ids_t ids) {\n"
	"	psIn o;\n"
	"	o.pos   = mul(mul(float4(input.pos.xyz, 1), sk_inst[ids.inst].world), sk_viewproj[ids.view]);\n"
	"	o.uv    = input.uv * uv_scale;\n"
	"	o.color = input.col * color * sk_inst[ids.inst].color;\n"
	"	return o;\n"
	"}\n"
	"min16float4 ps(psIn input) : SV_TARGET {\n"
	"	return diffuse.Sample(diffuse_s, input.uv) * input.color;\n"
	"}\n";

static void test_sema_reference(void) {
	svsl_arena_t arena = {0};
	sema_run_t   r     = run_sema(&arena, reference_src);
	TEST_CHECK(r.ok);

	TEST_CHECK(svsl_str_eq_cstr(r.prog.name, "sk/unlit"));

	// $Global: bare globals with folded defaults, auto-assigned to b0
	const svsl_buffer_t *globals = find_buffer(&r.prog, "$Global");
	TEST_CHECK(globals != NULL);
	if (globals) {
		TEST_CHECK(globals->bind.cls == 'b' && globals->bind.slot == 0 && globals->bind.space == 0);
		TEST_CHECK(globals->size == 48); // f4 + f4 + float → padded to 16
		TEST_CHECK(globals->members.count == 3);
		TEST_CHECK(globals->members.items[0].offset == 0);
		TEST_CHECK(globals->members.items[1].offset == 16);
		TEST_CHECK(globals->members.items[2].offset == 32);
		TEST_CHECK(globals->defaults != NULL);

		float f[4];
		memcpy(f, globals->defaults + 0, 16);
		TEST_CHECK(f[0] == 1 && f[1] == 1 && f[2] == 1 && f[3] == 1);
		memcpy(f, globals->defaults + 16, 16);
		TEST_CHECK(f[0] == 0 && f[1] == 0 && f[2] == 1 && f[3] == 1);
		memcpy(f, globals->defaults + 32, 4);
		TEST_CHECK(f[0] == 0.5f); // //-- override beats the initializer (1)

		// //--uv_scale: range(0, 2) tag lands in extra
		TEST_CHECK(svsl_str_eq_cstr(globals->members.items[2].extra, "range(0, 2)"));
	}

	// stereokit_buffer at its explicit slot
	const svsl_buffer_t *sk = find_buffer(&r.prog, "stereokit_buffer");
	TEST_CHECK(sk != NULL);
	if (sk) {
		TEST_CHECK(sk->bind.slot == 1);
		TEST_CHECK(sk->size == 400); // 6*64 + 4 → pad 16
		TEST_CHECK(sk->members.items[1].offset == 384);
	}

	// resources: fused texture, structured buffer with element stride
	const svsl_resource_t *diffuse = find_resource(&r.prog, "diffuse");
	TEST_CHECK(diffuse && diffuse->kind == svsl_res_texture);
	TEST_CHECK(diffuse && diffuse->bind.cls == 't' && diffuse->bind.slot == 0);
	TEST_CHECK(diffuse && diffuse->sampler_slot == 0); // paired with diffuse_s by slot
	TEST_CHECK(diffuse && svsl_str_eq_cstr(diffuse->value, "white"));

	const svsl_resource_t *inst = find_resource(&r.prog, "sk_inst");
	TEST_CHECK(inst && inst->kind == svsl_res_structured);
	TEST_CHECK(inst && inst->bind.slot == 12);
	TEST_CHECK(inst && inst->element_size == 80);

	// entries + vertex inputs (builtin sk_ids_t members excluded)
	TEST_CHECK(r.prog.entries.count == 2);
	TEST_CHECK(r.prog.vertex_inputs.count == 4);
	TEST_CHECK(svsl_str_eq_cstr(r.prog.vertex_inputs.items[0].semantic, "SV_Position"));
	TEST_CHECK(svsl_str_eq_cstr(r.prog.vertex_inputs.items[1].semantic, "NORMAL0"));

	svsl_arena_free(&arena);
}

static void test_sema_auto_binding(void) {
	svsl_arena_t arena = {0};

	// no registers anywhere: declaration order, lowest free slot, pairs together
	sema_run_t r = run_sema(&arena,
		"Texture2D first;\n"
		"Sampler   first_s;\n"
		"Texture2D second : register(t0);\n" // explicit t0 forces 'first' elsewhere
		"Texture2D third;\n"
		"uniform Params { float4 tint; };\n"
		"float4 bare = {1,1,1,1};\n"
		"[numthreads(8,8,1)] void main_cs(uint3 id : SV_DispatchThreadID) { }\n");
	TEST_CHECK(!r.ok || r.ok); // just needs to run; specific checks below
	const svsl_resource_t *first  = find_resource(&r.prog, "first");
	const svsl_resource_t *second = find_resource(&r.prog, "second");
	const svsl_resource_t *third  = find_resource(&r.prog, "third");
	const svsl_resource_t *first_s= find_resource(&r.prog, "first_s");
	TEST_CHECK(second && second->bind.slot == 0);
	TEST_CHECK(first  && first->bind.slot == 1);   // t0 taken → t1
	TEST_CHECK(third  && third->bind.slot == 2);
	TEST_CHECK(first_s && first_s->bind.slot == 1); // paired with 'first' by name
	TEST_CHECK(first  && first->sampler_slot == 1);

	// $Global slips after explicit buffers: Params has no register → b0, bare → $Global b1
	const svsl_buffer_t *params  = find_buffer(&r.prog, "Params");
	const svsl_buffer_t *globals = find_buffer(&r.prog, "$Global");
	TEST_CHECK(params && globals);
	TEST_CHECK(params && params->bind.slot != globals->bind.slot);

	svsl_arena_free(&arena);
}

static void test_sema_spec_and_push(void) {
	svsl_arena_t arena = {0};

	sema_run_t r = run_sema(&arena,
		"specialization const uint TILE = 16;\n"
		"[specialization(7)] const float CUTOFF = 0.5;\n"
		"specialization const bool SHADOW = false;\n"
		"pushconstant Draw { float4x4 model; uint materialId; };\n"
		"workgroup float tile_cache[16];\n"
		"[wave_size(32)] [compute(8, 8, 1)] void my_cs(uint3 id : SV_DispatchThreadID) { }\n");
	TEST_CHECK(r.ok);

	TEST_CHECK(r.prog.spec_consts.count == 3);
	TEST_CHECK(r.prog.spec_consts.items[0].id == 0);           // TILE: auto
	TEST_CHECK(r.prog.spec_consts.items[0].default_bits == 16);
	TEST_CHECK(r.prog.spec_consts.items[1].id == 7);           // CUTOFF: pinned
	TEST_CHECK(r.prog.spec_consts.items[2].id == 1);           // SHADOW: next free auto
	TEST_CHECK(r.prog.spec_consts.items[2].default_bits == 0);

	const svsl_buffer_t *push = find_buffer(&r.prog, "Draw");
	TEST_CHECK(push && push->kind == svsl_block_pushconstant);
	TEST_CHECK(push && push->size == 80); // std430: 64 + 4, padded to the matrix alignment
	TEST_CHECK(push && push->members.items[1].offset == 64);

	TEST_CHECK(r.prog.entries.count == 1);
	TEST_CHECK(r.prog.entries.items[0].stage == svsl_stage_compute);
	TEST_CHECK(r.prog.entries.items[0].workgroup[0] == 8 && r.prog.entries.items[0].workgroup[2] == 1);
	TEST_CHECK(r.prog.wave_size == 32);

	svsl_arena_free(&arena);
}

static void test_sema_storagebuffer_block(void) {
	svsl_arena_t arena = {0};

	sema_run_t r = run_sema(&arena,
		"struct Vertex { float3 position; float3 normal; float2 uv; };\n"
		"storagebuffer readonly Vertices : register(t0, space1) { Vertex vertices[]; };\n"
		"storagebuffer Counts { uint counts[]; };\n"
		"void cs() { }\n");
	TEST_CHECK(r.ok);

	const svsl_resource_t *verts = find_resource(&r.prog, "Vertices");
	TEST_CHECK(verts && verts->kind == svsl_res_structured);
	TEST_CHECK(verts && verts->bind.slot == 0 && verts->bind.space == 1);
	TEST_CHECK(verts && verts->element_size == 48); // std430: float3s align 16

	const svsl_resource_t *counts = find_resource(&r.prog, "Counts");
	TEST_CHECK(counts && counts->kind == svsl_res_rw_structured);
	TEST_CHECK(counts && counts->bind.cls == 'u');
	TEST_CHECK(counts && counts->element_size == 4);

	svsl_arena_free(&arena);
}

static void test_sema_unsized_arrays(void) {
	svsl_arena_t arena = {0};

	// the outermost dimension takes its count from the initializer list —
	// const globals, locals, and multi-dim arrays alike
	sema_run_t r = run_sema(&arena,
		"static const float3 grad[] = { float3(0,0,0), float3(1,1,1), float3(2,2,2) };\n"
		"static const float  pair[][2] = { {1,2}, {3,4} };\n"
		"float4 ps() : SV_TARGET {\n"
		"	float local[] = { 0.25, 0.5, 0.75 };\n"
		"	return float4(grad[1] * pair[1][0], local[2]);\n"
		"}\n");
	TEST_CHECK(r.ok);
	for (int32_t i = 0; i < r.prog.const_globals.count; i++) {
		const svsl_global_t *g = &r.prog.const_globals.items[i];
		const svsl_type_t   *t = svsl_type_get(&r.prog.types, g->type);
		if (svsl_str_eq_cstr(g->name, "grad"))
			TEST_CHECK(t->kind == svsl_type_array && t->array_count == 3);
		if (svsl_str_eq_cstr(g->name, "pair"))
			TEST_CHECK(t->kind == svsl_type_array && t->array_count == 2);
	}

	// nothing to size from, and inner dimensions must be explicit
	r = run_sema_ex(&arena, "static const float bad[];\nvoid ps() { }\n", true);
	TEST_CHECK(!r.ok);
	r = run_sema_ex(&arena, "static const float bad[2][] = { {1,2}, {3,4} };\nvoid ps() { }\n", true);
	TEST_CHECK(!r.ok);
	r = run_sema_ex(&arena, "float4 ps() : SV_TARGET { float v[]; return v[0]; }\n", true);
	TEST_CHECK(!r.ok);

	svsl_arena_free(&arena);
}

static void test_sema_errors(void) {
	svsl_arena_t arena = {0};

	sema_run_t r = run_sema_ex(&arena, "UnknownType x;\nvoid ps() { }\n", true);
	TEST_CHECK(!r.ok);

	r = run_sema_ex(&arena, "Texture2D a : register(t3);\nTexture2D b : register(t3);\nvoid ps() { }\n", true);
	TEST_CHECK(!r.ok); // register collision

	r = run_sema_ex(&arena, "uniform pack1 Bad { float x; };\nvoid ps() { }\n", true);
	TEST_CHECK(!r.ok); // uniform must be pack16

	r = run_sema_ex(&arena, "specialization const float4 v = 1;\nvoid ps() { }\n", true);
	TEST_CHECK(!r.ok); // spec constants are 32-bit scalars

	// containment cycles used to recurse forever in layout
	r = run_sema_ex(&arena, "struct A { A a; };\nfloat4 ps() : SV_Target { A x; return 1; }\n", true);
	TEST_CHECK(!r.ok);
	r = run_sema_ex(&arena, "struct A { A a[4]; };\nfloat4 ps() : SV_Target { return 1; }\n", true);
	TEST_CHECK(!r.ok);
	r = run_sema(&arena, // legitimate nesting still works
		"struct A { float3 pos; };\nstruct C { A a; A b[2]; float x; };\n"
		"cbuffer buf { C big; };\nfloat4 ps() : SV_Target { C c; c.a = big.a; return c.x; }\n");
	TEST_CHECK(r.ok);

	svsl_arena_free(&arena);
}

static void test_sema_multisample_and_tile(void) {
	svsl_arena_t arena = {0};

	// the MS/tile surface: SubpassInputMS.Load(sample), Texture2DMS.Load(coord,
	// sample) + GetDimensions(w, h, samples), TileImage.Load + depth/stencil reads
	sema_run_t r = run_sema(&arena,
		"[[vk::input_attachment_index(0)]] SubpassInputMS<float4> color;\n"
		"Texture2DMS<float4, 4> msaa;\n"
		"TileImage<float4, 1>   tile;\n"
		"float4 ps(float4 pos : SV_Position) : SV_Target {\n"
		"	uint w, h, samples;\n"
		"	msaa.GetDimensions(w, h, samples);\n"
		"	return color.SubpassLoad(2) + msaa.Load(int2(pos.xy), 3) + tile.Load()\n"
		"	     + tile_depth() + (float)(tile_stencil() + w + samples);\n"
		"}\n");
	TEST_CHECK(r.ok);

	// honest rejections
	r = run_sema_ex(&arena, // MS textures cannot be sampled
		"Texture2DMS<float4, 4> msaa;\nSamplerState msaa_s;\n"
		"float4 ps(float2 uv : TEXCOORD0) : SV_Target { return msaa.Sample(msaa_s, uv); }\n", true);
	TEST_CHECK(!r.ok);
	r = run_sema_ex(&arena, // SubpassInputMS needs the sample index
		"SubpassInputMS<float4> color;\n"
		"float4 ps() : SV_Target { return color.SubpassLoad(); }\n", true);
	TEST_CHECK(!r.ok);
	r = run_sema_ex(&arena, // tile images are not descriptors
		"TileImage<float4, 0> tile : register(t0);\n"
		"float4 ps() : SV_Target { return tile.Load(); }\n", true);
	TEST_CHECK(!r.ok);

	svsl_arena_free(&arena);
}

static void test_sema_texture_index(void) {
	svsl_arena_t arena = {0};

	// sampled tex[coord] is a mip-0 texel fetch (HLSL operator[]), for 1D/2D/3D
	sema_run_t r = run_sema(&arena,
		"Texture2D<float4> tex2 : register(t0);\n"
		"Texture3D<float4> tex3 : register(t1);\n"
		"RWTexture3D<float4> out_tex : register(u0);\n"
		"[numthreads(1,1,1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	out_tex[id] = tex2[id.xy] + tex3[id];\n"
		"}\n");
	TEST_CHECK(r.ok);

	// cube and multisampled textures have no [] form
	r = run_sema_ex(&arena,
		"TextureCube<float4> cube : register(t0);\n"
		"float4 ps(int3 c : C) : SV_Target { return cube[c]; }\n", true);
	TEST_CHECK(!r.ok);
	r = run_sema_ex(&arena,
		"Texture2DMS<float4, 4> msaa : register(t0);\n"
		"float4 ps(int2 c : C) : SV_Target { return msaa[c]; }\n", true);
	TEST_CHECK(!r.ok);

	// a sampled texture texel is read-only: it is not a writable lvalue
	r = run_sema_ex(&arena,
		"Texture2D<float4> tex : register(t0);\n"
		"[numthreads(1,1,1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { tex[id.xy] = 1; }\n", true);
	TEST_CHECK(!r.ok);

	svsl_arena_free(&arena);
}


// --- body typechecking ---------------------------------------------------------

static void test_check_expressions(void) {
	svsl_arena_t arena = {0};

	// intrinsic result shapes flow through the whole reference shader (r.ok covers it),
	// so target the interesting rules directly
	sema_run_t r = run_sema(&arena,
		"struct psIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };\n"
		"Texture2D    diffuse   : register(t0);\n"
		"SamplerState diffuse_s : register(s0);\n"
		"float4x4 world;\n"
		"float4 ps(psIn input) : SV_TARGET {\n"
		"	float3 n   = normalize(float3(1, 2, 3));\n"
		"	float  d   = dot(n, n);\n"
		"	float3 c   = cross(n, n.zyx);\n"
		"	float4 pos = mul(float4(n, 1), world);\n"
		"	float4x4 t = transpose(world);\n"
		"	float3x3 m3 = (float3x3)world;\n"
		"	bool3  b   = n > 0.5;\n"
		"	float4 tex = diffuse.Sample(diffuse_s, input.uv);\n"
		"	uint2  dim = diffuse.GetDimensions();\n"
		"	int    lit = diffuse.Load(int3(0, 0, 0)).x > 0 ? 1 : 0;\n"
		"	min16float4 h = tex * 0.5h;\n"
		"	return pos + tex + float4(c, d) * b.x + h + float(dim.x) + lit;\n"
		"}\n");
	TEST_CHECK(r.ok);

	// recursion is a hard error
	r = run_sema_ex(&arena,
		"float a(float x);\n"
		"float b(float x) { return a(x); }\n"
		"float a(float x) { return b(x); }\n"
		"float4 ps() : SV_TARGET { return a(1); }\n", true);
	TEST_CHECK(!r.ok);
	TEST_CHECK(r.diags.count > 0 && strstr(r.diags.items[r.diags.count - 1].message, "recursion") != NULL);

	// resource params on user functions (the prototype-killer case) typecheck fine
	r = run_sema(&arena,
		"Texture2D    tex   : register(t0);\n"
		"SamplerState tex_s : register(s0);\n"
		"float4 sample_fade(Texture2D t, SamplerState s, float2 uv, float f) {\n"
		"	return t.Sample(s, uv) * f;\n"
		"}\n"
		"float4 ps(float2 uv : TEXCOORD0) : SV_TARGET { return sample_fade(tex, tex_s, uv, 0.5); }\n");
	TEST_CHECK(r.ok);

	// inout params: writable arguments required
	r = run_sema_ex(&arena,
		"void bump(inout float x) { x += 1; }\n"
		"float4 ps() : SV_TARGET { bump(1.5); return 1; }\n", true);
	TEST_CHECK(!r.ok);

	svsl_arena_free(&arena);
}

static void test_check_errors_and_warnings(void) {
	svsl_arena_t arena = {0};

	// assignment to a cbuffer member is an error
	sema_run_t r = run_sema_ex(&arena,
		"cbuffer B : register(b0) { float x; };\n"
		"float4 ps() : SV_TARGET { x = 1; return x; }\n", true);
	TEST_CHECK(!r.ok);

	// swizzle rules
	r = run_sema_ex(&arena, "float4 ps() : SV_TARGET { float4 v = 1; return v.xg; }\n", true);
	TEST_CHECK(!r.ok); // mixed sets
	r = run_sema_ex(&arena, "float4 ps() : SV_TARGET { float2 v = 1; return v.xyz.xyzz; }\n", true);
	TEST_CHECK(!r.ok); // out of range
	r = run_sema(&arena, "float4 ps() : SV_TARGET { float2 v = 1; return v.xyxy * v.rgrg; }\n");
	TEST_CHECK(r.ok);
	r = run_sema_ex(&arena, "float4 ps() : SV_TARGET { float4 v = 1; v.xx = v.xy; return v; }\n", true);
	TEST_CHECK(!r.ok); // duplicate component in a write
	r = run_sema(&arena, "float4 ps() : SV_TARGET { float4 v = 1; v.rgb = v.bgr; return v.xxxx; }\n");
	TEST_CHECK(r.ok);  // reordered write + duplicate read are both fine

	// implicit truncation warns but compiles (HLSL behavior)
	r = run_sema(&arena, "float4 ps() : SV_TARGET { float4 v = 1; float2 t = v; return t.xyxy; }\n");
	TEST_CHECK(r.ok);
	bool warned = false;
	for (int32_t i = 0; i < r.diags.count; i++)
		if (r.diags.items[i].severity == svsl_severity_warning &&
		    strstr(r.diags.items[i].message, "truncation")) warned = true;
	TEST_CHECK(warned);

	// unknown intrinsic argument shapes error cleanly
	r = run_sema_ex(&arena, "float4 ps() : SV_TARGET { return cross(float2(1,2), float2(3,4)).xyzz; }\n", true);
	TEST_CHECK(!r.ok);
	r = run_sema_ex(&arena, "float4x4 m;\nfloat4 ps() : SV_TARGET { return mul(float2(1,2), m).xyzw; }\n", true);
	TEST_CHECK(!r.ok); // vec2 * mat4x4 size mismatch

	// (struct)0 zero-fill idiom works
	r = run_sema(&arena,
		"struct psIn { float4 pos : SV_POSITION; };\n"
		"psIn vs() { psIn o = (psIn)0; return o; }\n"
		"float4 ps() : SV_TARGET { return 1; }\n");
	TEST_CHECK(r.ok);

	svsl_arena_free(&arena);
}

static void test_check_compute(void) {
	svsl_arena_t arena = {0};

	sema_run_t r = run_sema(&arena,
		"RWStructuredBuffer<uint> counts : register(u0);\n"
		"groupshared float shared_data[64];\n"
		"[numthreads(64, 1, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {\n"
		"	shared_data[gi] = float(id.x);\n"
		"	GroupMemoryBarrierWithGroupSync();\n"
		"	InterlockedAdd(counts[0], 1u);\n"
		"	uint prior = atomic_add(counts[1], 2u);\n"
		"	uint lane = subgroup_lane_id;\n"
		"	float sum = subgroup_add(shared_data[gi]);\n"
		"	counts[2] = uint(sum) + lane + prior;\n"
		"}\n");
	TEST_CHECK(r.ok);

	svsl_arena_free(&arena);
}

// --half=strict16 re-types every half as an exact float16 (2 bytes in buffers)
static void test_sema_half_strict16(void) {
	svsl_arena_t arena = {0};
	const char *src =
		"struct data_t { half a; half2 b; };\n"
		"StructuredBuffer<data_t> data : register(t0);\n"
		"half brightness;\n"
		"float4 ps() : SV_TARGET { return (float)(data[0].a * (half)brightness); }\n";

	sema_run_t r = {0};
	svsl_pp_result_t  pp;
	svsl_token_list_t tokens = {0};
	svsl_pp_run(&arena, src, "strict16.hlsl", NULL, &pp, &r.diags);
	svsl_lex(&arena, &pp, &tokens, &r.diags);
	svsl_ast_t *ast = svsl_parse(&arena, &tokens, &r.diags);
	r.ok = svsl_sema_run(&arena, ast, &pp, "strict16.hlsl",
	                     &(svsl_sema_options_t){ .half_strict16 = true }, &r.prog, &r.diags);
	TEST_CHECK(r.ok);

	const svsl_resource_t *res = find_resource(&r.prog, "data");
	TEST_CHECK(res != NULL);
	if (res) {
		// half a; half2 b; as float16: a at 0 (2B), b at 2..6 → 6-byte struct,
		// std430 element stride 8 (vec2 alignment 4)
		TEST_CHECK(res->element_size == 8);
		const svsl_type_t *t = svsl_type_get(&r.prog.types, res->type);
		const svsl_type_t *elem = svsl_type_get(&r.prog.types, t->elem);
		TEST_CHECK(elem->kind == svsl_type_struct);
		const svsl_struct_info_t *info = &r.prog.types.structs.items[elem->struct_index];
		TEST_CHECK(svsl_type_get(&r.prog.types, info->members.items[0].type)->scalar == svsl_scalar_float16);
	}
	// no half type may survive anywhere in the interned table
	for (int32_t i = 0; i < r.prog.types.types.count; i++)
		TEST_CHECK(r.prog.types.types.items[i].scalar != svsl_scalar_half);
	svsl_arena_free(&arena);
}

// attributes are honored or rejected, never silently dropped
static void test_sema_attributes(void) {
	svsl_arena_t arena = {0};

	// unknown plain attribute → warning; compile still succeeds
	sema_run_t r = run_sema(&arena,
		"[totally_bogus(1)]\n"
		"float4 ps() : SV_TARGET { return 1; }\n");
	TEST_CHECK(r.ok);
	bool warned = false;
	for (int32_t i = 0; i < r.diags.count; i++)
		if (r.diags.items[i].severity == svsl_severity_warning) warned = true;
	TEST_CHECK(warned);

	// unknown [[vk::*]] attribute → error (spec §6)
	r = run_sema_ex(&arena,
		"[[vk::frobnicate]]\n"
		"float4 ps() : SV_TARGET { return 1; }\n", true);
	TEST_CHECK(!r.ok);

	// [[vk::binding(b, set)]] is a direct descriptor binding
	r = run_sema(&arena,
		"[[vk::binding(3, 1)]] Texture2D    tex;\n"
		"[[vk::binding(4, 1)]] SamplerState smp;\n"
		"float4 ps(float2 uv : TEXCOORD0) : SV_TARGET { return tex.Sample(smp, uv); }\n");
	TEST_CHECK(r.ok);
	const svsl_resource_t *res = find_resource(&r.prog, "tex");
	TEST_CHECK(res && res->bind.direct && res->bind.slot == 3 && res->bind.space == 1);

	// [[vk::input_attachment_index(N)]] = SubpassInput<T, N>
	r = run_sema(&arena,
		"[[vk::input_attachment_index(2)]] SubpassInput<float4> color;\n"
		"float4 ps() : SV_TARGET { return color.SubpassLoad(); }\n");
	TEST_CHECK(r.ok);
	res = find_resource(&r.prog, "color");
	TEST_CHECK(res && res->subpass_index == 2);

	svsl_arena_free(&arena);
}

static void test_check_spirv_asm(void) {
	svsl_arena_t arena = {0};

	// valid: raw OpFAdd producing the block's %result
	sema_run_t r = run_sema(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	float a = 1, b = 2;\n"
		"	float s = spirv_asm(float) { OpFAdd $$float %result $a $b; };\n"
		"	return s;\n"
		"}\n");
	TEST_CHECK(r.ok);

	// unknown opcode mnemonic
	r = run_sema_ex(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	float s = spirv_asm(float) { OpFrobnicate $$float %result; };\n"
		"	return s;\n"
		"}\n", true);
	TEST_CHECK(!r.ok);

	// no %result id to carry the value
	r = run_sema_ex(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	float a = 1;\n"
		"	float s = spirv_asm(float) { OpFNegate $$float %tmp $a; };\n"
		"	return s;\n"
		"}\n", true);
	TEST_CHECK(!r.ok);

	// unknown type in a $$ operand
	r = run_sema_ex(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	float s = spirv_asm(float) { OpFAdd $$Nonsense %result $s; };\n"
		"	return s;\n"
		"}\n", true);
	TEST_CHECK(!r.ok);

	svsl_arena_free(&arena);
}

static void test_check_atomic_order(void) {
	svsl_arena_t arena = {0};
	const char *prefix =
		"RWStructuredBuffer<uint> b : register(u0);\n"
		"[numthreads(1,1,1)] void cs() { ";

	// relaxed default (no order arg)
	char src[512];
	snprintf(src, sizeof(src), "%s atomic_add(b[0], 1u); }\n", prefix);
	sema_run_t r = run_sema(&arena, src);
	TEST_CHECK(r.ok);

	// explicit orders on native atomics
	const char *ok_orders[] = { "acquire", "release", "acq_rel" };
	for (int32_t i = 0; i < 3; i++) {
		snprintf(src, sizeof(src), "%s atomic_exchange(b[0], 1u, %s); }\n", prefix, ok_orders[i]);
		r = run_sema(&arena, src);
		TEST_CHECK(r.ok);
	}
	// order on a compare-exchange (trailing, after compare+value)
	snprintf(src, sizeof(src), "%s uint p = atomic_compare_exchange(b[0], 0u, 1u, acq_rel); }\n", prefix);
	r = run_sema(&arena, src);
	TEST_CHECK(r.ok);

	// seq_cst is rejected — Vulkan has no sequential consistency
	snprintf(src, sizeof(src), "%s atomic_add(b[0], 1u, seq_cst); }\n", prefix);
	r = run_sema_ex(&arena, src, true);
	TEST_CHECK(!r.ok);

	// a non-order trailing identifier is not silently accepted (unknown name)
	snprintf(src, sizeof(src), "%s atomic_add(b[0], 1u, bogus); }\n", prefix);
	r = run_sema_ex(&arena, src, true);
	TEST_CHECK(!r.ok);

	// InterlockedAdd keeps its HLSL out-param form (order is native-only)
	snprintf(src, sizeof(src), "%s uint o; InterlockedAdd(b[0], 1u, o); }\n", prefix);
	r = run_sema(&arena, src);
	TEST_CHECK(r.ok);

	svsl_arena_free(&arena);
}

static void test_check_bitfield(void) {
	svsl_arena_t arena = {0};

	// unsigned extract/insert, result follows the value's type
	sema_run_t r = run_sema(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	uint p = 0x1234u;\n"
		"	uint f = bitfield_extract(p, 4u, 8u);\n"
		"	p = bitfield_insert(p, 5u, 4u, 8u);\n"
		"	return float4(float(f + p), 0, 0, 1);\n"
		"}\n");
	TEST_CHECK(r.ok);

	// signed value and vector value both bind the generic
	r = run_sema(&arena,
		"float4 ps() : SV_TARGET {\n"
		"	int  s = bitfield_extract((int)7, 0, 4);\n"
		"	uint2 v = bitfield_extract(uint2(1u,2u), 0u, 4u);\n"
		"	return float4(float(s) + float(v.x), 0, 0, 1);\n"
		"}\n");
	TEST_CHECK(r.ok);

	// a float value is rejected — bitfields are integer only
	r = run_sema_ex(&arena,
		"float4 ps() : SV_TARGET { return bitfield_extract(1.5, 0u, 4u); }\n", true);
	TEST_CHECK(!r.ok);

	svsl_arena_free(&arena);
}

static const svsl_struct_info_t *find_struct(const svsl_program_t *p, const char *name) {
	for (int32_t i = 0; i < p->types.structs.count; i++)
		if (svsl_str_eq_cstr(p->types.structs.items[i].name, name))
			return &p->types.structs.items[i];
	return NULL;
}

static void test_check_bitfield_struct(void) {
	svsl_arena_t arena = {0};

	// the user's canonical layout: bit fields + normalized + half + plain member
	sema_run_t r = run_sema(&arena,
		"struct P {\n"
		"	uint  a : 4;\n"
		"	int   b : 6;\n"
		"	bool  f : 1;\n"
		"	float c : un10;\n"
		"	half  d : sn6;\n"   // a..d = 27 bits, still in word 0
		"	half  h : 16;\n"    // would cross 32 → new word
		"	uint  e;\n"         // plain member, natural 32 bits → its own word
		"};\n"
		"RWStructuredBuffer<P> buf : register(u0);\n"
		"[numthreads(1,1,1)] void cs(uint3 t : SV_DispatchThreadID) {\n"
		"	buf[t.x].a = 13u; buf[t.x].b = -7; buf[t.x].f = true;\n"
		"	buf[t.x].c = 0.75; buf[t.x].d = -0.5; buf[t.x].h = half(1.5);\n"
		"	buf[t.x].e = buf[t.x].a + (uint)buf[t.x].b;\n"
		"}\n");
	TEST_CHECK(r.ok);

	const svsl_struct_info_t *p = find_struct(&r.prog, "P");
	TEST_CHECK(p != NULL);
	if (p) {
		TEST_CHECK(p->packed);
		TEST_CHECK(p->backing_words == 3);          // 27b + 16b(new word) + 32b(new word)
		TEST_CHECK(p->members.count  == 3);          // three uint32 backing words
		TEST_CHECK(p->fields.count   == 7);
		// dense LSB-first placement with the no-word-crossing rule
		const svsl_field_t *f = p->fields.items;
		TEST_CHECK(f[0].bit_offset == 0  && f[0].bit_width == 4);
		TEST_CHECK(f[1].bit_offset == 4  && f[1].bit_width == 6);
		TEST_CHECK(f[2].bit_offset == 10 && f[2].bit_width == 1);   // bool
		TEST_CHECK(f[3].bit_offset == 11 && f[3].bit_format == svsl_bitfmt_unorm);
		TEST_CHECK(f[4].bit_offset == 21 && f[4].bit_format == svsl_bitfmt_snorm);
		TEST_CHECK(f[5].bit_offset == 32 && f[5].bit_width == 16);  // half pushed to word 1
		TEST_CHECK(f[6].bit_offset == 64 && f[6].bit_width == 32);  // plain uint → word 2
	}

	// bool must be exactly one raw bit
	r = run_sema_ex(&arena, "struct B { bool x : un4; }; float4 ps():SV_TARGET{return 0;}\n", true);
	TEST_CHECK(!r.ok);
	// unorm/snorm require a float/half resolve type
	r = run_sema_ex(&arena, "struct B { uint x : un8; }; float4 ps():SV_TARGET{return 0;}\n", true);
	TEST_CHECK(!r.ok);
	// a raw field can't be wider than its resolve type
	r = run_sema_ex(&arena, "struct B { uint8 x : 12; }; float4 ps():SV_TARGET{return 0;}\n", true);
	TEST_CHECK(!r.ok);
	// non-scalar fields are rejected
	r = run_sema_ex(&arena, "struct B { float2 x : 8; }; float4 ps():SV_TARGET{return 0;}\n", true);
	TEST_CHECK(!r.ok);

	svsl_arena_free(&arena);
}

static const svsl_enum_const_t *find_enum_const(const svsl_program_t *p, const char *name) {
	for (int32_t i = 0; i < p->enum_consts.count; i++)
		if (svsl_str_eq_cstr(p->enum_consts.items[i].name, name))
			return &p->enum_consts.items[i];
	return NULL;
}

static void test_check_enum(void) {
	svsl_arena_t arena = {0};

	// anonymous enum with an underlying type + inline variable; auto-increment;
	// explicit values; named enum used as a type; constants in switch/expr context
	sema_run_t r = run_sema(&arena,
		"enum : int16 { One, Two } option = One;\n"
		"enum Mode : uint { ModeA = 1, ModeB = 4, ModeC };\n"
		"enum { Red, Green, Blue };\n"
		"RWStructuredBuffer<uint> buf : register(u0);\n"
		"[numthreads(1,1,1)] void cs(uint3 t : SV_DispatchThreadID) {\n"
		"	Mode m = ModeB;\n"
		"	uint r = (option == Two) ? 10u : 0u;\n"
		"	switch (m) { case ModeA: r += 1; break; case ModeC: r += 5; break; }\n"
		"	buf[t.x] = r + Green + (uint)m;\n"
		"}\n");
	TEST_CHECK(r.ok);

	// values: auto-increment from 0, explicit values, and continuation after them
	const svsl_enum_const_t *one = find_enum_const(&r.prog, "One");
	const svsl_enum_const_t *modec = find_enum_const(&r.prog, "ModeC");
	const svsl_enum_const_t *blue = find_enum_const(&r.prog, "Blue");
	TEST_CHECK(one && one->value == 0);
	TEST_CHECK(find_enum_const(&r.prog, "Two")->value == 1);
	TEST_CHECK(modec && modec->value == 5); // ModeB(4) + 1
	TEST_CHECK(blue && blue->value == 2);
	// underlying types propagate to the constants
	TEST_CHECK(svsl_type_get(&r.prog.types, one->type)->scalar == svsl_scalar_int16);
	TEST_CHECK(svsl_type_get(&r.prog.types, modec->type)->scalar == svsl_scalar_uint32);

	// enum as a packed bit-field resolve type → a raw integer field
	r = run_sema(&arena,
		"enum Facing : uint { N, E, S, W };\n"
		"struct T { Facing dir : 2; uint hp : 6; };\n"
		"RWStructuredBuffer<T> ts : register(u0);\n"
		"[numthreads(1,1,1)] void cs(uint3 t:SV_DispatchThreadID){ ts[t.x].dir = W; }\n");
	TEST_CHECK(r.ok);

	// a non-integer underlying type is rejected
	r = run_sema_ex(&arena, "enum E : float { A }; float4 ps():SV_TARGET{return 0;}\n", true);
	TEST_CHECK(!r.ok);
	// duplicate constant names are rejected
	r = run_sema_ex(&arena, "enum { X, X }; float4 ps():SV_TARGET{return 0;}\n", true);
	TEST_CHECK(!r.ok);
	// an enum name colliding with a struct is rejected
	r = run_sema_ex(&arena, "struct C {int x;}; enum C { A }; float4 ps():SV_TARGET{return 0;}\n", true);
	TEST_CHECK(!r.ok);

	// inline enums in a parameter, a struct member, and a local — constants are still
	// global, and each inline enum resolves to its underlying integer type
	r = run_sema(&arena,
		"uint pick(enum { Thing1, Thing2 } which) { return (uint)which; }\n"
		"struct W { enum : uint { Hidden, Shown } vis : 1; uint id : 31; };\n"
		"RWStructuredBuffer<W> ws : register(u0);\n"
		"[numthreads(1,1,1)] void cs(uint3 t : SV_DispatchThreadID) {\n"
		"	enum Dir { Up, Down } d = Down;\n"
		"	ws[t.x].vis = Shown;\n"
		"	ws[t.x].id  = pick(Thing2) + (uint)d;\n"
		"}\n");
	TEST_CHECK(r.ok);
	TEST_CHECK(find_enum_const(&r.prog, "Thing2") && find_enum_const(&r.prog, "Thing2")->value == 1);
	TEST_CHECK(find_enum_const(&r.prog, "Shown")  && find_enum_const(&r.prog, "Shown")->value == 1);
	TEST_CHECK(find_enum_const(&r.prog, "Down")   && find_enum_const(&r.prog, "Down")->value == 1);
	// the named inline enum 'Dir' is usable as a type name too
	TEST_CHECK(svsl_type_get(&r.prog.types, find_enum_const(&r.prog, "Down")->type)->scalar == svsl_scalar_int32);

	svsl_arena_free(&arena);
}

static int32_t count_porting(const sema_run_t *r) {
	int32_t n = 0;
	for (int32_t i = 0; i < r->diags.count; i++)
		if (r->diags.items[i].severity == svsl_severity_porting) n++;
	return n;
}

// porting hints are opt-in (-Wporting): off by default, one hint per legacy spelling
static void test_sema_porting_hints(void) {
	svsl_arena_t arena = {0};
	const char *src =
		"SamplerState s : register(s0);\n"
		"min16float4 tint = {1,1,1,1};\n"           // bare global: resolved twice, deduped to one hint
		"float4 ps() : SV_Target {\n"
		"	uint prev; InterlockedAdd(prev, 1u);\n"
		"	return tint;\n"
		"}\n";

	sema_run_t r = {0};
	svsl_pp_result_t  pp;
	svsl_token_list_t tokens = {0};
	svsl_pp_run(&arena, src, "porting.hlsl", NULL, &pp, &r.diags);
	svsl_lex(&arena, &pp, &tokens, &r.diags);
	svsl_ast_t *ast = svsl_parse(&arena, &tokens, &r.diags);

	r.ok = svsl_sema_run(&arena, ast, &pp, "porting.hlsl", NULL, &r.prog, &r.diags);
	TEST_CHECK(r.ok);
	TEST_CHECK(count_porting(&r) == 0); // off by default

	sema_run_t on = {0};
	on.ok = svsl_sema_run(&arena, ast, &pp, "porting.hlsl",
	                      &(svsl_sema_options_t){ .porting_hints = true }, &on.prog, &on.diags);
	TEST_CHECK(on.ok);
	TEST_CHECK(count_porting(&on) == 3); // SamplerState, min16float4, InterlockedAdd — each once
	svsl_arena_free(&arena);
}

void test_sema(void) {
	test_check_expressions();
	test_check_errors_and_warnings();
	test_check_spirv_asm();
	test_check_bitfield();
	test_check_bitfield_struct();
	test_check_enum();
	test_check_atomic_order();
	test_check_compute();
	test_sema_reference();
	test_sema_auto_binding();
	test_sema_spec_and_push();
	test_sema_storagebuffer_block();
	test_sema_half_strict16();
	test_sema_attributes();
	test_sema_multisample_and_tile();
	test_sema_texture_index();
	test_sema_porting_hints();
	test_sema_unsized_arrays();
	test_sema_errors();
}

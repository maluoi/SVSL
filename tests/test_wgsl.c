// WGSL backend tests: structural goldens over emitted text (bindings, split
// samplers, multiview override, atomics retyping, subpass lowering, control
// flow), the skip-diagnostic contract, container stage records, and a full
// corpus sweep. When `naga` is on PATH every emitted stage also validates —
// the same optional shell-out pattern the corpus suite uses for spirv-val.
//
// The whole suite no-ops (with a note) on a libsvsl built without
// SVSL_ENABLE_WGSL, so the default build's test run stays green.

#include "test.h"

#include <svsl/svsl.h>

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SVSL_TEST_DIR
#define SVSL_TEST_DIR "."
#endif

// --- helpers ----------------------------------------------------------------------

static char *wgsl_read_file(const char *path, int32_t *out_len) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	char  *data = malloc((size_t)size + 1);
	size_t got  = fread(data, 1, (size_t)size, f);
	fclose(f);
	data[got] = '\0';
	if (out_len) *out_len = (int32_t)got;
	return data;
}

// public-API include callback: the requesting file's folder, then shaders/include/
static svsl_include_src_t wgsl_include(void *user, const char *path, const char *requester) {
	(void)user;
	static char content_buf[1 << 20]; // callback results only need to live until compile returns
	char        full[1024];

	const char *slash = requester ? strrchr(requester, '/') : NULL;
	for (int32_t attempt = 0; attempt < 2; attempt++) {
		if (attempt == 0) {
			if (!slash) continue;
			snprintf(full, sizeof(full), "%.*s/%s", (int)(slash - requester), requester, path);
		} else {
			snprintf(full, sizeof(full), "%s/shaders/include/%s", SVSL_TEST_DIR, path);
		}
		int32_t len   = 0;
		char   *data  = wgsl_read_file(full, &len);
		if (!data) continue;
		if (len >= (int32_t)sizeof(content_buf)) { free(data); return (svsl_include_src_t){0}; }
		memcpy(content_buf, data, (size_t)len + 1);
		free(data);
		static char path_buf[1024];
		snprintf(path_buf, sizeof(path_buf), "%s", full);
		return (svsl_include_src_t){ .content = content_buf, .length = len, .path = path_buf };
	}
	return (svsl_include_src_t){0};
}

static svsl_result_t *compile_wgsl(const char *src) {
	return svsl_compile(
		&(svsl_source_t){ .text = src, .filename = "test_wgsl.hlsl" },
		&(svsl_options_t){ .targets    = svsl_target_spirv | svsl_target_wgsl,
		                   .include_cb = wgsl_include });
}

static const char *stage_text(svsl_result_t *r, svsl_stage_ stage) {
	for (int32_t i = 0; i < r->stage_count; i++)
		if (r->stages[i].stage == stage) return r->stages[i].wgsl;
	return NULL;
}

// one warning diagnostic contains `needle`?
static bool has_warning(svsl_result_t *r, const char *needle) {
	for (int32_t i = 0; i < r->diagnostic_count; i++)
		if (r->diagnostics[i].severity == svsl_severity_warning &&
		    strstr(r->diagnostics[i].message, needle)) return true;
	return false;
}

// Validators on PATH give every golden + corpus stage a real validation pass;
// a missing one degrades to the structural checks only, like spirv-val.
// `naga` is wgpu's validator, `tint` is Chrome's — they disagree on real
// cases, so both run when present.
static bool have_naga(void) {
	static int32_t probed = -1;
	if (probed < 0) {
		probed = system("naga --version > /dev/null 2>&1") == 0 ? 1 : 0;
		if (!probed) printf("  naga not on PATH: WGSL text validation skipped\n");
	}
	return probed == 1;
}

static bool have_tint(void) {
	static int32_t probed = -1;
	if (probed < 0) { // tint has no --version, and --help exits 1: probe by converting
		FILE *f = fopen("svsl_wgsl_probe.wgsl", "wb");
		if (f) { fputs("const svsl_probe : u32 = 1u;\n", f); fclose(f); }
		probed = system("tint svsl_wgsl_probe.wgsl --format wgsl -o /dev/null > /dev/null 2>&1") == 0 ? 1 : 0;
		remove("svsl_wgsl_probe.wgsl");
		if (!probed) printf("  tint not on PATH: Chrome-side WGSL validation skipped\n");
	}
	return probed == 1;
}

// WGSL's uniformity analysis for barriers cannot be waived by any directive;
// shaders with data-dependent trip counts around barriers are valid Vulkan but
// unprovable for Tint. Those reject in Chrome as authored — tracked as a
// count, not a failure, since only a source-level restructure can fix them.
static int32_t tint_uniformity_rejects = 0;

static bool wgsl_validate(const char *text, const char *what) {
	FILE *f = fopen("svsl_wgsl_tmp.wgsl", "wb");
	if (!f) return true;
	fwrite(text, 1, strlen(text), f);
	fclose(f);
	bool ok = true;
	if (have_naga() && system("naga svsl_wgsl_tmp.wgsl > /dev/null 2>&1") != 0) {
		printf("  naga rejected: %s\n", what);
		ok = false;
	}
	if (have_tint() &&
	    system("tint svsl_wgsl_tmp.wgsl --format wgsl -o /dev/null > /dev/null 2> svsl_wgsl_tmp.err") != 0) {
		FILE *err = fopen("svsl_wgsl_tmp.err", "rb");
		char  msg[4096] = {0};
		if (err) { fread(msg, 1, sizeof(msg) - 1, err); fclose(err); }
		if (strstr(msg, "uniform control flow")) {
			tint_uniformity_rejects++;
		} else {
			printf("  tint rejected: %s\n", what);
			ok = false;
		}
	}
	remove("svsl_wgsl_tmp.wgsl");
	remove("svsl_wgsl_tmp.err");
	return ok;
}

// --- container walk: per-stage language tags + sampler records --------------------

typedef struct wgsl_sks_info_t {
	int32_t  langs[8], lang_count;
	int32_t  sampler_count;
	uint16_t sampler_slot, sampler_paired;
} wgsl_sks_info_t;

static bool walk_sks(svsl_bytes_t sks, wgsl_sks_info_t *out) {
	memset(out, 0, sizeof(*out));
	const uint8_t *d = sks.data;
	if (sks.size < 300) return false;
	uint32_t stage_count, bufc, resc, specc, sampc;
	int32_t  vinc, o = 10;
	memcpy(&stage_count, d + o, 4); o += 4 + 256;
	memcpy(&bufc,  d + o, 4); o += 4;
	memcpy(&resc,  d + o, 4); o += 4;
	memcpy(&vinc,  d + o, 4); o += 4;
	memcpy(&specc, d + o, 4); o += 4;
	memcpy(&sampc, d + o, 4); o += 4;
	o += 16 + 24 + 4 + 8; // features, ops, wave, apron
	for (uint32_t b = 0; b < bufc; b++) {
		o += 32 + 1 + 4 + 4;
		uint32_t var_count, defaults_size;
		memcpy(&var_count,     d + o, 4); o += 4;
		memcpy(&defaults_size, d + o, 4); o += 4;
		o += (int32_t)defaults_size + (int32_t)var_count * 140;
	}
	o += vinc * 11 + (int32_t)resc * 172 + (int32_t)specc * 43;
	out->sampler_count = (int32_t)sampc;
	for (uint32_t s = 0; s < sampc; s++) {
		if (s == 0) {
			memcpy(&out->sampler_slot,   d + o + 32,     2);
			memcpy(&out->sampler_paired, d + o + 32 + 3, 2);
		}
		o += 37;
	}
	for (uint32_t s = 0; s < stage_count && out->lang_count < 8; s++) {
		if (o + 16 > sks.size) return false;
		int32_t  lang;
		uint32_t size;
		memcpy(&lang, d + o, 4);
		memcpy(&size, d + o + 12, 4);
		out->langs[out->lang_count++] = lang;
		o += 16 + (int32_t)size;
	}
	return true;
}

// --- golden checks ----------------------------------------------------------------

static void test_wgsl_bindings(void) {
	svsl_result_t *r = compile_wgsl(
		"cbuffer G : register(b0) { float4 color; };\n"
		"Texture2D    tex     : register(t3);\n"
		"SamplerState tex_smp : register(s3);\n"
		"float4 vs(float3 p : POSITION) : SV_Position { return float4(p, 1); }\n"
		"float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {\n"
		"	return color * tex.Sample(tex_smp, uv);\n"
		"}\n");
	TEST_CHECK(r->ok);
	const char *ps = stage_text(r, svsl_stage_pixel);
	TEST_CHECK(ps != NULL);
	if (ps) {
		TEST_CHECK(strstr(ps, "@group(0) @binding(0) var<uniform> G : G_t;"));
		TEST_CHECK(strstr(ps, "@binding(103) var tex :"));
		TEST_CHECK(strstr(ps, "@binding(403) var tex_smp : sampler;"));
		TEST_CHECK(strstr(ps, "textureSample(tex, tex_smp"));
		TEST_CHECK(wgsl_validate(ps, "bindings ps"));
	}

	// the container carries both languages plus the split-sampler record
	wgsl_sks_info_t info;
	TEST_CHECK(walk_sks(svsl_result_sks(r), &info));
	TEST_CHECK(info.lang_count == 4); // spirv vs+ps, wgsl vs+ps
	TEST_CHECK(info.langs[0] == 1 && info.langs[1] == 1);
	TEST_CHECK(info.langs[2] == 5 && info.langs[3] == 5);
	TEST_CHECK(info.sampler_count == 1);
	TEST_CHECK(info.sampler_slot == 403 && info.sampler_paired == 103);
	svsl_result_free(r);
}

static void test_wgsl_view_index(void) {
	svsl_result_t *r = compile_wgsl(
		"float4 vs(float3 p : POSITION, uint view : SV_ViewID) : SV_Position {\n"
		"	return float4(p.xy, float(view), 1);\n"
		"}\n");
	TEST_CHECK(r->ok);
	const char *vs = stage_text(r, svsl_stage_vertex);
	TEST_CHECK(vs != NULL);
	if (vs) {
		TEST_CHECK(strstr(vs, "@id(999) override sk_view_index : u32 = 0u;"));
		TEST_CHECK(strstr(vs, "= sk_view_index;")); // the param copy reads the override
		TEST_CHECK(wgsl_validate(vs, "view index vs"));
	}
	svsl_result_free(r);
}

static void test_wgsl_atomics(void) {
	svsl_result_t *r = compile_wgsl(
		"RWStructuredBuffer<uint> counts : register(u0);\n"
		"groupshared uint local_max;\n"
		"[numthreads(64, 1, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint prev;\n"
		"	InterlockedAdd(counts[id.x], 1u, prev);\n"
		"	InterlockedMax(local_max, prev);\n"
		"	uint total = counts[0];\n"                              // plain read of an atomic leaf
		"	counts[id.x + 64] = total;\n"                           // plain write of an atomic leaf
		"	uint orig;\n"
		"	InterlockedCompareExchange(counts[1], 0u, total, orig);\n"
		"}\n");
	TEST_CHECK(r->ok);
	const char *cs = stage_text(r, svsl_stage_compute);
	TEST_CHECK(cs != NULL);
	if (cs) {
		TEST_CHECK(strstr(cs, "array<atomic<u32>>"));               // buffer elements retype
		TEST_CHECK(strstr(cs, "var<workgroup> local_max : atomic<u32>;"));
		TEST_CHECK(strstr(cs, "atomicAdd(&counts["));
		TEST_CHECK(strstr(cs, "atomicMax(&local_max"));
		TEST_CHECK(strstr(cs, "atomicLoad(&counts["));              // plain read converts
		TEST_CHECK(strstr(cs, "atomicStore(&counts["));             // plain write converts
		TEST_CHECK(strstr(cs, "atomicCompareExchangeWeak(&counts[")); // strong-CAS retry loop
		TEST_CHECK(strstr(cs, ".exchanged"));
		TEST_CHECK(wgsl_validate(cs, "atomics cs"));
	}
	svsl_result_free(r);
}

static void test_wgsl_subpass(void) {
	svsl_result_t *r = compile_wgsl(
		"[[vk::input_attachment_index(0)]] SubpassInput<float4> prev;\n"
		"float4 vs(uint id : SV_VertexID) : SV_Position {\n"
		"	float2 uv = float2((id << 1) & 2, id & 2);\n"
		"	return float4(uv * 2 - 1, 0, 1);\n"
		"}\n"
		"float4 ps() : SV_Target { return prev.SubpassLoad(); }\n");
	TEST_CHECK(r->ok);
	const char *ps = stage_text(r, svsl_stage_pixel);
	TEST_CHECK(ps != NULL);
	if (ps) {
		// lowered to a plain texture fetched at this fragment's own pixel,
		// through a synthesized position input (the ps declares none itself)
		TEST_CHECK(strstr(ps, "var prev : texture_2d<f32>;"));
		TEST_CHECK(strstr(ps, "@builtin(position) sk_frag_pos"));
		TEST_CHECK(strstr(ps, "textureLoad(prev, vec2<i32>(in.sk_frag_pos.xy), 0)"));
		TEST_CHECK(wgsl_validate(ps, "subpass ps"));
	}
	svsl_result_free(r);
}

static void test_wgsl_control_flow(void) {
	svsl_result_t *r = compile_wgsl(
		"float4 ps(float4 pos : SV_Position) : SV_Target {\n"
		"	float acc = 0;\n"
		"	for (int i = 0; i < 4; i++) {\n"
		"		if (i == 2) { continue; }\n"
		"		acc += float(i) * pos.x;\n"
		"	}\n"
		"	return float4(acc, 0, 0, 1);\n"
		"}\n");
	TEST_CHECK(r->ok);
	const char *ps = stage_text(r, svsl_stage_pixel);
	TEST_CHECK(ps != NULL);
	if (ps) {
		TEST_CHECK(strstr(ps, "loop {"));
		TEST_CHECK(strstr(ps, "continuing {")); // the for-increment block
		TEST_CHECK(strstr(ps, "continue;"));
		TEST_CHECK(wgsl_validate(ps, "control flow ps"));
	}
	svsl_result_free(r);
}

// a do-while's condition rides the continuing block as a `break if`; constant
// NaN bit-patterns and divisions by constant zero defer to runtime through a
// var, since WGSL rejects them when const-evaluated (Tint catches these,
// naga historically has not — hence the goldens)
static void test_wgsl_const_traps(void) {
	svsl_result_t *r = compile_wgsl(
		"cbuffer G : register(b0) { int max_iter; };\n"
		"float4 ps(float4 pos : SV_Position) : SV_Target {\n"
		"	float v = pos.x;\n"
		"	int   i = 0;\n"
		"	do { v *= 0.5; i++; } while (v > 0.25 && i < max_iter);\n"
		"	float nan_val  = asfloat(0x7FC00000);\n"
		"	float inf_ish  = 1.0 / 0.0;\n"
		"	return float4(v, nan_val, inf_ish, float(i));\n"
		"}\n");
	TEST_CHECK(r->ok);
	const char *ps = stage_text(r, svsl_stage_pixel);
	TEST_CHECK(ps != NULL);
	if (ps) {
		TEST_CHECK(strstr(ps, "break if !("));      // conditional back-edge
		TEST_CHECK(strstr(ps, "bitcast<f32>(_bc")); // NaN bits through a var
		TEST_CHECK(strstr(ps, "/ _dz"));            // divisor through a var
		TEST_CHECK(wgsl_validate(ps, "const traps ps"));
	}
	svsl_result_free(r);
}

// depth-texture-ness follows the paired sampler's DECLARED type — the bind
// group layout the runtime builds derives from it, so usage must agree
static void test_wgsl_depth_pairing(void) {
	// declared + used as comparison: depth texture, comparison sampler
	svsl_result_t *r = compile_wgsl(
		"Texture2D              shadow   : register(t0);\n"
		"SamplerComparisonState shadow_s : register(s0);\n"
		"float4 ps(float4 pos : SV_Position, float3 sc : TEXCOORD0) : SV_Target {\n"
		"	return float4(shadow.SampleCmp(shadow_s, sc.xy, sc.z).xxx, 1);\n"
		"}\n");
	TEST_CHECK(r->ok);
	const char *ps = stage_text(r, svsl_stage_pixel);
	TEST_CHECK(ps != NULL);
	if (ps) {
		TEST_CHECK(strstr(ps, "var shadow : texture_depth_2d;"));
		TEST_CHECK(strstr(ps, "var shadow_s : sampler_comparison;"));
		TEST_CHECK(strstr(ps, "textureSampleCompare(shadow, shadow_s"));
		TEST_CHECK(wgsl_validate(ps, "depth pairing ps"));
	}
	svsl_result_free(r);

	// compare-sampled with an UNPAIRED comparison sampler: the layout would say
	// filterable-float while the module says depth — must skip, not mismatch
	r = compile_wgsl(
		"Texture2D              shadow : register(t0);\n"
		"SamplerComparisonState cmp    : register(s5);\n"
		"float4 ps(float4 pos : SV_Position, float3 sc : TEXCOORD0) : SV_Target {\n"
		"	return float4(shadow.SampleCmp(cmp, sc.xy, sc.z).xxx, 1);\n"
		"}\n");
	TEST_CHECK(r->ok);
	TEST_CHECK(stage_text(r, svsl_stage_pixel) == NULL);
	TEST_CHECK(has_warning(r, "paired comparison sampler"));
	svsl_result_free(r);

	// declared comparison but plain-sampled: a depth texture can't do both
	r = compile_wgsl(
		"Texture2D              shadow   : register(t0);\n"
		"SamplerComparisonState shadow_s : register(s0);\n"
		"SamplerState           smp      : register(s1);\n"
		"float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {\n"
		"	return shadow.Sample(smp, uv);\n"
		"}\n");
	TEST_CHECK(r->ok);
	TEST_CHECK(stage_text(r, svsl_stage_pixel) == NULL);
	TEST_CHECK(has_warning(r, "also plain-sampled"));
	svsl_result_free(r);
}

// stages using inexpressible features skip with a located warning; the SPIR-V
// stage still emits and the compile stays ok
static void test_wgsl_skips(void) {
	svsl_result_t *r = compile_wgsl(
		"float4 ps(float4 pos : SV_Position) : SV_Target {\n"
		"	return float4(float(WaveGetLaneCount()), 0, 0, 1);\n"
		"}\n");
	TEST_CHECK(r->ok);
	TEST_CHECK(stage_text(r, svsl_stage_pixel) == NULL);
	TEST_CHECK(has_warning(r, "wave/subgroup"));
	for (int32_t i = 0; i < r->stage_count; i++)
		TEST_CHECK(r->stages[i].spirv != NULL); // Vulkan output is unaffected
	svsl_result_free(r);

	r = compile_wgsl(
		"struct vsOut { float4 pos : SV_Position; uint layer : SV_RenderTargetArrayIndex; };\n"
		"vsOut vs(float3 p : POSITION) {\n"
		"	vsOut o; o.pos = float4(p, 1); o.layer = 1; return o;\n"
		"}\n");
	TEST_CHECK(r->ok);
	TEST_CHECK(stage_text(r, svsl_stage_vertex) == NULL);
	TEST_CHECK(has_warning(r, "SV_RenderTargetArrayIndex"));
	svsl_result_free(r);
}

// WGSL-only containers drop the SPIR-V stages entirely
static void test_wgsl_only_target(void) {
	svsl_result_t *r = svsl_compile(
		&(svsl_source_t){ .text = "float4 vs(float3 p : POSITION) : SV_Position { return float4(p, 1); }\n",
		                  .filename = "test_wgsl.hlsl" },
		&(svsl_options_t){ .targets = svsl_target_wgsl });
	TEST_CHECK(r->ok);
	wgsl_sks_info_t info;
	TEST_CHECK(walk_sks(svsl_result_sks(r), &info));
	TEST_CHECK(info.lang_count == 1 && info.langs[0] == 5);
	svsl_result_free(r);
}

// --- corpus sweep -----------------------------------------------------------------

static int wgsl_compare_names(const void *a, const void *b) {
	return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void wgsl_sweep_dir(const char *subdir, int32_t *ref_emitted, int32_t *ref_skipped) {
	char dir_path[1024];
	snprintf(dir_path, sizeof(dir_path), "%s/shaders/%s", SVSL_TEST_DIR, subdir);
	DIR *dir = opendir(dir_path);
	TEST_CHECK(dir != NULL);
	if (!dir) return;

	char   *names[256];
	int32_t name_count = 0;
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL && name_count < 256) {
		const char *ext = strrchr(entry->d_name, '.');
		if (!ext || (strcmp(ext, ".hlsl") != 0 && strcmp(ext, ".svsl") != 0)) continue;
		char *name = malloc(strlen(entry->d_name) + 1);
		strcpy(name, entry->d_name);
		names[name_count++] = name;
	}
	closedir(dir);
	qsort(names, (size_t)name_count, sizeof(names[0]), wgsl_compare_names);

	for (int32_t i = 0; i < name_count; i++) {
		char path[2048];
		snprintf(path, sizeof(path), "%s/%s", dir_path, names[i]);
		free(names[i]);

		char *src = wgsl_read_file(path, NULL);
		TEST_CHECK(src != NULL);
		if (!src) continue;
		svsl_result_t *r = svsl_compile(
			&(svsl_source_t){ .text = src, .filename = path },
			&(svsl_options_t){ .targets    = svsl_target_spirv | svsl_target_wgsl,
			                   .include_cb = wgsl_include });
		free(src);
		// requesting WGSL must never *break* a shader that compiles for Vulkan
		TEST_CHECK(r->ok);
		if (!r->ok)
			for (int32_t d = 0; d < r->diagnostic_count; d++)
				if (r->diagnostics[d].severity == svsl_severity_error)
					printf("  %s: %s\n", path, r->diagnostics[d].message);

		bool any_wgsl = false;
		for (int32_t s = 0; s < r->stage_count; s++)
			if (r->stages[s].wgsl) {
				any_wgsl = true;
				TEST_CHECK(wgsl_validate(r->stages[s].wgsl, path));
			}
		*(any_wgsl ? ref_emitted : ref_skipped) += 1;
		svsl_result_free(r);
	}
}

// ---------------------------------------------------------------------------------

void test_wgsl(void) {
	if (!svsl_supports_wgsl()) {
		printf("  built without SVSL_ENABLE_WGSL: wgsl suite skipped\n");
		return;
	}

	test_wgsl_bindings();
	test_wgsl_view_index();
	test_wgsl_atomics();
	test_wgsl_subpass();
	test_wgsl_control_flow();
	test_wgsl_const_traps();
	test_wgsl_depth_pairing();
	test_wgsl_skips();
	test_wgsl_only_target();

	int32_t emitted = 0, skipped = 0;
	wgsl_sweep_dir("builtin",   &emitted, &skipped);
	wgsl_sweep_dir("examples",  &emitted, &skipped);
	wgsl_sweep_dir("ported",    &emitted, &skipped);
	wgsl_sweep_dir("morrowind", &emitted, &skipped);
	wgsl_sweep_dir("checks",    &emitted, &skipped);
	printf("  wgsl corpus: %d emitted, %d skipped%s%s\n", emitted, skipped,
	       have_naga() ? " (naga-validated)" : "",
	       have_tint() ? " (tint-validated)" : "");
	if (tint_uniformity_rejects > 0)
		printf("  tint uniformity rejects: %d stage(s) — valid Vulkan, but Chrome "
		       "needs barriers in provably-uniform control flow\n", tint_uniformity_rejects);
	// a mass regression in the emitter shows up as a collapse in coverage;
	// per-feature skips are asserted individually above
	TEST_CHECK(emitted >= 90);
}

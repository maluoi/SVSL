// The comparison engine: compile with skshaderc + svslc, render both, diff.

#include "view.h"

#include "back/emit_spirv.h"
#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "ir/ir.h"
#include "out/sks_write.h"
#include "sema/sema.h"
#include "util/arena.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- compiling -------------------------------------------------------------------

static void *read_file(svsl_arena_t *arena, const char *path, int32_t *out_len) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	uint8_t *data = svsl_arena_alloc(arena, (size_t)size + 1);
	size_t   got  = fread(data, 1, (size_t)size, f);
	fclose(f);
	data[got] = '\0';
	if (out_len) *out_len = (int32_t)got;
	return data;
}

// reference compiler: shell out to the skshaderc built alongside sk_renderer
static bool compile_reference(svsl_arena_t *arena, const char *shader_path,
                              const char *include_dir, void **out_data, int32_t *out_size) {
	char cmd[2048];
	snprintf(cmd, sizeof(cmd), "%s -f -i %s -o svsl_view_ref.sks %s > /dev/null 2>&1",
	         SVSL_SKSHADERC_PATH, include_dir, shader_path);
	if (system(cmd) != 0) return false;
	*out_data = read_file(arena, "svsl_view_ref.sks", out_size);
	remove("svsl_view_ref.sks");
	if (*out_data == NULL) return false;
	// skshaderc exits 0 even when it finds no entry points, leaving a
	// stage-less container — that is not a usable reference
	uint32_t stage_count = 0;
	if (*out_size >= 14) memcpy(&stage_count, (const uint8_t *)*out_data + 10, 4);
	return stage_count > 0;
}

typedef struct include_ctx_t {
	svsl_arena_t *arena;
	const char   *dir;
} include_ctx_t;

static svsl_include_src_t view_include(void *user, const char *path, const char *requester) {
	include_ctx_t *ctx = user;
	char           full[1024];
	const char *slash = requester ? strrchr(requester, '/') : NULL;
	if (slash) {
		snprintf(full, sizeof(full), "%.*s/%s", (int32_t)(slash - requester), requester, path);
		int32_t len;
		void   *content = read_file(ctx->arena, full, &len);
		if (content) return (svsl_include_src_t){ .content = content, .length = len,
			.path = svsl_arena_strndup(ctx->arena, full, strlen(full)) };
	}
	snprintf(full, sizeof(full), "%s/%s", ctx->dir, path);
	int32_t len;
	void   *content = read_file(ctx->arena, full, &len);
	if (!content) return (svsl_include_src_t){0};
	return (svsl_include_src_t){ .content = content, .length = len,
		.path = svsl_arena_strndup(ctx->arena, full, strlen(full)) };
}

// Texture dimensionality per resource name, taken from our own sema
// reflection — the SKS container doesn't record it, and both compilers see
// identical declarations, so the names transfer to the reference build too.
typedef struct tex_kind_t {
	char    name[33];
	uint8_t dim;     // svsl_texdim_
	bool    arrayed;
} tex_kind_t;

#define TEX_KINDS_MAX 16

enum { subpass_none = 0, subpass_plain, subpass_msaa };

static bool compile_ours(svsl_arena_t *arena, const char *shader_path, const char *include_dir,
                         const void **out_data, int32_t *out_size, bool *out_compute_only,
                         bool *out_renderable, tex_kind_t *out_kinds, int32_t *out_kind_count,
                         bool *out_writes_layer, int32_t *out_uses_subpass) {
	char *src = read_file(arena, shader_path, NULL);
	if (!src) return false;

	svsl_diag_list_t  diags = {0};
	include_ctx_t     ictx  = { .arena = arena, .dir = include_dir };
	svsl_pp_options_t popt  = { .include_cb = view_include, .include_user = &ictx };
	svsl_pp_result_t  pp;
	svsl_pp_run(arena, src, shader_path, &popt, &pp, &diags);
	// Vulkan forbids writing the layer index inside a multiview render pass, so
	// shaders using SV_RenderTargetArrayIndex skip the multiview comparison
	*out_writes_layer = pp.text && strstr(pp.text, "SV_RenderTargetArrayIndex") != NULL;
	svsl_token_list_t tokens = {0};
	svsl_lex(arena, &pp, &tokens, &diags);
	svsl_ast_t *ast = svsl_parse(arena, &tokens, &diags);
	svsl_program_t prog;
	svsl_sema_run(arena, ast, &pp, shader_path, NULL, &prog, &diags);
	if (diags.error_count) return false;

	*out_kind_count  = 0;
	*out_uses_subpass = subpass_none;
	for (int32_t i = 0; i < prog.resources.count && *out_kind_count < TEX_KINDS_MAX; i++) {
		const svsl_resource_t *res = &prog.resources.items[i];
		if (res->kind == svsl_res_subpass) {
			const svsl_type_t *st = svsl_type_get(&prog.types, res->type);
			*out_uses_subpass = st->multisampled ? subpass_msaa : subpass_plain;
		}
		if (res->kind != svsl_res_texture && res->kind != svsl_res_image) continue;
		const svsl_type_t *t = svsl_type_get(&prog.types, res->type);
		if (t->kind == svsl_type_array) t = svsl_type_get(&prog.types, t->elem);
		tex_kind_t *k   = &out_kinds[(*out_kind_count)++];
		int32_t     len = res->name.len < 32 ? res->name.len : 32;
		memcpy(k->name, res->name.ptr, (size_t)len);
		k->name[len] = '\0';
		k->dim       = (uint8_t)t->dim;
		k->arrayed   = t->arrayed;
	}

	*out_compute_only = true;
	bool has_vs = false, has_ps = false;
	for (int32_t i = 0; i < prog.entries.count; i++) {
		if (prog.entries.items[i].stage != svsl_stage_compute) *out_compute_only = false;
		if (prog.entries.items[i].stage == svsl_stage_vertex)   has_vs = true;
		if (prog.entries.items[i].stage == svsl_stage_pixel)    has_ps = true;
	}
	*out_renderable = has_vs && has_ps;

	svsl_ir_module_t ir = {0};
	svsl_ir_build(arena, &prog, svsl_opt_default, &ir, &diags);
	if (diags.error_count) return false;
	svsl_spirv_blob_t blobs[8] = {0};
	for (int32_t i = 0; i < ir.func_count && i < 8; i++)
		if (!svsl_spirv_emit(arena, &prog, &ir.funcs[i], &blobs[i], &diags)) return false;
	svsl_sks_blob_t sks = {0};
	svsl_sks_write(arena, &prog, &ir, blobs, NULL,
	               &(svsl_sks_options_t){ .targets = svsl_target_spirv }, &sks);
	*out_data = sks.bytes;
	*out_size = sks.size;
	return true;
}

// --- container parsing ----------------------------------------------------------------

typedef struct sks_res_t {
	char     name[33];
	uint8_t  register_type; // skr_register_
	uint32_t element_size;
	uint8_t  shape;         // dim/arrayed/ms/comparison bits (0 = unreported)
	uint8_t  image_format;  // SpvImageFormat of storage images (0 = none)
} sks_res_t;

typedef struct sks_stage_t {
	int32_t        lang;  // skr_shader_lang_, 1 = SPIR-V
	int32_t        stage; // skr_stage_
	uint32_t       wave_size;
	const uint8_t *data;
	uint32_t       size;
} sks_stage_t;

typedef struct sks_buf_t {
	char     name[33];
	uint16_t slot; // raw register slot (b#)
	uint32_t size;
} sks_buf_t;

typedef struct sks_info_t {
	sks_buf_t   bufs  [8];
	sks_res_t   res   [16];
	sks_stage_t stages[4];
	char        vars  [64][33]; // constant buffer variable names, all buffers pooled
	uint64_t    features;
	int32_t     buf_count, res_count, stage_count, var_count;
} sks_info_t;

// walks the SKS v12 byte layout (matches sksc_build_file / sksc_file.c)
static bool sks_parse(const uint8_t *sks, int32_t size, sks_info_t *out) {
	memset(out, 0, sizeof(*out));
	if (size < 300) return false;
	int32_t  o = 8;
	uint16_t version;
	memcpy(&version, sks + o, 2); o += 2;
	if (version != 13) return false; // one version at a time
	uint32_t stage_count, bufc, resc, specc, samplerc;
	int32_t  vinc;
	memcpy(&stage_count, sks + o, 4); o += 4;
	o += 256; // name
	memcpy(&bufc, sks + o, 4); o += 4;
	memcpy(&resc, sks + o, 4); o += 4;
	memcpy(&vinc, sks + o, 4); o += 4;
	memcpy(&specc, sks + o, 4); o += 4;
	memcpy(&samplerc, sks + o, 4); o += 4; // v12
	memcpy(&out->features, sks + o, 8); o += 16; // mask + reserved word
	o += 24 + 4 + 8; // ops + wave + tile apron (v11)

	for (uint32_t b = 0; b < bufc; b++) {
		// only $Global vars: they're the ones skr_compute_set_param can reach
		bool is_global = memcmp(sks + o, "$Global", 8) == 0;
		if (out->buf_count < 8) {
			sks_buf_t *buf = &out->bufs[out->buf_count++];
			memcpy(buf->name, sks + o, 32);
			memcpy(&buf->slot, sks + o + 33, 2);     // after name + space byte
			memcpy(&buf->size, sks + o + 33 + 4, 4); // after the 4-byte bind
		}
		o += 32 + 1 + 4 + 4; // name, space, bind, size
		uint32_t var_count, defaults_size;
		memcpy(&var_count, sks + o, 4); o += 4;
		memcpy(&defaults_size, sks + o, 4); o += 4;
		o += (int32_t)defaults_size;
		for (uint32_t v = 0; v < var_count; v++) {
			if (is_global && out->var_count < 64)
				memcpy(out->vars[out->var_count++], sks + o, 32);
			o += 32 + 64 + 32 + 4 + 4 + 2 + 2; // name, extra, type name, offset, size, type, count
		}
	}
	o += vinc * 11; // format, count, semantic, slot, location

	for (uint32_t r = 0; r < resc && out->res_count < 16; r++) {
		sks_res_t *res = &out->res[out->res_count++];
		memcpy(res->name, sks + o, 32);
		o += 32 + 64 + 64; // name, value, tags
		res->register_type = sks[o + 3];
		o += 4;
		memcpy(&res->element_size, sks + o, 4); o += 4;
		res->shape        = sks[o];
		res->image_format = sks[o + 1];
		o += 4; // shape, format, reserved pad
	}
	o += (int32_t)specc * (32 + 4 + 4 + 2 + 1);
	o += (int32_t)samplerc * (32 + 2 + 1 + 2); // v12 sampler records (WGSL only)

	for (uint32_t s = 0; s < stage_count && out->stage_count < 4; s++) {
		if (o + 16 > size) return false;
		sks_stage_t *st = &out->stages[out->stage_count++];
		memcpy(&st->lang,      sks + o, 4); o += 4;
		memcpy(&st->stage,     sks + o, 4); o += 4;
		memcpy(&st->wave_size, sks + o, 4); o += 4;
		memcpy(&st->size,      sks + o, 4); o += 4;
		st->data = sks + o;
		o += (int32_t)st->size;
		if (o > size) return false;
	}
	return true;
}

// picks a scene texture for a reflected binding, by dimensionality when known
// (kinds come from our sema reflection) with name heuristics as the fallback
static skr_tex_t *tex_for_binding(scene_t *scene, const char *name, bool is_image,
                                  const tex_kind_t *kinds, int32_t kind_count) {
	const tex_kind_t *k = NULL;
	for (int32_t i = 0; i < kind_count; i++)
		if (strcmp(kinds[i].name, name) == 0) k = &kinds[i];
	if (is_image)                          return &scene->tex_3d_rw; // corpus images are volumes
	if (k && k->dim == svsl_texdim_3d)     return &scene->tex_3d;
	if (k && k->dim == svsl_texdim_cube)   return &scene->cubemap;
	if (k && k->arrayed)                   return &scene->tex_array;
	if (!k && (strstr(name, "cubemap") || strstr(name, "cube"))) return &scene->cubemap;
	if (!k && strstr(name, "shadow"))      return &scene->tex_array;
	return &scene->checker;
}

// binds every reflected texture and storage image by name
static void bind_textures(scene_t *scene, skr_material_t *material,
                          const uint8_t *sks, int32_t size,
                          const tex_kind_t *kinds, int32_t kind_count) {
	sks_info_t info;
	if (!sks_parse(sks, size, &info)) return;
	for (int32_t r = 0; r < info.res_count; r++) {
		const sks_res_t *res      = &info.res[r];
		bool             is_image = res->register_type == skr_register_readwrite_tex;
		if (res->register_type != skr_register_texture && !is_image) continue;
		skr_material_set_tex(material, res->name,
		                     tex_for_binding(scene, res->name, is_image, kinds, kind_count));
	}
}

// --- SPIR-V diff ------------------------------------------------------------------------

// Disassembles matching stages from both containers and prints a unified diff.
// Informational only: skshaderc's SPIR-V is optimized and ours is not, so the
// diff is always noisy — it exists for chasing codegen differences by hand.
static void spirv_diff(const void *ref, int32_t ref_size, const void *ours, int32_t ours_size) {
	if (system("spirv-dis --version > /dev/null 2>&1") != 0) {
		printf("  (spirv-dis not on PATH; skipping SPIR-V diff)\n");
		return;
	}
	sks_info_t a, b;
	if (!sks_parse(ref, ref_size, &a) || !sks_parse(ours, ours_size, &b)) return;

	for (int32_t i = 0; i < a.stage_count; i++) {
		const sks_stage_t *sa = &a.stages[i], *sb = NULL;
		for (int32_t k = 0; k < b.stage_count; k++)
			if (b.stages[k].stage == sa->stage) sb = &b.stages[k];
		if (!sb || sa->lang != skr_shader_lang_spirv || sb->lang != skr_shader_lang_spirv)
			continue;

		FILE *f = fopen("svsl_view_a.spv", "wb");
		if (!f) return;
		fwrite(sa->data, 1, sa->size, f);
		fclose(f);
		f = fopen("svsl_view_b.spv", "wb");
		if (!f) { remove("svsl_view_a.spv"); return; }
		fwrite(sb->data, 1, sb->size, f);
		fclose(f);

		const char *stage_name =
			sa->stage == skr_stage_vertex  ? "vertex"  :
			sa->stage == skr_stage_pixel   ? "pixel"   :
			sa->stage == skr_stage_compute ? "compute" : "?";
		printf("--- SPIR-V %s stage: skshaderc vs svslc ---\n", stage_name);
		fflush(stdout);
		system("spirv-dis svsl_view_a.spv -o svsl_view_a.spvasm 2> /dev/null && "
		       "spirv-dis svsl_view_b.spv -o svsl_view_b.spvasm 2> /dev/null && "
		       "diff -u svsl_view_a.spvasm svsl_view_b.spvasm");
		remove("svsl_view_a.spv");    remove("svsl_view_b.spv");
		remove("svsl_view_a.spvasm"); remove("svsl_view_b.spvasm");
	}
}

// --- compute checks ----------------------------------------------------------------------

// Compute shaders can't be pixel-diffed; instead both compilers' builds run on
// identical deterministic inputs and every read-write output is compared.
// Dispatch chains, parameters, and buffer sizes are per-shader table entries.
// Verification tiers, applied in combination per config:
//   - reference:    bitwise output diff against the skshaderc build (default)
//   - expect[]:     golden outputs, for shaders skshaderc can't compile —
//     bitwise words when eps is 0, |got - value| <= eps otherwise (float
//     buffer words, or texture readback bytes with .tex)
//   - radix_golden: CPU replay of one radix-sort pass (the gpu_sort chain,
//     where glslang's inclusive-scan WavePrefixSum makes the reference wrong)
//   - fallback:     none of the above applies (smoke)
// Every tier additionally requires that the run actually changed some output —
// two dispatches that silently no-op produce identical fills and would
// otherwise "match" perfectly.

enum { fill_hash = 0, fill_zero, fill_ramp }; // storage buffer initial contents

typedef struct compute_param_t {
	const char *name;
	uint16_t    type;     // sksc_shader_var_
	uint16_t    count;    // components, 0 = scalar
	float       value[4];
} compute_param_t;

typedef struct compute_pass_t {
	const char     *file;      // sibling shader to dispatch, NULL = the shader under test
	uint32_t        dispatch[3];
	compute_param_t params[2]; // per-pass values, applied over the shared params
} compute_pass_t;

typedef struct compute_cfg_t {
	const char     *file;         // shader basename prefix
	compute_pass_t  passes[4];    // dispatched in order on the same buffers/textures
	int32_t         tex_size;     // 2D storage texture dimensions, 0 = none
	bool            no_reference; // skip skshaderc: outputs known to diverge by design
	bool            radix_golden; // b_alt/b_altPayload must equal a CPU radix pass
	struct { const char *name; int32_t count; uint8_t fill; } buffers[8];
	compute_param_t params[8];    // applied to every pass that declares the variable
	struct { int32_t index; uint32_t bits; float value, eps; bool tex; } expect[16];
} compute_cfg_t;

// the gpu_sort chain: one full LSD radix pass (upsweep -> scan -> downsweep)
// over deterministic keys, verified against a CPU replay. radix_golden assumes
// exactly this buffer set: b_sort hash-filled, b_sortPayload ramp-filled.
#define GPU_SORT_CHAIN \
	.passes       = { { "40_gpu_sort_upsweep.hlsl",   {   2, 1, 1 } },   \
	                  { "39_gpu_sort_scan.hlsl",      { 256, 1, 1 } },   \
	                  /* globalHist counts -> exclusive prefix */        \
	                  { "38_gpu_sort_init.hlsl",      {   1, 1, 1 },     \
	                    { { "e_initPass", sksc_shader_var_uint, .value = { 2 } } } }, \
	                  { "37_gpu_sort_downsweep.hlsl", {   2, 1, 1 } } }, \
	.no_reference = true, /* WavePrefixSum: glslang emits InclusiveScan */ \
	.radix_golden = true, \
	.buffers      = { { "b_sort",        7680 },            \
	                  { "splats",           4 }, /* init reflects it; pass 2 leaves it alone */ \
	                  { "b_alt",         7680, fill_zero }, \
	                  { "b_sortPayload", 7680, fill_ramp }, \
	                  { "b_altPayload",  7680, fill_zero }, \
	                  { "b_globalHist",  1024, fill_zero }, \
	                  { "b_passHist",     512, fill_zero } }, \
	.params       = { { "e_numKeys",     sksc_shader_var_uint, .value = { 7680 } }, \
	                  { "e_radixShift",  sksc_shader_var_uint, .value = { 0 } },    \
	                  { "e_threadBlocks", sksc_shader_var_uint, .value = { 2 } } }

static const compute_cfg_t compute_cfgs[] = {
	{ .file     = "shader_builtin_sh_compute",
	  .passes   = { { .dispatch = { 1, 1, 1 } } },
	  .buffers  = { { "sh_output", 27 } },
	  .params   = { { "face_size", sksc_shader_var_uint, .value = { 16 } },
	                { "mip_level", sksc_shader_var_uint, .value = { 0 } } } },
	{ .file     = "compute_reaction",
	  .passes   = { { .dispatch = { 64 / 8, 64 / 8, 1 } } },
	  .tex_size = 64,
	  .buffers  = { { "input", 64 * 64 }, { "output", 64 * 64 } },
	  .params   = { { "feed",     sksc_shader_var_float, .value = { 0.042f  } },
	                { "kill",     sksc_shader_var_float, .value = { 0.059f  } },
	                { "diffuseA", sksc_shader_var_float, .value = { 0.2097f } },
	                { "diffuseB", sksc_shader_var_float, .value = { 0.105f  } },
	                { "timestep", sksc_shader_var_float, .value = { 0.8f    } },
	                { "size",     sksc_shader_var_uint,  .value = { 64      } } } },
	// GI pipeline (Morrowind): all numthreads(4,4,4); (4,4,4) dispatch covers
	// the harness's 16^3 volumes. Sampled volumes bind the scene gradient.
	{ .file = "gi_voxel_clear",  .passes = { { .dispatch = { 4, 4, 4 } } } },
	{ .file = "gi_sdf_seed",     .passes = { { .dispatch = { 4, 4, 4 } } } },
	{ .file = "gi_sdf_jfa",      .passes = { { .dispatch = { 4, 4, 4 } } },
	  .params = { { "step_size", sksc_shader_var_uint, .value = { 4 } } } },
	{ .file = "gi_sdf_finalize", .passes = { { .dispatch = { 4, 4, 4 } } } },
	{ .file = "gi_voxel_to_sh",  .passes = { { .dispatch = { 4, 4, 4 } } },
	  .buffers = { { "sh_hist", 16 * 16 * 16 } }, // 16-byte entries, one per probe
	  .params  = { { "frame_seed", sksc_shader_var_uint, .value = { 1 } } } },
	// ported corpus, reference-comparable HLSL
	{ .file     = "29_bloom_downsample", // reads/writes storage textures only
	  .passes   = { { .dispatch = { 8, 8, 1 } } },
	  .tex_size = 64 },
	{ .file     = "30_bloom_upsample", // BloomParams is a named cbuffer: the
	  .passes   = { { .dispatch = { 8, 8, 1 } } }, // harness fill supplies it
	  .tex_size = 64 },
	{ .file     = "31_compute_test", // the reaction-diffusion shader, loose-uniform form
	  .passes   = { { .dispatch = { 64 / 8, 64 / 8, 1 } } },
	  .tex_size = 64,
	  .buffers  = { { "input", 64 * 64 }, { "output", 64 * 64 } },
	  .params   = { { "feed",     sksc_shader_var_float, .value = { 0.042f  } },
	                { "kill",     sksc_shader_var_float, .value = { 0.059f  } },
	                { "diffuseA", sksc_shader_var_float, .value = { 0.2097f } },
	                { "diffuseB", sksc_shader_var_float, .value = { 0.105f  } },
	                { "timestep", sksc_shader_var_float, .value = { 0.8f    } },
	                { "size",     sksc_shader_var_uint,  .value = { 64      } } } },
	{ .file    = "38_gpu_sort_init", // two passes: clear histogram, then key init
	  .passes  = { { NULL, { 4, 1, 1 }, { { "e_initPass", sksc_shader_var_uint, .value = { 0 } } } },
	               { NULL, { 1, 1, 1 }, { { "e_initPass", sksc_shader_var_uint, .value = { 1 } } } } },
	  .buffers = { { "splats", 256 },
	               { "b_sort", 256, fill_zero }, { "b_sortPayload", 256, fill_zero },
	               { "b_globalHist", 1024, fill_zero } },
	  .params  = { { "e_numKeys", sksc_shader_var_uint,  .value = { 256 } },
	               { "e_camPos",  sksc_shader_var_float, 3, { 0.5f, 1.0f, -2.0f } } } },
	{ .file = "37_gpu_sort_downsweep", GPU_SORT_CHAIN },
	{ .file = "39_gpu_sort_scan",      GPU_SORT_CHAIN },
	{ .file = "40_gpu_sort_upsweep",   GPU_SORT_CHAIN },
	{ .file    = "42_orbital_particles_compute",
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "input", 256 }, { "output", 256, fill_zero } },
	  .params  = { { "time",           sksc_shader_var_float, .value = { 0.5f   } },
	               { "delta_time",     sksc_shader_var_float, .value = { 0.016f } },
	               { "damping",        sksc_shader_var_float, .value = { 0.98f  } },
	               { "max_speed",      sksc_shader_var_float, .value = { 2.0f   } },
	               { "strength",       sksc_shader_var_float, .value = { 1.0f   } },
	               { "particle_count", sksc_shader_var_uint,  .value = { 256    } } } },
	// ported corpus, SVSL dialect (no skshaderc reference): epsilon goldens,
	// CPU-derived from the deterministic fills (named cbuffers like Params get
	// the make_global_buffers fill, so word 0 of a b0 cbuffer is 0.25 exactly)
	{ .file     = "07_compute_basic", // color = (uv.x, uv.y, sin(time), 1)
	  .passes   = { { .dispatch = { 8, 8, 1 } } },
	  .tex_size = 64,
	  // resolution = (0.25, 0.6065), time = 0.96305 -> sin = 0.8209 -> 209
	  .expect   = { { 0, .value = 0,   .eps = 1, .tex = true },  // (0,0).r: 0/0.25
	                { 1, .value = 0,   .eps = 1, .tex = true },  // (0,0).g
	                { 2, .value = 209, .eps = 3, .tex = true },  // (0,0).b: sin(time)
	                { 3, .value = 255, .eps = 1, .tex = true },  // (0,0).a
	                { 4, .value = 255, .eps = 1, .tex = true },  // (1,0).r: 1/0.25 sat.
	                { 5, .value = 0,   .eps = 1, .tex = true } } },
	{ .file     = "13_storage_image", // box blur; the fill-derived radius spreads
	  .passes   = { { .dispatch = { 8, 8, 1 } } }, // taps texture-wide -> constant
	  .tex_size = 64, // observed average of the clamped checker, pinned +-4 for
	  .expect   = {   // linear-filter precision differences across drivers
	                { 0,    .value =  84, .eps = 4, .tex = true },
	                { 1,    .value = 116, .eps = 4, .tex = true },
	                { 2,    .value = 144, .eps = 4, .tex = true },
	                { 3,    .value = 255, .eps = 1, .tex = true },
	                { 8320, .value =  84, .eps = 4, .tex = true },   // center pixel
	                { 8322, .value = 144, .eps = 4, .tex = true } } },
	{ .file    = "21_storage_buffer", // results = values*scale*mult; pos += vel*mult
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "InputData", 1 }, { "OutputData", 1, fill_zero },
	               { "particles", 16 }, { "particles_rw", 16, fill_zero } },
	  // scale = fill(65) = 0.89822, multiplier = 0.25 (Params word 0)
	  .expect  = { {  69, .value = 0.10674571f, .eps = 1e-4f }, // results[0].y
	               {  70, .value = 0.21349141f, .eps = 1e-4f },
	               {  71, .value = 0.09567812f, .eps = 1e-4f },
	               { 131, .value = 0.21276844f, .eps = 1e-4f }, // results[15].w
	               { 260, .value = 0.22536050f, .eps = 1e-4f }, // rw[0].position
	               { 261, .value = 0.56956208f, .eps = 1e-4f },
	               { 382, .value = 1.21732664f, .eps = 1e-4f } } }, // rw[15].position.z
	{ .file    = "78_spec_atomics",
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "accum", 64 } },
	  // slots 1..15 collect exact atomic float adds: fill(s) + 0.25*(4s + 96) =
	  // fill(s) + s + 24, order-independent to ~1e-5. Slot 0 mixes in the
	  // atomic_min and slots 62/63 are exchange races — excluded by design.
	  .expect  = { {  1, .value = 25.475365f, .eps = 1e-3f },
	               {  2, .value = 26.950729f, .eps = 1e-3f },
	               {  3, .value = 27.426077f, .eps = 1e-3f },
	               {  4, .value = 28.901442f, .eps = 1e-3f },
	               {  5, .value = 29.376791f, .eps = 1e-3f },
	               {  6, .value = 30.852156f, .eps = 1e-3f },
	               {  7, .value = 31.327505f, .eps = 1e-3f },
	               {  8, .value = 32.802868f, .eps = 1e-3f },
	               {  9, .value = 33.278217f, .eps = 1e-3f },
	               { 10, .value = 34.753582f, .eps = 1e-3f },
	               { 11, .value = 35.228931f, .eps = 1e-3f },
	               { 12, .value = 36.704296f, .eps = 1e-3f },
	               { 13, .value = 37.179646f, .eps = 1e-3f },
	               { 14, .value = 38.655010f, .eps = 1e-3f },
	               { 15, .value = 39.130360f, .eps = 1e-3f } } },
	// dedicated runtime checks (tests/shaders/checks/)
	{ .file = "check_wave",    .passes = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "result", 64 * 14 } } },
	{ .file = "check_atomics", .passes = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "result", 16 } } },
	{ .file    = "check_pack_layout", // C-packed default vs pack16 elements
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "parts", 8, fill_zero }, { "parts16", 8, fill_zero } },
	  // No reference: glslang packs neither like C nor like std430 (hybrid), so
	  // goldens come from the layouts by hand. parts element i = 4 words at 4i:
	  // [id=i, pos.x=.25i, pos.y=.5i, scale=2(1+i)]; parts16 element i = 8 words
	  // at 32+8i: [id, pad, pos.x, pos.y, scale=(1+i), pad, pad, pad].
	  .expect  = { {  0, .value = 0, .eps = 0.1f }, // parts[0].id = 0
	               {  3, 0x40000000 },              // parts[0].scale = 2*(1+0) = 2.0f
	               {  4, 1 },                       // parts[1].id
	               {  5, 0x3E800000 },              // parts[1].pos.x = 0.25f
	               {  6, 0x3F000000 },              // parts[1].pos.y = 0.5f
	               {  7, 0x40800000 },              // parts[1].scale = 4.0f
	               { 28, 7 },                       // parts[7].id
	               { 29, 0x3FE00000 },              // parts[7].pos.x = 1.75f
	               { 30, 0x40600000 },              // parts[7].pos.y = 3.5f
	               { 31, 0x41800000 },              // parts[7].scale = 16.0f
	               { 40, 1 },                       // parts16[1].id (word 32+8)
	               { 41, .value = 0, .eps = 0.1f }, // parts16[1] std140 pad word
	               { 42, 0x3E800000 },              // parts16[1].pos.x at offset 8
	               { 44, 0x40000000 },              // parts16[1].scale = 2.0f (not doubled)
	               { 95, .value = 0, .eps = 0.1f } } }, // tail pad of parts16[7]
	{ .file    = "check_bool_store", // bool struct members: uint 0/1 in memory
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "results", 8, fill_zero } },
	  // rec_t is 4 words per element: pair.x, pair.y, flag, v.
	  // pair = (i>1, i>5), flag = !(i>3), v = i*0.5 — one thread per element, exact.
	  .expect  = { {  0, .value = 0, .eps = 0.1f }, // [0] pair.x: 0>1 = 0
	               {  2, 1 },                       // [0] flag: !(0>3) = 1
	               {  8, 1 },                       // [2] pair.x: 2>1
	               { 11, 0x3F800000 },              // [2] v = 1.0f
	               { 16, 1 },                       // [4] pair.x: 4>1
	               { 17, .value = 0, .eps = 0.1f }, // [4] pair.y: 4>5 = 0
	               { 18, .value = 0, .eps = 0.1f }, // [4] flag: !(4>3) = 0
	               { 29, 1 },                       // [7] pair.y: 7>5
	               { 31, 0x40600000 } } },          // [7] v = 3.5f
	{ .file    = "check_pack_half_vec", // vector f32tof16/f16tof32 lowering
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "result", 9, fill_zero } },
	  // half bits: 1.0=0x3C00 -2.0=0xC000 0.5=0x3800 4096=0x6C00 -0.25=0xB400
	  //            65504=0x7BFF -65504=0xFBFF 1.5=0x3E00 (all exact in f16)
	  .expect  = { { 0, 0xC0003C00 },   // h2: 1.0, -2.0
	               { 1, 0x6C003800 },   // h3.xy: 0.5, 4096
	               { 2, 0xB400 },       // h3.z: -0.25
	               { 3, 0xFBFF7BFF },   // h4.xy: 65504, -65504
	               { 4, 0x3E000000 },   // h4.zw: 0, 1.5
	               { 5, 0xBF800000 },   // f2.x + f2.y = -1.0f
	               { 6, 0x45000000 },   // f3.x * f3.y = 2048.0f
	               { 7, 0xBE800000 },   // f3.z = -0.25f
	               { 8, 0x3FC00000 } } }, // f4 sum = 1.5f
	// SVSL-native checks (enums, bit fields, atomic orders): golden words.
	// check_atomic_order only pins buf[0] — the other slots are racy by design.
	{ .file    = "check_atomic_order",
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "buf", 4, fill_zero } },
	  .expect  = { { 0, 2016 } } }, // sum 0..63
	{ .file    = "check_bitfield",
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "buf", 1, fill_zero } },
	  .expect  = { { 0, 0xC8A1F802 } } }, // packed + 3 + 500 + (uint)-24 + 200 + 200
	{ .file    = "check_bitfield_struct",
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "buf", 1, fill_zero } },
	  // word 0: a=13 | b=-7<<4 | f<<10 | un10(0.75)=767<<11 | sn6(-0.5)=-15<<21
	  // (normalized fields truncate toward zero on write, per the spec)
	  .expect  = { { 0, 0x0637FF9D },
	               { 1, 0x3E00 },        // half(1.5) raw in its own word
	               { 2, 0xDEADBF00 } } }, // uint(sum): float(0xDEADBEEF) absorbs +8.77
	{ .file    = "check_enum",
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "buf", 1, fill_zero } },
	  .expect  = { { 0, 8 } } }, // default(3) + Green(1) + ModeB(4)
	{ .file    = "check_enum_bitfield",
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "tiles", 1, fill_zero } },
	  .expect  = { { 0, 0x1FF } } }, // dir West(3) | hp 63<<2 | solid<<8
	{ .file    = "check_enum_inline",
	  .passes  = { { .dispatch = { 1, 1, 1 } } },
	  .buffers = { { "widgets", 1, fill_zero } },
	  .expect  = { { 0, 0xB } } }, // Shown(1) | (doThing(Thing2)=2 + Right(3))<<1
};

#define COMPUTE_MAX_FLOATS 36864 // gpu_sort chain: 4 * 7680 keys + histograms
#define COMPUTE_MAX_TEXES  4
#define COMPUTE_MAX_PASSES 4
#define TEX_INIT_BYTES     (16 * 16 * 16 * 4) // volumes are 16^3, 2D up to 64^2

typedef struct compute_out_t {
	float   floats[COMPUTE_MAX_FLOATS];
	uint8_t tex   [COMPUTE_MAX_TEXES * TEX_INIT_BYTES];
	int32_t float_count, tex_bytes;
	int32_t buf_offset[8]; // word offset of each config buffer, -1 = not bound
	bool    wrote;         // at least one output word changed from its initial fill
} compute_out_t;

static const compute_cfg_t *compute_cfg_find(const char *shader_path) {
	const char *base = strrchr(shader_path, '/') ? strrchr(shader_path, '/') + 1 : shader_path;
	const char *dot  = strchr(base, '.');
	size_t      len  = dot ? (size_t)(dot - base) : strlen(base);
	for (size_t i = 0; i < sizeof(compute_cfgs) / sizeof(compute_cfgs[0]); i++)
		if (strlen(compute_cfgs[i].file) == len && strncmp(base, compute_cfgs[i].file, len) == 0)
			return &compute_cfgs[i];
	return NULL;
}

// deterministic input fill, identical for both compilers
static float fill_value(int32_t i) {
	return (float)(((uint32_t)i * 2654435761u) & 0xFFFF) / 65535.0f;
}

static uint32_t buf_init_word(uint8_t fill, int32_t i) {
	if (fill == fill_zero) return 0;
	if (fill == fill_ramp) return (uint32_t)i;
	uint32_t bits;
	float    f = fill_value(i);
	memcpy(&bits, &f, 4);
	return bits;
}

// storage texture texel fill, one byte per channel
static uint8_t tex_init_byte(int32_t i) {
	return (uint8_t)(((uint32_t)i * 2654435761u) >> 16);
}

// Constant buffers the runtime doesn't own are the app's job to bind
// (SKMorrowind binds them via skr_renderer_set_global_constants). The harness
// fills them with a deterministic pattern kept off zero, so divisions stay
// finite and both compilers see identical parameters. Returns the buffer
// count; out_indices maps each buffer back to its info->bufs record.
// For render passes that's every slot above material + system; for compute
// it's every named cbuffer — skr_compute only auto-manages "$Global".
static int32_t make_global_buffers(const sks_info_t *info, skr_buffer_t *out_bufs,
                                   int32_t *out_indices, int32_t max, bool compute) {
	int32_t count = 0;
	for (int32_t b = 0; b < info->buf_count && count < max; b++) {
		if (compute ? strcmp(info->bufs[b].name, "$Global") == 0
		            : info->bufs[b].slot <= 1) continue;
		static float fill[4096];
		uint32_t floats = info->bufs[b].size / 4;
		if (floats > 4096) floats = 4096;
		for (uint32_t i = 0; i < floats; i++)
			fill[i] = 0.25f + 0.75f * fill_value((int32_t)i);
		if (skr_buffer_create(fill, info->bufs[b].size, 1, skr_buffer_type_constant,
		                      skr_use_static, &out_bufs[count]) != skr_err_success)
			continue;
		out_indices[count++] = b;
	}
	return count;
}

static bool sks_has_var(const sks_info_t *info, const char *name) {
	for (int32_t i = 0; i < info->var_count; i++)
		if (strcmp(info->vars[i], name) == 0) return true;
	return false;
}

// sets each named parameter the stage actually declares (chain passes share
// one param list, but used-only reflection differs per stage)
static void set_params(skr_compute_t *compute, const sks_info_t *info,
                       const compute_param_t *params, int32_t max) {
	for (int32_t p = 0; p < max && params[p].name; p++) {
		if (!sks_has_var(info, params[p].name)) continue;
		uint32_t n = params[p].count ? params[p].count : 1;
		if (params[p].type == sksc_shader_var_uint) {
			uint32_t v[4];
			for (uint32_t i = 0; i < n; i++) v[i] = (uint32_t)params[p].value[i];
			skr_compute_set_param(compute, params[p].name, sksc_shader_var_uint, n, v);
		} else {
			skr_compute_set_param(compute, params[p].name, params[p].type, n, params[p].value);
		}
	}
}

// Dispatches the config's pass chain on shared buffers/textures, then reads
// every storage buffer and texture back. blobs[] holds one compiled container
// per pass, all from the same compiler.
static bool run_compute(scene_t *scene, const void *const *blobs, const int32_t *sizes,
                        int32_t pass_count, const compute_cfg_t *cfg,
                        const tex_kind_t *kinds, int32_t kind_count, compute_out_t *out) {
	memset(out, 0, sizeof(*out));
	for (int32_t b = 0; b < 8; b++) out->buf_offset[b] = -1;

	sks_info_t infos[COMPUTE_MAX_PASSES];
	for (int32_t p = 0; p < pass_count; p++)
		if (!sks_parse(blobs[p], sizes[p], &infos[p])) return false;

	// storage buffers from the config, shared by every pass; element sizes come
	// from whichever pass reflects the name (used-only reflection: a buffer no
	// pass touches is skipped identically for both compilers)
	skr_buffer_t buffers  [8] = {0};
	int32_t      buf_words[8] = {0};
	bool         ok = true;

	for (int32_t b = 0; b < 8 && ok && cfg->buffers[b].name; b++) {
		uint32_t element_size = 0;
		for (int32_t p = 0; p < pass_count && !element_size; p++)
			for (int32_t r = 0; r < infos[p].res_count; r++) {
				const sks_res_t *res = &infos[p].res[r];
				if ((res->register_type == skr_register_read_buffer ||
				     res->register_type == skr_register_readwrite) &&
				    strcmp(res->name, cfg->buffers[b].name) == 0) {
					element_size = res->element_size;
					break;
				}
			}
		if (!element_size) continue;
		int32_t words = cfg->buffers[b].count * (int32_t)element_size / 4;
		if (words <= 0 || words > COMPUTE_MAX_FLOATS) { ok = false; break; }
		static uint32_t init[COMPUTE_MAX_FLOATS];
		for (int32_t i = 0; i < words; i++)
			init[i] = buf_init_word(cfg->buffers[b].fill, i);
		ok = skr_buffer_create(init, (uint32_t)cfg->buffers[b].count, element_size,
		                       skr_buffer_type_storage,
		                       skr_use_dynamic | skr_use_compute_readwrite,
		                       &buffers[b]) == skr_err_success;
		buf_words[b] = ok ? words : 0;
	}

	// storage textures, union across passes keyed by name: fresh deterministic
	// contents per run so unwritten texels diff clean and reads see real data.
	// Volumes are fixed 16^3; 2D sizes come from the config.
	skr_tex_t storage_texs [COMPUTE_MAX_TEXES] = {0};
	char      storage_names[COMPUTE_MAX_TEXES][33];
	int32_t   tex_count = 0;
	static uint8_t tex_init[TEX_INIT_BYTES];
	for (int32_t i = 0; i < TEX_INIT_BYTES; i++)
		tex_init[i] = tex_init_byte(i);

	for (int32_t p = 0; p < pass_count && ok; p++)
		for (int32_t r = 0; r < infos[p].res_count && ok; r++) {
			const sks_res_t *res = &infos[p].res[r];
			if (res->register_type != skr_register_readwrite_tex) continue;
			bool exists = false;
			for (int32_t x = 0; x < tex_count; x++)
				if (strcmp(storage_names[x], res->name) == 0) exists = true;
			if (exists) continue;
			const tex_kind_t *k = NULL;
			for (int32_t i = 0; i < kind_count; i++)
				if (strcmp(kinds[i].name, res->name) == 0) k = &kinds[i];
			bool volume = k && k->dim == svsl_texdim_3d;
			if ((!volume && cfg->tex_size <= 0) || tex_count == COMPUTE_MAX_TEXES) {
				ok = false;
				break;
			}
			skr_vec3i_t size = volume ? (skr_vec3i_t){ 16, 16, 16 }
			                          : (skr_vec3i_t){ cfg->tex_size, cfg->tex_size, 1 };
			skr_tex_data_t init = { .data = tex_init, .mip_count = 1, .layer_count = 1 };
			ok = skr_tex_create(skr_tex_fmt_rgba32_linear,
			                    (volume ? skr_tex_flags_3d : skr_tex_flags_none) |
			                    skr_tex_flags_compute | skr_tex_flags_readable,
			                    (skr_tex_sampler_t){ .sample  = skr_tex_sample_linear,
			                                         .address = skr_tex_address_clamp },
			                    size, 1, 1, &init, &storage_texs[tex_count]) == skr_err_success;
			if (ok) {
				snprintf(storage_names[tex_count], 33, "%s", res->name);
				tex_count++;
			}
		}

	// the pass chain, fully synchronized between dispatches
	for (int32_t p = 0; p < pass_count && ok; p++) {
		const sks_info_t *info = &infos[p];
		skr_shader_t shader;
		if (skr_shader_create(blobs[p], (uint32_t)sizes[p], &shader) != skr_err_success) {
			ok = false;
			break;
		}
		skr_compute_t compute;
		if (skr_compute_create(&shader, (skr_compute_info_t){0}, &compute) != skr_err_success) {
			skr_shader_destroy(&shader);
			ok = false;
			break;
		}

		// engine-owned constant buffers (GIBuffer and friends), bound by name
		skr_buffer_t cbufs[4];
		int32_t      cbuf_indices[4];
		int32_t      cbuf_count = make_global_buffers(info, cbufs, cbuf_indices, 4, true);
		for (int32_t c = 0; c < cbuf_count; c++)
			skr_compute_set_buffer(&compute, info->bufs[cbuf_indices[c]].name, &cbufs[c]);

		for (int32_t r = 0; r < info->res_count && ok; r++) {
			const sks_res_t *res = &info->res[r];
			if (res->register_type == skr_register_texture) {
				skr_compute_set_tex(&compute, res->name,
				                    tex_for_binding(scene, res->name, false, kinds, kind_count));
			}
			else if (res->register_type == skr_register_read_buffer ||
			         res->register_type == skr_register_readwrite) {
				ok = false; // every reflected buffer needs a config entry
				for (int32_t b = 0; b < 8 && cfg->buffers[b].name; b++)
					if (buf_words[b] && strcmp(res->name, cfg->buffers[b].name) == 0) {
						skr_compute_set_buffer(&compute, res->name, &buffers[b]);
						ok = true;
					}
			}
			else if (res->register_type == skr_register_readwrite_tex) {
				ok = false;
				for (int32_t x = 0; x < tex_count; x++)
					if (strcmp(storage_names[x], res->name) == 0) {
						skr_compute_set_tex(&compute, res->name, &storage_texs[x]);
						ok = true;
					}
			}
		}

		if (ok) {
			set_params(&compute, info, cfg->params, 8);
			set_params(&compute, info, cfg->passes[p].params, 2);
			skr_compute_execute(&compute, cfg->passes[p].dispatch[0],
			                    cfg->passes[p].dispatch[1], cfg->passes[p].dispatch[2]);
			skr_future_t done = skr_future_get();
			skr_future_wait(&done);
		}
		for (int32_t c = 0; c < cbuf_count; c++)
			skr_buffer_destroy(&cbufs[c]);
		skr_compute_destroy(&compute);
		skr_shader_destroy(&shader);
	}

	// read back everything, in config order so word indices are stable
	for (int32_t b = 0; b < 8 && ok; b++) {
		if (!buf_words[b]) continue;
		if (out->float_count + buf_words[b] > COMPUTE_MAX_FLOATS) { ok = false; break; }
		out->buf_offset[b] = out->float_count;
		skr_buffer_get(&buffers[b], out->floats + out->float_count,
		               (uint32_t)buf_words[b] * 4);
		for (int32_t i = 0; i < buf_words[b]; i++) {
			uint32_t bits;
			memcpy(&bits, &out->floats[out->float_count + i], 4);
			if (bits != buf_init_word(cfg->buffers[b].fill, i)) out->wrote = true;
		}
		out->float_count += buf_words[b];
	}
	for (int32_t x = 0; x < tex_count && ok; x++) {
		skr_tex_readback_t rb;
		ok = skr_tex_readback(&storage_texs[x], 0, 0, &rb) == skr_err_success;
		if (ok) {
			skr_future_wait(&rb.future);
			int32_t room = (int32_t)sizeof(out->tex) - out->tex_bytes;
			int32_t take = rb.size <= (uint32_t)room ? (int32_t)rb.size : room;
			memcpy(out->tex + out->tex_bytes, rb.data, (size_t)take);
			for (int32_t i = 0; i < take; i++)
				if (out->tex[out->tex_bytes + i] != tex_init_byte(i)) out->wrote = true;
			out->tex_bytes += take;
			skr_tex_readback_destroy(&rb);
		}
	}

	for (int32_t b = 0; b < 8; b++)
		if (buf_words[b]) skr_buffer_destroy(&buffers[b]);
	for (int32_t x = 0; x < tex_count; x++)
		skr_tex_destroy(&storage_texs[x]);
	return ok;
}

// CPU replay of the gpu_sort chain's single radix pass (shift 0): b_alt must
// hold b_sort stably sorted by low byte, b_altPayload the ramp payloads
// carried along. Wave-size independent, so it needs no reference compiler.
// Returns the mismatching word count, -1 on a config problem.
static int32_t radix_golden_diff(const compute_cfg_t *cfg, const compute_out_t *ours,
                                 bool verbose) {
	int32_t alt_off = -1, pay_off = -1, count = 0;
	for (int32_t b = 0; b < 8 && cfg->buffers[b].name; b++) {
		if (strcmp(cfg->buffers[b].name, "b_alt") == 0) {
			alt_off = ours->buf_offset[b];
			count   = cfg->buffers[b].count;
		}
		if (strcmp(cfg->buffers[b].name, "b_altPayload") == 0)
			pay_off = ours->buf_offset[b];
	}
	if (alt_off < 0 || pay_off < 0 || count <= 0 || count > 8192) return -1;

	static uint32_t expect_key[8192], expect_pay[8192];
	uint32_t prefix[257] = {0};
	for (int32_t i = 0; i < count; i++)
		prefix[(buf_init_word(fill_hash, i) & 255u) + 1]++;
	for (int32_t d = 0; d < 256; d++)
		prefix[d + 1] += prefix[d];
	for (int32_t i = 0; i < count; i++) {
		uint32_t key = buf_init_word(fill_hash, i);
		uint32_t pos = prefix[key & 255u]++;
		expect_key[pos] = key;
		expect_pay[pos] = (uint32_t)i;
	}

	int32_t bad = 0;
	for (int32_t i = 0; i < count; i++) {
		uint32_t key, pay;
		memcpy(&key, &ours->floats[alt_off + i], 4);
		memcpy(&pay, &ours->floats[pay_off + i], 4);
		if (key != expect_key[i] || pay != expect_pay[i]) {
			if (verbose && bad < 12)
				printf("  radix[%5d]  key %08x want %08x   payload %8u want %8u\n",
				       i, key, expect_key[i], pay, expect_pay[i]);
			bad++;
		}
	}
	return bad;
}

// --- rendering ------------------------------------------------------------------------

// Renders one frame and reads it back. With `multiview`, both views of a
// 2-layer target render in one pass (view_mask 0x3) and out_pixels receives
// both layers back to back (2 * VIEW_SIZE * VIEW_SIZE * 4 bytes).
static bool render_with(scene_t *scene, const void *sks_data, int32_t sks_size,
                        const tex_kind_t *kinds, int32_t kind_count, bool multiview,
                        skr_tex_t *color, skr_tex_t *depth, uint8_t *out_pixels) {
	skr_shader_t shader;
	if (skr_shader_create(sks_data, (uint32_t)sks_size, &shader) != skr_err_success)
		return false;

	skr_material_info_t info = {
		.shader      = &shader,
		.cull        = skr_cull_back,
		.write_mask  = skr_write_default,
		.depth_test  = skr_compare_less_or_eq,
		.blend_state = { .src_color_factor = skr_blend_one,
		                 .dst_color_factor = skr_blend_zero,
		                 .color_op         = skr_blend_op_add,
		                 .src_alpha_factor = skr_blend_one,
		                 .dst_alpha_factor = skr_blend_zero,
		                 .alpha_op         = skr_blend_op_add },
	};
	skr_material_t material;
	if (skr_material_create(info, &material) != skr_err_success) {
		skr_shader_destroy(&shader);
		return false;
	}
	bind_textures(scene, &material, sks_data, sks_size, kinds, kind_count);

	uint32_t view_mask = multiview ? 0x3 : 0x1;
	skr_renderer_frame_begin();
	skr_renderer_begin_pass(color, depth, NULL, skr_clear_all,
	                        (skr_vec4_t){ 0.12f, 0.12f, 0.14f, 1.0f }, 1.0f, 0,
	                        view_mask, view_mask);
	skr_renderer_set_viewport((skr_rect_t){ 0, 0, VIEW_SIZE, VIEW_SIZE });
	skr_renderer_set_scissor((skr_recti_t){ 0, 0, VIEW_SIZE, VIEW_SIZE });
	skr_render_list_clear(&scene->list);
	skr_render_list_add(&scene->list, &scene->sphere, &material,
	                    &scene->instance, sizeof(scene->instance), 1);
	skr_renderer_draw(&scene->list, multiview ? &scene->system_mv : &scene->system,
	                  sizeof(scene->system));
	skr_renderer_end_pass();
	skr_renderer_frame_end(NULL, 0);

	bool ok = true;
	for (int32_t layer = 0; layer < (multiview ? 2 : 1) && ok; layer++) {
		skr_tex_readback_t readback;
		ok = skr_tex_readback(color, 0, (uint32_t)layer, &readback) == skr_err_success;
		if (ok) {
			skr_future_wait(&readback.future);
			if (readback.size >= VIEW_SIZE * VIEW_SIZE * 4)
				memcpy(out_pixels + layer * VIEW_SIZE * VIEW_SIZE * 4, readback.data,
				       VIEW_SIZE * VIEW_SIZE * 4);
			else
				ok = false;
			skr_tex_readback_destroy(&readback);
		}
	}

	skr_material_destroy(&material);
	skr_shader_destroy(&shader);
	return ok;
}

// --- PNG output --------------------------------------------------------------------

// Minimal dependency-free PNG: zlib stream with stored (uncompressed) deflate
// blocks. ~77KB more per image than a real compressor; universally viewable.

static void png_be32(uint8_t *out, uint32_t v) {
	out[0] = (uint8_t)(v >> 24); out[1] = (uint8_t)(v >> 16);
	out[2] = (uint8_t)(v >> 8);  out[3] = (uint8_t)v;
}

static uint32_t png_crc(const uint8_t *data, size_t len, uint32_t crc) {
	static uint32_t table[256];
	if (!table[1]) {
		for (uint32_t n = 0; n < 256; n++) {
			uint32_t c = n;
			for (int32_t k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
			table[n] = c;
		}
	}
	for (size_t i = 0; i < len; i++) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
	return crc;
}

static void png_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t len) {
	uint8_t head[8];
	png_be32(head, len);
	memcpy(head + 4, type, 4);
	fwrite(head, 1, 8, f);
	if (len) fwrite(data, 1, len, f);
	uint8_t tail[4];
	png_be32(tail, png_crc(data, len, png_crc(head + 4, 4, 0xFFFFFFFFu)) ^ 0xFFFFFFFFu);
	fwrite(tail, 1, 4, f);
}

// --- postfx / SubpassInput rendering -----------------------------------------------

// Geometry for the postfx pass: the unlit sphere, built ONCE with our own
// compiler and shared by both sides — so the diff isolates the postfx shader.
static skr_material_t *baseline_material(scene_t *scene) {
	static skr_material_t material;
	static skr_shader_t   shader;
	static int32_t        state = 0; // 0 untried, 1 ok, -1 failed
	if (state != 0) return state > 0 ? &material : NULL;
	state = -1;

	svsl_arena_t arena = {0};
	char         path[1024], include_dir[1024];
	snprintf(path, sizeof(path), "%s/builtin/shader_builtin_unlit.hlsl", SVSL_SHADER_DIR);
	snprintf(include_dir, sizeof(include_dir), "%s/include", SVSL_SHADER_DIR);
	const void *sks = NULL;
	int32_t     size = 0;
	bool        compute_only, writes_layer, renderable;
	int32_t     uses_subpass;
	tex_kind_t  kinds[TEX_KINDS_MAX];
	int32_t     kind_count = 0;
	if (compile_ours(&arena, path, include_dir, &sks, &size, &compute_only,
	                 &renderable, kinds, &kind_count, &writes_layer, &uses_subpass) &&
	    skr_shader_create(sks, (uint32_t)size, &shader) == skr_err_success &&
	    skr_material_create((skr_material_info_t){
	    	.shader     = &shader,
	    	.cull       = skr_cull_back,
	    	.write_mask = skr_write_default,
	    	.depth_test = skr_compare_less_or_eq,
	    }, &material) == skr_err_success) {
		skr_material_set_tex(&material, "diffuse", &scene->checker);
		state = 1;
	}
	svsl_arena_free(&arena);
	return state > 0 ? &material : NULL;
}

// Renders the baseline sphere into the 4x MSAA target, then the candidate
// shader as the manual resolve subpass (SubpassInputMS reads the samples
// in tile memory, writes the resolved image).
static bool render_resolve_with(scene_t *scene, const void *sks_data, int32_t sks_size,
                                skr_tex_t *out_color, uint8_t *out_pixels) {
	skr_material_t *base = baseline_material(scene);
	if (!base) return false;

	skr_shader_t shader;
	if (skr_shader_create(sks_data, (uint32_t)sks_size, &shader) != skr_err_success)
		return false;
	skr_material_t resolve;
	if (skr_material_create((skr_material_info_t){
	    	.shader     = &shader,
	    	.cull       = skr_cull_none,
	    	.depth_test = skr_compare_always,
	    }, &resolve) != skr_err_success) {
		skr_shader_destroy(&shader);
		return false;
	}

	skr_renderer_frame_begin();
	skr_render_list_clear(&scene->list);
	skr_render_list_add(&scene->list, &scene->sphere, base,
	                    &scene->instance, sizeof(scene->instance), 1);
	skr_pass_t pass = {
		.color            = &scene->color_ms,
		.depth            = &scene->depth_ms,
		.resolve          = out_color,
		.clear            = skr_clear_all,
		.clear_color      = { 0.12f, 0.12f, 0.14f, 1.0f },
		.clear_depth      = 1.0f,
		.viewport         = { 0, 0, VIEW_SIZE, VIEW_SIZE },
		.scissor          = { 0, 0, VIEW_SIZE, VIEW_SIZE },
		.view_count       = 1,
		.views_correlated = true,
	};
	skr_pass_add_draw(&pass, &scene->list, &scene->system, sizeof(scene->system));
	skr_pass_add_resolve(&pass, &resolve);
	skr_pass_submit(&pass);
	skr_renderer_frame_end(NULL, 0);

	skr_tex_readback_t readback;
	bool ok = skr_tex_readback(out_color, 0, 0, &readback) == skr_err_success;
	if (ok) {
		skr_future_wait(&readback.future);
		if (readback.size >= VIEW_SIZE * VIEW_SIZE * 4)
			memcpy(out_pixels, readback.data, VIEW_SIZE * VIEW_SIZE * 4);
		else
			ok = false;
		skr_tex_readback_destroy(&readback);
	}
	skr_material_destroy(&resolve);
	skr_shader_destroy(&shader);
	return ok;
}

// Renders the baseline sphere into the input attachment, then the candidate
// shader as a postfx subpass reading it via SubpassLoad.
static bool render_postfx_with(scene_t *scene, const void *sks_data, int32_t sks_size,
                               skr_tex_t *out_color, skr_tex_t *depth, uint8_t *out_pixels) {
	skr_material_t *base = baseline_material(scene);
	if (!base) return false;

	skr_shader_t shader;
	if (skr_shader_create(sks_data, (uint32_t)sks_size, &shader) != skr_err_success)
		return false;
	skr_material_t postfx;
	if (skr_material_create((skr_material_info_t){
	    	.shader     = &shader,
	    	.cull       = skr_cull_none,
	    	.depth_test = skr_compare_always,
	    }, &postfx) != skr_err_success) {
		skr_shader_destroy(&shader);
		return false;
	}

	skr_renderer_frame_begin();
	skr_render_list_clear(&scene->list);
	skr_render_list_add(&scene->list, &scene->sphere, base,
	                    &scene->instance, sizeof(scene->instance), 1);
	skr_pass_t pass = {
		.color            = &scene->color_pf,
		.depth            = depth,
		.postfx_output    = out_color,
		.clear            = skr_clear_all,
		.clear_color      = { 0.12f, 0.12f, 0.14f, 1.0f },
		.clear_depth      = 1.0f,
		.viewport         = { 0, 0, VIEW_SIZE, VIEW_SIZE },
		.scissor          = { 0, 0, VIEW_SIZE, VIEW_SIZE },
		.view_count       = 1,
		.views_correlated = true,
	};
	skr_pass_add_draw(&pass, &scene->list, &scene->system, sizeof(scene->system));
	skr_pass_add_postfx(&pass, &postfx);
	skr_pass_submit(&pass);
	skr_renderer_frame_end(NULL, 0);

	skr_tex_readback_t readback;
	bool ok = skr_tex_readback(out_color, 0, 0, &readback) == skr_err_success;
	if (ok) {
		skr_future_wait(&readback.future);
		if (readback.size >= VIEW_SIZE * VIEW_SIZE * 4)
			memcpy(out_pixels, readback.data, VIEW_SIZE * VIEW_SIZE * 4);
		else
			ok = false;
		skr_tex_readback_destroy(&readback);
	}
	skr_material_destroy(&postfx);
	skr_shader_destroy(&shader);
	return ok;
}

static void write_png(const char *path, const uint8_t *rgba) {
	// scanlines: filter byte 0 + RGB per row
	static uint8_t raw[VIEW_SIZE * (VIEW_SIZE * 3 + 1)];
	int32_t n = 0;
	for (int32_t y = 0; y < VIEW_SIZE; y++) {
		raw[n++] = 0;
		for (int32_t x = 0; x < VIEW_SIZE; x++) {
			const uint8_t *px = rgba + (y * VIEW_SIZE + x) * 4;
			raw[n++] = px[0]; raw[n++] = px[1]; raw[n++] = px[2];
		}
	}
	static uint8_t idat[sizeof(raw) + 2 + 4 + 5 * (sizeof(raw) / 65535 + 1)];
	int32_t m = 0;
	idat[m++] = 0x78; idat[m++] = 0x01; // zlib header, no compression preset
	for (int32_t off = 0; off < n; off += 65535) {
		uint16_t len = (uint16_t)(n - off > 65535 ? 65535 : n - off);
		idat[m++] = off + len >= n ? 1 : 0; // BFINAL + stored block
		idat[m++] = (uint8_t)len;        idat[m++] = (uint8_t)(len >> 8);
		idat[m++] = (uint8_t)~len & 0xFF; idat[m++] = (uint8_t)(~len >> 8);
		memcpy(idat + m, raw + off, len);
		m += len;
	}
	uint32_t a = 1, b = 0; // adler32 of the raw scanline data
	for (int32_t i = 0; i < n; i++) { a = (a + raw[i]) % 65521; b = (b + a) % 65521; }
	png_be32(idat + m, (b << 16) | a);
	m += 4;

	FILE *f = fopen(path, "wb");
	if (!f) return;
	fwrite("\x89PNG\r\n\x1a\n", 1, 8, f);
	uint8_t ihdr[13] = {0};
	png_be32(ihdr, VIEW_SIZE);
	png_be32(ihdr + 4, VIEW_SIZE);
	ihdr[8] = 8; ihdr[9] = 2; // 8-bit RGB
	png_chunk(f, "IHDR", ihdr, 13);
	png_chunk(f, "IDAT", idat, (uint32_t)m);
	png_chunk(f, "IEND", NULL, 0);
	fclose(f);
}

compare_result_t compare_shader(scene_t *scene, const char *shader_path,
                                const char *include_dir, const char *opt_out_dir,
                                bool show_spirv_diff) {
	compare_result_t r     = {0};
	svsl_arena_t     arena = {0};

	const void *ours = NULL;
	int32_t     ours_size = 0;
	bool        compute_only = false;
	bool        renderable   = false;
	bool        writes_layer = false;
	int32_t     uses_subpass = subpass_none;
	tex_kind_t  kinds[TEX_KINDS_MAX];
	int32_t     kind_count = 0;
	r.compiled_ours = compile_ours(&arena, shader_path, include_dir, &ours, &ours_size,
	                               &compute_only, &renderable, kinds, &kind_count,
	                               &writes_layer, &uses_subpass);
	if (!r.compiled_ours) {
		snprintf(r.message, sizeof(r.message), "svslc failed");
		svsl_arena_free(&arena);
		return r;
	}
	if (compute_only && !compute_cfg_find(shader_path)) {
		r.skipped = true;
		r.ok      = true;
		snprintf(r.message, sizeof(r.message), "compute-only (no test config)");
		svsl_arena_free(&arena);
		return r;
	}
	if (!compute_only && !renderable) { // compile-corpus shader, not a runnable vs+ps pair
		r.skipped = true;
		r.ok      = true;
		snprintf(r.message, sizeof(r.message), "no vs+ps pair (compile-only)");
		svsl_arena_free(&arena);
		return r;
	}

	if (compute_only) {
		const compute_cfg_t *cfg = compute_cfg_find(shader_path);
		int32_t pass_count = 0;
		while (pass_count < COMPUTE_MAX_PASSES && cfg->passes[pass_count].dispatch[0])
			pass_count++;

		// compile the dispatch chain with both compilers; sibling files come
		// from the shader's own directory
		const void *ours_blobs[COMPUTE_MAX_PASSES];
		int32_t     ours_sizes[COMPUTE_MAX_PASSES];
		void       *ref_blobs [COMPUTE_MAX_PASSES];
		int32_t     ref_sizes [COMPUTE_MAX_PASSES];
		bool        ref_ok = !cfg->no_reference;
		const char *slash  = strrchr(shader_path, '/');
		for (int32_t p = 0; p < pass_count; p++) {
			char path[1024];
			if (cfg->passes[p].file && slash)
				snprintf(path, sizeof(path), "%.*s/%s",
				         (int32_t)(slash - shader_path), shader_path, cfg->passes[p].file);
			else
				snprintf(path, sizeof(path), "%s", shader_path);
			if (strcmp(path, shader_path) == 0) {
				ours_blobs[p] = ours;
				ours_sizes[p] = ours_size;
			} else {
				bool       co, rn, wl;
				int32_t    us, kc;
				tex_kind_t kd[TEX_KINDS_MAX];
				if (!compile_ours(&arena, path, include_dir, &ours_blobs[p], &ours_sizes[p],
				                  &co, &rn, kd, &kc, &wl, &us)) {
					snprintf(r.message, sizeof(r.message), "svslc failed (%s)", path);
					svsl_arena_free(&arena);
					return r;
				}
			}
			if (ref_ok)
				ref_ok = compile_reference(&arena, path, include_dir, &ref_blobs[p], &ref_sizes[p]);
		}
		r.compiled_ref = ref_ok;
		if (ref_ok && show_spirv_diff)
			for (int32_t p = 0; p < pass_count; p++)
				spirv_diff(ref_blobs[p], ref_sizes[p], ours_blobs[p], ours_sizes[p]);

		static compute_out_t out_ref, out_ours;
		if (!run_compute(scene, ours_blobs, ours_sizes, pass_count, cfg,
		                 kinds, kind_count, &out_ours)) {
			snprintf(r.message, sizeof(r.message), "compute run failed (svsl)");
			svsl_arena_free(&arena);
			return r;
		}
		if (ref_ok && !run_compute(scene, (const void *const *)ref_blobs, ref_sizes,
		                           pass_count, cfg, kinds, kind_count, &out_ref)) {
			snprintf(r.message, sizeof(r.message), "compute run failed (ref)");
			svsl_arena_free(&arena);
			return r;
		}
		r.rendered = true;
		r.ok       = true;

		char    checks[192];
		int32_t len      = 0;
		bool    verified = false; // some tier beyond the smoke test applied
		checks[0] = '\0';

		if (ref_ok) {
			// buffer elements may hold integer data whose float interpretation
			// is a denormal (~1e-42), so a magnitude threshold would never see
			// integer bugs — buffers must match BITWISE; texture bytes get a
			// tolerance
			double  total    = 0;
			int32_t n        = 0;
			int32_t shown    = 0;
			int32_t bit_diff = 0;
			for (int32_t i = 0; i < out_ref.float_count && i < out_ours.float_count; i++, n++) {
				uint32_t bits_ref, bits_ours;
				memcpy(&bits_ref,  &out_ref.floats[i],  4);
				memcpy(&bits_ours, &out_ours.floats[i], 4);
				if (bits_ref == bits_ours) continue;
				bit_diff++;
				double d = fabs((double)out_ref.floats[i] - (double)out_ours.floats[i]);
				total += isnan(d) ? 1.0 : d;
				if (show_spirv_diff && shown < 16) {
					printf("  float[%5d]  ref %12.6f (%08x)   svsl %12.6f (%08x)\n",
					       i, out_ref.floats[i], bits_ref, out_ours.floats[i], bits_ours);
					shown++;
				}
			}
			for (int32_t i = 0; i < out_ref.tex_bytes && i < out_ours.tex_bytes; i++, n++)
				total += fabs((double)out_ref.tex[i] - (double)out_ours.tex[i]) / 255.0;
			r.avg_error = n > 0 ? total / n : 1.0;
			r.ok = out_ref.float_count == out_ours.float_count &&
			       out_ref.tex_bytes   == out_ours.tex_bytes   &&
			       bit_diff == 0 && r.avg_error < 0.01;
			verified = true;
			len += snprintf(checks + len, sizeof(checks) - (size_t)len,
			                "avg error %.5f, %d/%d bit-exact (%s)", r.avg_error,
			                out_ref.float_count - bit_diff, out_ref.float_count,
			                out_ref.tex_bytes > 0 ? "floats + tex" : "floats");
		} else if (show_spirv_diff) {
			// no reference: dump leading output words + texels to help author expect[]
			int32_t show = out_ours.float_count < 24 ? out_ours.float_count : 24;
			for (int32_t i = 0; i < show; i++) {
				uint32_t bits;
				memcpy(&bits, &out_ours.floats[i], 4);
				printf("  word[%3d] = %08x  %14.6f\n", i, bits, out_ours.floats[i]);
			}
			for (int32_t i = 0; i + 4 <= out_ours.tex_bytes && i < 16; i += 4)
				printf("  tex[%4d] = %3u %3u %3u %3u\n", i, out_ours.tex[i],
				       out_ours.tex[i + 1], out_ours.tex[i + 2], out_ours.tex[i + 3]);
		}

		int32_t expect_n = 0, expect_bad = 0;
		for (int32_t e = 0; e < 16 && (cfg->expect[e].index || cfg->expect[e].bits ||
		                               cfg->expect[e].eps > 0); e++) {
			int32_t idx  = cfg->expect[e].index;
			bool    ok_e;
			expect_n++;
			if (cfg->expect[e].eps > 0) { // epsilon: float buffer word or texture byte
				float got = -1e30f;
				if (cfg->expect[e].tex) {
					if (idx < out_ours.tex_bytes) got = (float)out_ours.tex[idx];
				} else {
					if (idx < out_ours.float_count) got = out_ours.floats[idx];
				}
				ok_e = fabsf(got - cfg->expect[e].value) <= cfg->expect[e].eps;
				if (!ok_e && show_spirv_diff)
					printf("  expect %s[%4d] = %f +-%g, got %f\n",
					       cfg->expect[e].tex ? "tex" : "word", idx,
					       cfg->expect[e].value, cfg->expect[e].eps, got);
			} else {
				uint32_t bits = 0;
				if (idx < out_ours.float_count)
					memcpy(&bits, &out_ours.floats[idx], 4);
				ok_e = bits == cfg->expect[e].bits;
				if (!ok_e && show_spirv_diff)
					printf("  expect word[%4d] = %08x, got %08x\n",
					       idx, cfg->expect[e].bits, bits);
			}
			if (!ok_e) expect_bad++;
		}
		if (expect_n) {
			r.ok     = r.ok && expect_bad == 0;
			verified = true;
			len += snprintf(checks + len, sizeof(checks) - (size_t)len,
			                "%s%d/%d golden words", len ? " | " : "",
			                expect_n - expect_bad, expect_n);
		}

		if (cfg->radix_golden) {
			int32_t bad = radix_golden_diff(cfg, &out_ours, show_spirv_diff);
			r.ok     = r.ok && bad == 0;
			verified = true;
			len += snprintf(checks + len, sizeof(checks) - (size_t)len,
			                "%sradix golden %s (%d wrong)", len ? " | " : "",
			                bad == 0 ? "ok" : "FAILED", bad < 0 ? -1 : bad);
		}

		if (!verified) // no stronger tier applies: report the write coverage
			len += snprintf(checks + len, sizeof(checks) - (size_t)len,
			                "%s%d words + %d tex bytes (smoke only)", len ? " | " : "",
			                out_ours.float_count, out_ours.tex_bytes);

		// every tier: a run that changes nothing verified nothing
		r.ok = r.ok && out_ours.wrote;
		snprintf(r.message, sizeof(r.message), "compute%s: %s%s",
		         ref_ok ? "" : " (no reference)", checks,
		         out_ours.wrote ? "" : " (FAILED: outputs untouched)");
		svsl_arena_free(&arena);
		return r;
	}

	void   *ref = NULL;
	int32_t ref_size = 0;
	r.compiled_ref = compile_reference(&arena, shader_path, include_dir, &ref, &ref_size);
	if (!r.compiled_ref) {
		// SVSL-native dialect and post-glslang features have no reference to
		// compare against; corpus tests still compile + spirv-val them
		r.skipped = true;
		r.ok      = true;
		snprintf(r.message, sizeof(r.message), "no reference (skshaderc can't compile)");
		svsl_arena_free(&arena);
		return r;
	}
	if (show_spirv_diff)
		spirv_diff(ref, ref_size, ours, ours_size);

	// engine-owned constant buffers (b2+) bound as renderer globals, shared by
	// both renders so the parameters are identical
	sks_info_t   info = {0};
	skr_buffer_t globals[8];
	int32_t      global_indices[8];
	int32_t      global_count = 0;
	if (sks_parse(ours, ours_size, &info)) {
		global_count = make_global_buffers(&info, globals, global_indices, 8, false);
		for (int32_t g = 0; g < global_count; g++)
			skr_renderer_set_global_constants(info.bufs[global_indices[g]].slot, &globals[g]);
	}

	static uint8_t pixels_ref [2 * VIEW_SIZE * VIEW_SIZE * 4];
	static uint8_t pixels_ours[2 * VIEW_SIZE * VIEW_SIZE * 4];
	static uint8_t mv_ref     [2 * VIEW_SIZE * VIEW_SIZE * 4];
	static uint8_t mv_ours    [2 * VIEW_SIZE * VIEW_SIZE * 4];
	bool rendered = uses_subpass == subpass_msaa
		// SubpassInputMS shaders run as the manual MSAA resolve of the baseline
		? render_resolve_with(scene, ref, ref_size, &scene->color_a, pixels_ref) &&
		  render_resolve_with(scene, ours, ours_size, &scene->color_b, pixels_ours)
		: uses_subpass == subpass_plain
		// SubpassInput shaders run as a postfx subpass over the baseline sphere
		? render_postfx_with(scene, ref, ref_size, &scene->color_a, &scene->depth_a, pixels_ref) &&
		  render_postfx_with(scene, ours, ours_size, &scene->color_b, &scene->depth_b, pixels_ours)
		: render_with(scene, ref, ref_size, kinds, kind_count, false,
		              &scene->color_a, &scene->depth_a, pixels_ref) &&
		  render_with(scene, ours, ours_size, kinds, kind_count, false,
		              &scene->color_b, &scene->depth_b, pixels_ours);
	// SV_ViewID shaders also render as a stereo pair in one multiview pass
	bool multiview = rendered && !writes_layer && !uses_subpass;
	if (multiview)
		rendered =
			render_with(scene, ref, ref_size, kinds, kind_count, true,
			            &scene->color_mv_a, &scene->depth_mv_a, mv_ref) &&
			render_with(scene, ours, ours_size, kinds, kind_count, true,
			            &scene->color_mv_b, &scene->depth_mv_b, mv_ours);

	for (int32_t g = 0; g < global_count; g++) {
		skr_renderer_set_global_constants(info.bufs[global_indices[g]].slot, NULL);
		skr_buffer_destroy(&globals[g]);
	}
	if (!rendered) {
		snprintf(r.message, sizeof(r.message), "render failed");
		svsl_arena_free(&arena);
		return r;
	}
	r.rendered = true;

	double total = 0, mean = 0, variance = 0;
	int32_t rgb_count = 0;
	for (int32_t i = 0; i < VIEW_SIZE * VIEW_SIZE * 4; i++) {
		total += fabs((double)pixels_ref[i] - (double)pixels_ours[i]) / 255.0;
		if ((i & 3) != 3) { mean += pixels_ref[i] / 255.0; rgb_count++; }
	}
	mean /= rgb_count;
	for (int32_t i = 0; i < VIEW_SIZE * VIEW_SIZE * 4; i += 97) { // sampled variance
		if ((i & 3) == 3) continue; // alpha is constant; RGB decides blankness
		variance += (pixels_ref[i] / 255.0 - mean) * (pixels_ref[i] / 255.0 - mean);
	}
	r.avg_error    = total / (VIEW_SIZE * VIEW_SIZE * 4);
	r.ref_variance = variance;
	r.ok           = r.avg_error < 0.01;

	double mv_error = 0;
	if (multiview) {
		for (int32_t i = 0; i < 2 * VIEW_SIZE * VIEW_SIZE * 4; i++)
			mv_error += fabs((double)mv_ref[i] - (double)mv_ours[i]) / 255.0;
		mv_error /= 2 * VIEW_SIZE * VIEW_SIZE * 4;
		r.ok = r.ok && mv_error < 0.01;
	}

	if (opt_out_dir) {
		const char *slash = strrchr(shader_path, '/');
		const char *stem  = slash ? slash + 1 : shader_path;
		int32_t     len   = (int32_t)(strchr(stem, '.') ? strchr(stem, '.') - stem
		                                                : (int64_t)strlen(stem));
		char path[1024];
		snprintf(path, sizeof(path), "%s/%.*s_ref.png", opt_out_dir, len, stem);
		write_png(path, pixels_ref);
		snprintf(path, sizeof(path), "%s/%.*s_svsl.png", opt_out_dir, len, stem);
		write_png(path, pixels_ours);
		if (multiview) { // second view of the stereo pair (first ≈ the single view)
			snprintf(path, sizeof(path), "%s/%.*s_view1_ref.png", opt_out_dir, len, stem);
			write_png(path, mv_ref + VIEW_SIZE * VIEW_SIZE * 4);
			snprintf(path, sizeof(path), "%s/%.*s_view1_svsl.png", opt_out_dir, len, stem);
			write_png(path, mv_ours + VIEW_SIZE * VIEW_SIZE * 4);
		}
	}
	if (multiview)
		snprintf(r.message, sizeof(r.message), "avg error %.5f | multiview %.5f%s",
		         r.avg_error, mv_error, r.ref_variance < 0.0001 ? " (WARNING: blank render)" : "");
	else
		snprintf(r.message, sizeof(r.message), "avg error %.5f%s",
		         r.avg_error, r.ref_variance < 0.0001 ? " (WARNING: blank render)" : "");
	svsl_arena_free(&arena);
	return r;
}

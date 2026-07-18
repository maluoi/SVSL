// SKS container tests: our writer's metadata must match what skshaderc
// produces for the same shader, record by record (both emit v11 — one version
// at a time). The reference comparison runs when the local skshaderc build is
// present; the structural checks always run.

#include "test.h"

#include "back/emit_spirv.h"
#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "ir/ir.h"
#include "out/sks_write.h"
#include "sema/sema.h"
#include "util/arena.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SVSL_TEST_DIR
#define SVSL_TEST_DIR "."
#endif
#define SKSHADERC "~/SK/sk_renderer/build/skshaderc/skshaderc"

// --- tiny SKS reader (metadata only) ----------------------------------------------

typedef struct sks_var_t {
	char     name[32], extra[64], type_name[32];
	uint32_t offset, size;
	uint16_t type, type_count;
} sks_var_t;

typedef struct sks_buf_t {
	char      name[32];
	uint8_t   space, stage_bits, register_type;
	uint16_t  slot;
	uint32_t  size, var_count, defaults_size;
	const uint8_t *defaults;
	sks_var_t vars[32];
} sks_buf_t;

typedef struct sks_res_t {
	char     name[32], value[64], tags[64];
	uint16_t slot;
	uint8_t  stage_bits, register_type;
	uint32_t element_size;
	uint8_t  shape;        // dim/arrayed/ms/comparison (0 = unreported)
	uint8_t  image_format; // SpvImageFormat of storage images
} sks_res_t;

typedef struct sks_spec_t {
	char     name[32];
	uint32_t id, default_bits;
	uint16_t type;
	uint8_t  stage_bits;
} sks_spec_t;

typedef struct sks_file_t {
	uint16_t   version;
	uint64_t   features;
	uint32_t   stage_count, buffer_count, resource_count, spec_count, wave_size;
	uint32_t   tile_apron[2];
	int32_t    vertex_input_count;
	char       name[256];
	sks_buf_t  bufs[8];
	sks_res_t  res[16];
	sks_spec_t specs[8];
	uint8_t    vins[16][11]; // format(4) count(1) semantic(4) slot(1) location(1)
} sks_file_t;

static bool sks_read(const uint8_t *data, int32_t size, sks_file_t *out) {
	const uint8_t *p = data;
	#define TAKE(dst, n) do { if ((p - data) + (n) > size) return false; \
	                          memcpy(dst, p, n); p += (n); } while (0)
	char tag[8];
	TAKE(tag, 8);
	if (memcmp(tag, "SKSHADER", 8) != 0) return false;
	TAKE(&out->version, 2);
	TAKE(&out->stage_count, 4);
	TAKE(out->name, 256);
	TAKE(&out->buffer_count, 4);
	TAKE(&out->resource_count, 4);
	if (out->version != 11) return false; // one version at a time
	TAKE(&out->vertex_input_count, 4);
	TAKE(&out->spec_count, 4);
	TAKE(&out->features, 8);
	p += 8;  // reserved features word
	p += 24; // ops
	TAKE(&out->wave_size, 4);
	TAKE(out->tile_apron, 8); // v11+
	for (uint32_t b = 0; b < out->buffer_count && b < 8; b++) {
		sks_buf_t *buf = &out->bufs[b];
		TAKE(buf->name, 32);
		TAKE(&buf->space, 1);
		TAKE(&buf->slot, 2);
		TAKE(&buf->stage_bits, 1);
		TAKE(&buf->register_type, 1);
		TAKE(&buf->size, 4);
		TAKE(&buf->var_count, 4);
		TAKE(&buf->defaults_size, 4);
		buf->defaults = p;
		p += buf->defaults_size;
		for (uint32_t v = 0; v < buf->var_count && v < 32; v++) {
			sks_var_t *var = &buf->vars[v];
			TAKE(var->name, 32); TAKE(var->extra, 64); TAKE(var->type_name, 32);
			TAKE(&var->offset, 4); TAKE(&var->size, 4);
			TAKE(&var->type, 2); TAKE(&var->type_count, 2);
		}
	}
	for (int32_t v = 0; v < out->vertex_input_count && v < 16; v++)
		TAKE(out->vins[v], 11);
	for (uint32_t r = 0; r < out->resource_count && r < 16; r++) {
		sks_res_t *res = &out->res[r];
		TAKE(res->name, 32); TAKE(res->value, 64); TAKE(res->tags, 64);
		TAKE(&res->slot, 2); TAKE(&res->stage_bits, 1); TAKE(&res->register_type, 1);
		TAKE(&res->element_size, 4);
		TAKE(&res->shape, 1); TAKE(&res->image_format, 1);
		p += 2; // reserved
	}
	for (uint32_t s = 0; s < out->spec_count && s < 8; s++) {
		sks_spec_t *spec = &out->specs[s];
		TAKE(spec->name, 32);
		TAKE(&spec->id, 4); TAKE(&spec->default_bits, 4);
		TAKE(&spec->type, 2); TAKE(&spec->stage_bits, 1);
	}
	#undef TAKE
	return true;
}

// --- compile helper -----------------------------------------------------------------

static char *sks_read_file(svsl_arena_t *arena, const char *path, int32_t *out_len) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *data = svsl_arena_alloc(arena, (size_t)fsize + 1);
	size_t got = fread(data, 1, (size_t)fsize, f);
	fclose(f);
	data[got] = '\0';
	if (out_len) *out_len = (int32_t)got;
	return data;
}

static svsl_include_src_t sks_include(void *user, const char *path, const char *requester) {
	svsl_arena_t *arena = user;
	char          full[1024];
	(void)requester;
	snprintf(full, sizeof(full), "%s/shaders/include/%s", SVSL_TEST_DIR, path);
	int32_t len;
	char   *content = sks_read_file(arena, full, &len);
	if (!content) return (svsl_include_src_t){0};
	// .path must outlive this callback (the pp copies it); `full` is stack-local
	char *path_out = svsl_arena_strndup(arena, full, strlen(full));
	return (svsl_include_src_t){ .content = content, .length = len, .path = path_out };
}

static bool compile_sks(svsl_arena_t *arena, const char *shader, svsl_sks_blob_t *out_blob) {
	char path[1024];
	snprintf(path, sizeof(path), "%s/shaders/%s", SVSL_TEST_DIR, shader);
	char *src = sks_read_file(arena, path, NULL);
	if (!src) return false;

	svsl_diag_list_t  diags = {0};
	svsl_pp_options_t popt  = { .include_cb = sks_include, .include_user = arena };
	svsl_pp_result_t  pp;
	svsl_pp_run(arena, src, path, &popt, &pp, &diags);
	svsl_token_list_t tokens = {0};
	svsl_lex(arena, &pp, &tokens, &diags);
	svsl_ast_t *ast = svsl_parse(arena, &tokens, &diags);
	svsl_program_t prog;
	svsl_sema_run(arena, ast, &pp, path, NULL, &prog, &diags);
	if (diags.error_count) return false;
	svsl_ir_module_t ir = {0};
	svsl_ir_build(arena, &prog, svsl_opt_default, &ir, &diags);
	if (diags.error_count) return false;
	svsl_spirv_blob_t blobs[8] = {0};
	for (int32_t i = 0; i < ir.func_count && i < 8; i++)
		if (!svsl_spirv_emit(arena, &prog, &ir.funcs[i], &blobs[i], &diags)) return false;
	svsl_sks_write(arena, &prog, &ir, blobs, out_blob);
	return true;
}

// --- structural checks (always run) ------------------------------------------------------

static void test_sks_structure(void) {
	svsl_arena_t arena = {0};

	svsl_sks_blob_t blob = {0};
	TEST_CHECK(compile_sks(&arena, "builtin/shader_builtin_unlit.hlsl", &blob));
	sks_file_t f = {0};
	TEST_CHECK(sks_read(blob.bytes, blob.size, &f));

	TEST_CHECK(f.version == 11);
	TEST_CHECK(f.stage_count == 2);
	TEST_CHECK(strcmp(f.name, "sk/unlit") == 0);
	TEST_CHECK(f.buffer_count == 2);
	TEST_CHECK(f.vertex_input_count == 3); // norm is unused and dropped

	// v10 locations mirror the SPIR-V: dropped `norm` still consumed location 1,
	// so the surviving pos/uv/col sit at 0/2/3 — the gap must be visible
	TEST_CHECK(f.vins[0][10] == 0);
	TEST_CHECK(f.vins[1][10] == 2);
	TEST_CHECK(f.vins[2][10] == 3);

	// $Global first (slot order), with baked defaults
	TEST_CHECK(strcmp(f.bufs[0].name, "$Global") == 0);
	TEST_CHECK(f.bufs[0].slot == 0 && f.bufs[0].register_type == 3);
	TEST_CHECK(f.bufs[0].stage_bits == 1); // vertex only
	TEST_CHECK(f.bufs[0].defaults_size == 32);
	float def[4];
	memcpy(def, f.bufs[0].defaults, 16);
	TEST_CHECK(def[0] == 1 && def[3] == 1);

	TEST_CHECK(strcmp(f.bufs[1].name, "stereokit_buffer") == 0);
	TEST_CHECK(f.bufs[1].size == 1920 && f.bufs[1].slot == 1);
	TEST_CHECK(f.bufs[1].vars[4].offset == 1536); // sk_lighting_sh
	TEST_CHECK(f.bufs[1].vars[4].type == 4 && f.bufs[1].vars[4].type_count == 28);

	// resources: fused diffuse at t0+100, pixel stage; sk_inst at t12+100, vertex
	bool found_diffuse = false, found_inst = false;
	for (uint32_t r = 0; r < f.resource_count; r++) {
		if (strcmp(f.res[r].name, "diffuse") == 0) {
			found_diffuse = true;
			TEST_CHECK(f.res[r].slot == 100 && f.res[r].register_type == 4);
			TEST_CHECK(f.res[r].stage_bits == 2);
			TEST_CHECK(strcmp(f.res[r].value, "white") == 0);
		}
		if (strcmp(f.res[r].name, "sk_inst") == 0) {
			found_inst = true;
			TEST_CHECK(f.res[r].slot == 112 && f.res[r].register_type == 5);
			TEST_CHECK(f.res[r].element_size == 80);
		}
	}
	TEST_CHECK(found_diffuse && found_inst);

	// v9 extras: unlit renders per-view (SV_ViewID → multiview) and needs
	// nothing else; its texture reports a plain 2D shape while the structured
	// buffer stays unreported
	TEST_CHECK(f.features == (1ull << 7));
	for (uint32_t r = 0; r < f.resource_count; r++) {
		if (strcmp(f.res[r].name, "diffuse") == 0) TEST_CHECK(f.res[r].shape == 1);
		if (strcmp(f.res[r].name, "sk_inst") == 0) TEST_CHECK(f.res[r].shape == 0);
	}

	// v10 location metadata: a stripped middle input leaves a hole, an unused
	// standalone param is stripped like glslang does, and [location(N)] records
	// the explicit value (see the shader's comment for the expected layout)
	svsl_sks_blob_t vl = {0};
	TEST_CHECK(compile_sks(&arena, "checks/check_vertex_locations.hlsl", &vl));
	sks_file_t fvl = {0};
	TEST_CHECK(sks_read(vl.bytes, vl.size, &fvl));
	TEST_CHECK(fvl.vertex_input_count == 3); // norm stripped
	TEST_CHECK(fvl.vins[0][10] == 0);        // pos
	TEST_CHECK(fvl.vins[1][10] == 2);        // uv  (norm consumed 1)
	TEST_CHECK(fvl.vins[2][10] == 6);        // col, explicit [[vk::location(6)]]

	// a wave shader must raise the subgroup feature bit
	svsl_sks_blob_t wave = {0};
	TEST_CHECK(compile_sks(&arena, "checks/check_wave.hlsl", &wave));
	sks_file_t fw = {0};
	TEST_CHECK(sks_read(wave.bytes, wave.size, &fw));
	TEST_CHECK((fw.features & (1ull << 5)) != 0); // subgroups
	TEST_CHECK((fw.features & (1ull << 63)) == 0); // nothing unknown

	// spec-constant expressions stay specializable (OpSpecConstantOp = opcode
	// 52 in the blob) and float atomics raise feature bit 15
	svsl_sks_blob_t fa = {0};
	TEST_CHECK(compile_sks(&arena, "ported/78_spec_atomics.hlsl", &fa));
	sks_file_t ffa = {0};
	TEST_CHECK(sks_read(fa.bytes, fa.size, &ffa));
	TEST_CHECK((ffa.features & (1ull << 15)) != 0); // float atomics
	TEST_CHECK((ffa.features & (1ull << 63)) == 0); // nothing unknown
	bool has_spec_op = false;
	for (int32_t b = 0; b + 4 <= fa.size; b += 4) {
		uint32_t word;
		memcpy(&word, fa.bytes + b, 4);
		if ((word & 0xFFFF) == 52 && (word >> 16) >= 4) has_spec_op = true;
	}
	TEST_CHECK(has_spec_op);

	// texture shapes: 1D, 3D, cube, and 2D-array from the texture-types corpus
	svsl_sks_blob_t shapes = {0};
	TEST_CHECK(compile_sks(&arena, "ported/25_texture_types.svsl", &shapes));
	sks_file_t fs = {0};
	TEST_CHECK(sks_read(shapes.bytes, shapes.size, &fs));
	for (uint32_t r = 0; r < fs.resource_count; r++) {
		if (strcmp(fs.res[r].name, "tex1D") == 0)      TEST_CHECK(fs.res[r].shape == 4);
		if (strcmp(fs.res[r].name, "tex3D") == 0)      TEST_CHECK(fs.res[r].shape == 2);
		if (strcmp(fs.res[r].name, "texCube") == 0)    TEST_CHECK(fs.res[r].shape == 3);
		if (strcmp(fs.res[r].name, "tex2DArray") == 0) TEST_CHECK(fs.res[r].shape == (1 | (1 << 3)));
	}

	// block-form storage buffers must reach the resource records (member
	// access marks the buffer; its resource twin must light up with it), and
	// the container's entry point is renamed to the canonical stage name
	// ("cs" here — the source entry is `main`) so the runtime can bind it
	svsl_sks_blob_t sb = {0};
	TEST_CHECK(compile_sks(&arena, "ported/21_storage_buffer.svsl", &sb));
	sks_file_t fsb = {0};
	TEST_CHECK(sks_read(sb.bytes, sb.size, &fsb));
	bool found_in = false, found_out = false;
	for (uint32_t r = 0; r < fsb.resource_count; r++) {
		if (strcmp(fsb.res[r].name, "InputData") == 0)  found_in = true;
		if (strcmp(fsb.res[r].name, "OutputData") == 0) found_out = true;
	}
	TEST_CHECK(found_in && found_out);
	int32_t spv_at = -1; // the stage blob starts at the SPIR-V magic (unaligned)
	for (int32_t b = 0; b + 4 <= sb.size && spv_at < 0; b++)
		if (memcmp(sb.bytes + b, "\x03\x02\x23\x07", 4) == 0) spv_at = b;
	TEST_CHECK(spv_at >= 0);
	bool entry_canon = false;
	for (int32_t b = spv_at + 5 * 4; b + 16 <= sb.size; ) {
		uint32_t word;
		memcpy(&word, sb.bytes + b, 4);
		if ((word >> 16) == 0) break;
		if ((word & 0xFFFF) == 15) { // OpEntryPoint: name string is word 3
			entry_canon = memcmp(sb.bytes + b + 12, "cs\0\0", 4) == 0;
			break;
		}
		b += (int32_t)(word >> 16) * 4;
	}
	TEST_CHECK(entry_canon);

	svsl_arena_free(&arena);
}

// bool buffer members are stored as uint 0/1 and reflect as var_uint — the type
// material_set_bool passes. Spec-const bools stay OpTypeBool and reflect as
// var_int, matching sksc's reflection of both.
static void test_sks_bool_params(void) {
	svsl_arena_t arena = {0};

	svsl_sks_blob_t blob = {0};
	TEST_CHECK(compile_sks(&arena, "checks/check_bool_param.hlsl", &blob));
	sks_file_t f = {0};
	TEST_CHECK(sks_read(blob.bytes, blob.size, &f));
	TEST_CHECK(f.buffer_count == 1 && strcmp(f.bufs[0].name, "$Global") == 0);
	TEST_CHECK(f.bufs[0].var_count == 3);
	TEST_CHECK(strcmp(f.bufs[0].vars[0].name, "on_off") == 0);
	TEST_CHECK(f.bufs[0].vars[0].type == 2 && f.bufs[0].vars[0].type_count == 1); // var_uint
	TEST_CHECK(strcmp(f.bufs[0].vars[0].type_name, "bool") == 0); // source truth kept
	TEST_CHECK(f.bufs[0].vars[1].type == 2 && f.bufs[0].vars[1].type_count == 2); // bool2
	TEST_CHECK(f.bufs[0].vars[2].type == 0); // struct member: var_none, named by type_name
	uint32_t def = 0; // `= true` bakes a uint 1 default
	TEST_CHECK(f.bufs[0].defaults_size >= 4);
	if (f.bufs[0].defaults_size >= 4) memcpy(&def, f.bufs[0].defaults, 4);
	TEST_CHECK(def == 1);

	svsl_sks_blob_t sa = {0};
	TEST_CHECK(compile_sks(&arena, "ported/78_spec_atomics.hlsl", &sa));
	sks_file_t fa = {0};
	TEST_CHECK(sks_read(sa.bytes, sa.size, &fa));
	bool found_half_rate = false;
	for (uint32_t s = 0; s < fa.spec_count && s < 8; s++)
		if (strcmp(fa.specs[s].name, "HALF_RATE") == 0) {
			found_half_rate = true;
			TEST_CHECK(fa.specs[s].type == 1); // var_int (VkBool32)
			TEST_CHECK(fa.specs[s].default_bits == 0);
		}
	TEST_CHECK(found_half_rate);

	svsl_arena_free(&arena);
}

// scalarBlockLayout has no SPIR-V capability, so feature-mask bit 16 is the only
// machine-readable signal that a pack1/pack8 layout broke core relaxed rules
static void test_sks_scalar_layout_feature(void) {
	svsl_arena_t arena = {0};

	svsl_sks_blob_t blob = {0};
	TEST_CHECK(compile_sks(&arena, "checks/check_pack_scalar.hlsl", &blob));
	sks_file_t f = {0};
	TEST_CHECK(sks_read(blob.bytes, blob.size, &f));
	TEST_CHECK((f.features & (1ull << 16)) != 0); // pack1 + straddling float4

	svsl_sks_blob_t plain = {0};
	TEST_CHECK(compile_sks(&arena, "checks/check_pack_layout.hlsl", &plain));
	sks_file_t fp = {0};
	TEST_CHECK(sks_read(plain.bytes, plain.size, &fp));
	TEST_CHECK((fp.features & (1ull << 16)) == 0); // C-packed default stays core-legal

	// object-form buffer whose element straddles only from element 1 onward: the
	// straddle is invisible at offset 0, so it is caught only by walking element
	// stride phases — must raise bit 16 like the equivalent block form
	svsl_sks_blob_t stride = {0};
	TEST_CHECK(compile_sks(&arena, "checks/check_pack_stride.hlsl", &stride));
	sks_file_t fst = {0};
	TEST_CHECK(sks_read(stride.bytes, stride.size, &fst));
	TEST_CHECK((fst.features & (1ull << 16)) != 0); // pack1 + element-1 straddle

	svsl_arena_free(&arena);
}

// --- reference comparison (runs when skshaderc is available) -----------------------------

static bool compare_with_reference(svsl_arena_t *arena, const char *shader) {
	char cmd[2048];
	snprintf(cmd, sizeof(cmd),
	         SKSHADERC " -f -i %s/shaders/include -o svsl_ref_tmp.sks %s/shaders/%s > /dev/null 2>&1",
	         SVSL_TEST_DIR, SVSL_TEST_DIR, shader);
	if (system(cmd) != 0) { printf("  skshaderc failed on %s\n", shader); return false; }

	int32_t  ref_size = 0;
	char    *ref_data = sks_read_file(arena, "svsl_ref_tmp.sks", &ref_size);
	remove("svsl_ref_tmp.sks");
	if (!ref_data) return false;

	sks_file_t ref = {0}, ours = {0};
	if (!sks_read((const uint8_t *)ref_data, ref_size, &ref)) return false;
	svsl_sks_blob_t blob = {0};
	if (!compile_sks(arena, shader, &blob)) { printf("  svslc failed on %s\n", shader); return false; }
	if (!sks_read(blob.bytes, blob.size, &ours)) return false;

	bool ok = true;
	#define CMP(cond, what) do { if (!(cond)) { printf("  %s: %s differs\n", shader, what); ok = false; } } while (0)
	CMP(strcmp(ref.name, ours.name) == 0, "name");
	CMP(ref.stage_count == ours.stage_count, "stage_count");
	CMP(ref.wave_size == ours.wave_size, "wave_size");
	CMP(ref.tile_apron[0] == ours.tile_apron[0] && ref.tile_apron[1] == ours.tile_apron[1], "tile_apron");
	CMP(ref.buffer_count == ours.buffer_count, "buffer_count");
	CMP(ref.resource_count == ours.resource_count, "resource_count");
	CMP(ref.vertex_input_count == ours.vertex_input_count, "vertex_input_count");
	for (int32_t v = 0; v < ref.vertex_input_count && v < 16; v++)
		CMP(memcmp(ref.vins[v], ours.vins[v], 11) == 0, "vertex input");
	for (uint32_t b = 0; b < ref.buffer_count && b < 8; b++) {
		sks_buf_t *rb = &ref.bufs[b], *ob = NULL;
		for (uint32_t k = 0; k < ours.buffer_count && k < 8; k++)
			if (strcmp(ours.bufs[k].name, rb->name) == 0) ob = &ours.bufs[k];
		if (!ob) { printf("  %s: missing buffer %s\n", shader, rb->name); ok = false; continue; }
		CMP(rb->space == ob->space && rb->slot == ob->slot &&
		    rb->stage_bits == ob->stage_bits && rb->register_type == ob->register_type &&
		    rb->size == ob->size && rb->var_count == ob->var_count, rb->name);
		CMP(rb->defaults_size == ob->defaults_size &&
		    memcmp(rb->defaults, ob->defaults, rb->defaults_size) == 0, "defaults");
		for (uint32_t v = 0; v < rb->var_count && v < 32; v++)
			CMP(memcmp(&rb->vars[v], &ob->vars[v], sizeof(sks_var_t)) == 0, rb->vars[v].name);
	}
	for (uint32_t r = 0; r < ref.resource_count && r < 16; r++) {
		sks_res_t *rr = &ref.res[r], *or_ = NULL;
		for (uint32_t k = 0; k < ours.resource_count && k < 16; k++)
			if (strcmp(ours.res[k].name, rr->name) == 0) or_ = &ours.res[k];
		if (!or_) { printf("  %s: missing resource %s\n", shader, rr->name); ok = false; continue; }
		// shape/format are ours to fill richly; sksc reports 0 = unreported
		CMP(strcmp(rr->value, or_->value) == 0 && strcmp(rr->tags, or_->tags) == 0 &&
		    rr->slot == or_->slot && rr->stage_bits == or_->stage_bits &&
		    rr->register_type == or_->register_type &&
		    rr->element_size == or_->element_size, rr->name);
	}
	#undef CMP
	return ok;
}

static void test_sks_reference(void) {
	// only when the local reference compiler exists
	if (system("test -x " SKSHADERC) != 0) {
		printf("  (skshaderc not found; reference comparison skipped)\n");
		return;
	}
	svsl_arena_t arena = {0};
	static const char *shaders[] = {
		"builtin/shader_builtin_unlit.hlsl",
		"builtin/shader_builtin_pbr.hlsl",
		"builtin/shader_builtin_font.hlsl",
		"builtin/shader_builtin_sh_compute.hlsl",
		"examples/skt_default_lighting.hlsl", // struct-typed cbuffer member
		"examples/basic_shadow.hlsl",
		"checks/check_vertex_locations.hlsl", // v10 location gaps + explicit [location]
	};
	for (int32_t i = 0; i < (int32_t)(sizeof(shaders) / sizeof(shaders[0])); i++)
		TEST_CHECK(compare_with_reference(&arena, shaders[i]));
	svsl_arena_free(&arena);
}

void test_sks(void) {
	test_sks_structure();
	test_sks_bool_params();
	test_sks_scalar_layout_feature();
	test_sks_reference();
}

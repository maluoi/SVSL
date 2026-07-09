// Public API tests: everything here goes through <svsl/svsl.h> only — no internal
// headers — so it doubles as a check that the public surface is self-sufficient.
// The headline case is a compile → SKS bytes → svsl_sks_parse round trip.

#include "test.h"

#include <svsl/svsl.h>

#include <string.h>

static const char *k_shader =
	"//--name = api/test\n"
	"cbuffer Params : register(b0) {\n"
	"	float4 tint;\n"
	"	float  amount;\n"
	"};\n"
	"Texture2D<float4> albedo : register(t0);\n"
	"SamplerState      albedo_s : register(s0);\n"
	"struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
	"VSOut vs(float3 p : POSITION, float2 uv : TEXCOORD0) {\n"
	"	VSOut o; o.pos = float4(p, 1); o.uv = uv; return o;\n"
	"}\n"
	"float4 ps(VSOut i) : SV_Target {\n"
	"	return albedo.Sample(albedo_s, i.uv) * tint * amount;\n"
	"}\n";

static void test_api_compile(void) {
	svsl_result_t *r = svsl_compile(&(svsl_source_t){ .text = k_shader, .filename = "api_test.hlsl" }, NULL);
	TEST_CHECK(r != NULL);
	if (!r) return;

	TEST_CHECK(r->ok);
	TEST_CHECK(r->error_count == 0);
	TEST_CHECK(r->stage_count == 2); // vs + ps

	bool have_vs = false, have_ps = false;
	for (int32_t i = 0; i < r->stage_count; i++) {
		const svsl_stage_output_t *s = &r->stages[i];
		TEST_CHECK(s->spirv != NULL && s->spirv_word_count > 4);
		TEST_CHECK(s->spirv[0] == 0x07230203u); // SPIR-V magic
		if (s->stage == svsl_stage_vertex) { have_vs = true; TEST_CHECK(strcmp(s->entry, "vs") == 0); }
		if (s->stage == svsl_stage_pixel)  { have_ps = true; TEST_CHECK(strcmp(s->entry, "ps") == 0); }
	}
	TEST_CHECK(have_vs && have_ps);

	// text products are non-empty
	TEST_CHECK(svsl_result_reflection(r) != NULL);
	TEST_CHECK(svsl_result_ir(r) != NULL);
	TEST_CHECK(svsl_result_header(r, "api_test") != NULL);

	svsl_result_free(r);
}

static void test_api_diagnostics(void) {
	// a deliberately broken shader reports errors and produces no stages
	svsl_result_t *r = svsl_compile(&(svsl_source_t){ .text = "float4 ps() : SV_Target { return nope; }\n" }, NULL);
	TEST_CHECK(r && !r->ok);
	TEST_CHECK(r && r->error_count > 0);
	TEST_CHECK(r && r->stage_count == 0);
	if (r) {
		bool saw_error = false;
		for (int32_t i = 0; i < r->diagnostic_count; i++)
			if (r->diagnostics[i].severity == svsl_severity_error) saw_error = true;
		TEST_CHECK(saw_error);
		TEST_CHECK(svsl_result_sks(r).data == NULL); // no product on failure
	}
	svsl_result_free(r);
}

// compile → SKS bytes → parse them back and check the structured view
static void test_api_sks_roundtrip(void) {
	svsl_result_t *r = svsl_compile(&(svsl_source_t){ .text = k_shader, .filename = "api_test.hlsl" }, NULL);
	TEST_CHECK(r && r->ok);
	if (!r || !r->ok) { svsl_result_free(r); return; }

	svsl_bytes_t sks = svsl_result_sks(r);
	TEST_CHECK(sks.data != NULL && sks.size > 0);

	svsl_sks_file_t *f = svsl_sks_parse(sks.data, sks.size);
	TEST_CHECK(f != NULL);
	if (f) {
		TEST_CHECK(f->version == SVSL_SKS_VERSION);
		TEST_CHECK(strcmp(f->name, "api/test") == 0);
		TEST_CHECK(f->stage_count == 2);

		// v10: each vertex input carries its SPIR-V location
		TEST_CHECK(f->vertex_input_count == 2);
		TEST_CHECK(f->vertex_inputs[0].location == 0); // p : POSITION
		TEST_CHECK(f->vertex_inputs[1].location == 1); // uv : TEXCOORD0

		// the $Global-style constant buffer survives with its members
		bool found_params = false;
		for (int32_t b = 0; b < f->buffer_count; b++) {
			if (strcmp(f->buffers[b].name, "Params") != 0) continue;
			found_params = true;
			TEST_CHECK(f->buffers[b].register_type == svsl_sks_register_constant);
			TEST_CHECK(f->buffers[b].var_count == 2);
			TEST_CHECK(strcmp(f->buffers[b].vars[0].name, "tint") == 0);
			TEST_CHECK(f->buffers[b].vars[0].type == svsl_sks_vartype_float);
			TEST_CHECK(f->buffers[b].vars[0].type_count == 4);
		}
		TEST_CHECK(found_params);

		// the sampled texture is present as a resource
		bool found_albedo = false;
		for (int32_t i = 0; i < f->resource_count; i++)
			if (strcmp(f->resources[i].name, "albedo") == 0) {
				found_albedo = true;
				TEST_CHECK(f->resources[i].register_type == svsl_sks_register_texture);
			}
		TEST_CHECK(found_albedo);

		// each parsed stage carries a valid SPIR-V module, matching the compile
		TEST_CHECK(f->stages[0].spirv[0] == 0x07230203u);
		int32_t words_compiled = 0, words_parsed = 0;
		for (int32_t i = 0; i < r->stage_count; i++)   words_compiled += r->stages[i].spirv_word_count;
		for (int32_t i = 0; i < f->stage_count; i++)   words_parsed   += f->stages[i].spirv_word_count;
		TEST_CHECK(words_compiled == words_parsed);

		svsl_sks_free(f);
	}
	svsl_result_free(r);

	// garbage in → NULL, not a crash
	TEST_CHECK(svsl_sks_parse("not an sks file at all", 22) == NULL);
	TEST_CHECK(svsl_sks_parse(NULL, 0) == NULL);
}

void test_api(void) {
	test_api_compile();
	test_api_diagnostics();
	test_api_sks_roundtrip();
}

// QCOM extension tests (VK_QCOM_image_processing[2], VK_QCOM_tile_shading):
// no desktop runtime implements these, so the emitted words are the testable
// surface — capabilities, extensions, decorations, storage classes, execution
// modes, the SPIR-V 1.4 bump for image processing, and the sks feature bits.

#include "test.h"

#include "back/emit_spirv.h"
#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "ir/ir.h"
#include "out/sks_write.h"
#include "sema/sema.h"
#include "util/arena.h"
#include "../vendor/spirv.h"

#include <string.h>

typedef struct qcom_case_t {
	svsl_arena_t      arena;
	svsl_program_t    program;
	svsl_ir_module_t  ir;
	svsl_spirv_blob_t blobs[4];
	int32_t           blob_count;
	svsl_diag_list_t  diags;
	int32_t           warnings;
} qcom_case_t;

static void qcom_compile(qcom_case_t *c, const char *src) {
	memset(c, 0, sizeof(*c));
	svsl_pp_result_t pp;
	svsl_pp_run(&c->arena, src, "test_qcom.hlsl", NULL, &pp, &c->diags);
	svsl_token_list_t tokens = {0};
	svsl_lex(&c->arena, &pp, &tokens, &c->diags);
	svsl_ast_t *ast = svsl_parse(&c->arena, &tokens, &c->diags);
	svsl_sema_run(&c->arena, ast, &pp, "test_qcom.hlsl", NULL, &c->program, &c->diags);
	if (c->diags.error_count == 0)
		svsl_ir_build(&c->arena, &c->program, svsl_opt_default, &c->ir, &c->diags);
	if (c->diags.error_count == 0)
		for (int32_t i = 0; i < c->ir.func_count && i < 4; i++)
			if (svsl_spirv_emit(&c->arena, &c->program, &c->ir.funcs[i], &c->blobs[c->blob_count], &c->diags))
				c->blob_count++;
	for (int32_t i = 0; i < c->diags.count; i++)
		if (c->diags.items[i].severity == svsl_severity_warning) c->warnings++;
}

// counts instructions where opcode matches and (when >= 0) the given operand
// words match; op_a/op_b index from the first word after the opcode word
static int32_t count_insts(const svsl_spirv_blob_t *b, SpvOp op, int64_t op_a, int64_t op_b) {
	int32_t n = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == (uint32_t)op &&
		    (op_a < 0 || (wc > 1 && b->words[i + 1] == (uint32_t)op_a)) &&
		    (op_b < 0 || (wc > 2 && b->words[i + 2] == (uint32_t)op_b)))
			n++;
		i += (int32_t)wc;
	}
	return n;
}

static int32_t count_decoration(const svsl_spirv_blob_t *b, SpvDecoration dec) {
	int32_t n = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == SpvOpDecorate && wc > 2 && b->words[i + 2] == (uint32_t)dec)
			n++;
		i += (int32_t)wc;
	}
	return n;
}

static bool has_extension(const svsl_spirv_blob_t *b, const char *name) {
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == SpvOpExtension &&
		    strncmp((const char *)&b->words[i + 1], name, (size_t)(wc - 1) * 4) == 0)
			return true;
		i += (int32_t)wc;
	}
	return false;
}

// a window-decorated sampler is exclusive to Window ops, so those get their own
// sampler + textures (winSmp / winTarget / winReference)
static const char *src_image_processing =
	"Texture2D      tex;\n"
	"Texture2D      target;\n"
	"Texture2D      reference;\n"
	"Texture2DArray weights;\n"
	"Texture2D      winTarget;\n"
	"Texture2D      winReference;\n"
	"SamplerState   smp;\n"
	"SamplerState   bmSmp;\n"
	"SamplerState   winSmp;\n"
	"float4 ps(float2 uv : TEXCOORD0) : SV_Target {\n"
	"	float4 w = tex.SampleWeightedQCOM(smp, uv, weights);\n"
	"	float4 b = tex.BoxFilterQCOM(smp, uv, float2(3, 3));\n"
	"	float4 m = target.BlockMatchSADQCOM(bmSmp, uint2(8, 8), reference, uint2(4, 4), uint2(8, 8));\n"
	"	float4 g = winTarget.BlockMatchWindowSSDQCOM(winSmp, uint2(8, 8), winReference, uint2(4, 4), uint2(8, 8));\n"
	"	return w + b + m + g;\n"
	"}\n";

static const char *src_tile_cs =
	"//--apron = 8, 4\n"
	"[tile_attachment] Texture2D<float4>   lastFrame;\n"
	"[tile_attachment] RWTexture2D<float4> color;\n"
	"RWStructuredBuffer<float4> results;\n"
	"[tile_shading_rate_qcom(2, 2, 1)]\n"
	"void cs(uint3 id : SV_DispatchThreadID) {\n"
	"	uint2 off = tile_offset_qcom();\n"
	"	uint3 dim = tile_dimension_qcom();\n"
	"	uint2 ap  = tile_apron_size_qcom();\n"
	"	float4 prev = lastFrame.Load(int3(id.xy, 0));\n"
	"	color[int2(id.xy)] = prev * 0.5;\n"
	"	results[id.x] = float4((float)off.x, (float)dim.z, (float)ap.y, prev.x);\n"
	"}\n";

void test_qcom(void) {
	qcom_case_t c;

	// --- image processing: caps, extensions, decorations, 1.4 bump ---------------
	qcom_compile(&c, src_image_processing);
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(c.blob_count == 1);
	const svsl_spirv_blob_t *b = &c.blobs[0];
	TEST_CHECK(b->word_count > 5 && b->words[1] == 0x00010400u); // version bump
	TEST_CHECK(count_insts(b, SpvOpCapability, SpvCapabilityTextureSampleWeightedQCOM, -1) == 1);
	TEST_CHECK(count_insts(b, SpvOpCapability, SpvCapabilityTextureBoxFilterQCOM, -1) == 1);
	TEST_CHECK(count_insts(b, SpvOpCapability, SpvCapabilityTextureBlockMatchQCOM, -1) == 1);
	TEST_CHECK(count_insts(b, SpvOpCapability, SpvCapabilityTextureBlockMatch2QCOM, -1) == 1);
	TEST_CHECK(has_extension(b, "SPV_QCOM_image_processing"));
	TEST_CHECK(has_extension(b, "SPV_QCOM_image_processing2"));
	TEST_CHECK(count_decoration(b, SpvDecorationWeightTextureQCOM) == 1);
	TEST_CHECK(count_decoration(b, SpvDecorationBlockMatchTextureQCOM) == 4); // target/reference + win pair
	TEST_CHECK(count_decoration(b, SpvDecorationBlockMatchSamplerQCOM) >= 1);
	TEST_CHECK(count_insts(b, SpvOpImageSampleWeightedQCOM, -1, -1) == 1);
	TEST_CHECK(count_insts(b, SpvOpImageBoxFilterQCOM, -1, -1) == 1);
	TEST_CHECK(count_insts(b, SpvOpImageBlockMatchSADQCOM, -1, -1) == 1);
	TEST_CHECK(count_insts(b, SpvOpImageBlockMatchWindowSSDQCOM, -1, -1) == 1);

	// per-op sks feature bits: 17 sample_weighted, 18 box_filter,
	// 19 block_match, 20 image_processing2
	{
		svsl_sks_blob_t sks = {0};
		svsl_sks_write(&c.arena, &c.program, &c.ir, c.blobs, NULL,
		               &(svsl_sks_options_t){ .targets = svsl_target_spirv }, &sks);
		// features u64 sits after tag(8) + version(2) + count(4) + name(256) +
		// buffers(4) + resources(4) + inputs(4) + spec consts(4) + samplers(4, v12)
		uint64_t features;
		memcpy(&features, sks.bytes + 8 + 2 + 4 + 256 + 4 + 4 + 4 + 4 + 4, 8); // v12: + sampler_count
		TEST_CHECK(features & (1ull << 17));
		TEST_CHECK(features & (1ull << 18));
		TEST_CHECK(features & (1ull << 19));
		TEST_CHECK(features & (1ull << 20));
		TEST_CHECK(!(features & (1ull << 63))); // known bits, not "unknown"
	}
	svsl_arena_free(&c.arena);

	// --- exclusivity: block-match texture also sampled plainly is an error --------
	qcom_compile(&c,
		"Texture2D t; Texture2D r; SamplerState s;\n"
		"float4 ps(float2 uv : TEXCOORD0) : SV_Target {\n"
		"	float4 a = t.Sample(s, uv);\n"
		"	float4 b = t.BlockMatchSADQCOM(s, uint2(0, 0), r, uint2(0, 0), uint2(4, 4));\n"
		"	return a + b;\n"
		"}\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	// --- tile shading compute: storage class, exec mode, builtins, 1.3 kept -------
	qcom_compile(&c, src_tile_cs);
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(c.blob_count == 1);
	b = &c.blobs[0];
	TEST_CHECK(b->words[1] == 0x00010300u); // tile shading stays SPIR-V 1.3
	TEST_CHECK(count_insts(b, SpvOpCapability, SpvCapabilityTileShadingQCOM, -1) == 1);
	TEST_CHECK(has_extension(b, "SPV_QCOM_tile_shading"));
	TEST_CHECK(count_insts(b, SpvOpVariable, -1, -1) > 0);
	// two tile attachment variables in the TileAttachmentQCOM storage class
	{
		int32_t tile_vars = 0;
		for (int32_t i = 5; i < b->word_count; ) {
			uint32_t wc = b->words[i] >> 16;
			if (wc == 0) break;
			if ((b->words[i] & 0xFFFF) == SpvOpVariable && wc > 3 &&
			    b->words[i + 3] == (uint32_t)SpvStorageClassTileAttachmentQCOM)
				tile_vars++;
			i += (int32_t)wc;
		}
		TEST_CHECK(tile_vars == 2);
	}
	// TileShadingRateQCOM(2, 2, 1) replaces LocalSize
	TEST_CHECK(count_insts(b, SpvOpExecutionMode, -1, SpvExecutionModeTileShadingRateQCOM) == 1);
	TEST_CHECK(count_insts(b, SpvOpExecutionMode, -1, SpvExecutionModeLocalSize) == 0);
	TEST_CHECK(count_decoration(b, SpvDecorationBuiltIn) >= 3); // tile builtins (+ thread id)
	// //--apron flows to the program
	TEST_CHECK(c.program.tile_apron[0] == 8 && c.program.tile_apron[1] == 4);
	// sks feature bit 21 (tile shading)
	{
		svsl_sks_blob_t sks = {0};
		svsl_sks_write(&c.arena, &c.program, &c.ir, c.blobs, NULL,
		               &(svsl_sks_options_t){ .targets = svsl_target_spirv }, &sks);
		uint64_t features;
		memcpy(&features, sks.bytes + 8 + 2 + 4 + 256 + 4 + 4 + 4 + 4 + 4, 8); // v12: + sampler_count
		TEST_CHECK(features & (1ull << 21));
	}
	svsl_arena_free(&c.arena);

	// --- rate + numthreads is a conflict ------------------------------------------
	qcom_compile(&c,
		"[numthreads(8, 8, 1)]\n"
		"[tile_shading_rate_qcom(2, 2, 1)]\n"
		"void cs() {}\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	// --- non-pow2 tile rate is an error -------------------------------------------
	qcom_compile(&c, "[tile_shading_rate_qcom(3, 2, 1)]\nvoid cs() {}\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	// --- [tile_attachment] on a non-2D resource is an error -----------------------
	qcom_compile(&c,
		"[tile_attachment] Texture3D<float4> vol;\n"
		"float4 ps() : SV_Target { return vol.Load(int4(0, 0, 0, 0)); }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	// --- //--apron without tile usage warns; bad apron errors ----------------------
	qcom_compile(&c, "//--apron = 2\nfloat4 ps() : SV_Target { return (float4)1; }\n");
	TEST_CHECK(c.diags.error_count == 0 && c.warnings > 0);
	svsl_arena_free(&c.arena);
	qcom_compile(&c, "//--apron = fish\nfloat4 ps() : SV_Target { return (float4)1; }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	printf("  qcom: image processing + tile shading word checks\n");
}

// Storage image formats and per-target compilation.
//
// An undeclared storage image format means SpvImageFormatUnknown — the shader
// is agnostic to the bound view's format, like DXC. The old texel-type
// inference survives only inside the WGSL backend, which has no formatless
// storage texture to emit (see test_wgsl.c). The emitted words are the testable
// surface here: the OpTypeImage format operand, the per-usage
// StorageImage*WithoutFormat capabilities, and the sks feature bits they map to.

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

#include <svsl/svsl.h>

#include <string.h>

typedef struct fmt_case_t {
	svsl_arena_t      arena;
	svsl_program_t    program;
	svsl_ir_module_t  ir;
	svsl_spirv_blob_t blobs[4];
	int32_t           blob_count;
	svsl_diag_list_t  diags;
} fmt_case_t;

static void fmt_compile(fmt_case_t *c, const char *src) {
	memset(c, 0, sizeof(*c));
	svsl_pp_result_t pp;
	svsl_pp_run(&c->arena, src, "test_formats.hlsl", NULL, &pp, &c->diags);
	svsl_token_list_t tokens = {0};
	svsl_lex(&c->arena, &pp, &tokens, &c->diags);
	svsl_ast_t *ast = svsl_parse(&c->arena, &tokens, &c->diags);
	svsl_sema_run(&c->arena, ast, &pp, "test_formats.hlsl", NULL, &c->program, &c->diags);
	if (c->diags.error_count == 0)
		svsl_ir_build(&c->arena, &c->program, svsl_opt_default, &c->ir, &c->diags);
	if (c->diags.error_count == 0)
		for (int32_t i = 0; i < c->ir.func_count && i < 4; i++)
			if (svsl_spirv_emit(&c->arena, &c->program, &c->ir.funcs[i], &c->blobs[c->blob_count], &c->diags))
				c->blob_count++;
}

static int32_t count_caps(const svsl_spirv_blob_t *b, SpvCapability cap) {
	int32_t n = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == (uint32_t)SpvOpCapability && wc > 1 &&
		    b->words[i + 1] == (uint32_t)cap)
			n++;
		i += (int32_t)wc;
	}
	return n;
}

static int32_t count_insts(const svsl_spirv_blob_t *b, SpvOp op) {
	int32_t n = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == (uint32_t)op) n++;
		i += (int32_t)wc;
	}
	return n;
}

// True when the module declares a 32-bit OpConstant of this value — enough to
// see which memory-semantics mask an atomic was handed, since the tiny shaders
// here have no other source for the bit pattern.
static bool has_u32_const(const svsl_spirv_blob_t *b, uint32_t value) {
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == (uint32_t)SpvOpConstant && wc == 4 && b->words[i + 3] == value)
			return true;
		i += (int32_t)wc;
	}
	return false;
}

// OpTypeImage: result, sampled type, dim, depth, arrayed, MS, sampled, format
static int64_t first_image_format(const svsl_spirv_blob_t *b) {
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == (uint32_t)SpvOpTypeImage && wc >= 9)
			return (int64_t)b->words[i + 8];
		i += (int32_t)wc;
	}
	return -1;
}

static uint64_t sks_features(fmt_case_t *c) {
	svsl_sks_blob_t sks = {0};
	svsl_sks_write(&c->arena, &c->program, &c->ir, c->blobs, NULL,
	               &(svsl_sks_options_t){ .targets = svsl_target_spirv }, &sks);
	// tag(8) + version(2) + count(4) + name(256) + buffers(4) + resources(4)
	// + inputs(4) + spec consts(4) + samplers(4, v12)
	uint64_t features = 0;
	memcpy(&features, sks.bytes + 8 + 2 + 4 + 256 + 4 + 4 + 4 + 4 + 4, 8);
	return features;
}

#define SKS_FEAT_FORMATLESS 13 // joint read/write, see sks_write.c

static const char *k_write_only =
	"RWTexture2DArray<float4> dst : register(u0);\n"
	"[numthreads(8, 8, 1)]\n"
	"void cs(uint3 id : SV_DispatchThreadID) { dst[id] = float4(1, 0, 0, 1); }\n";

void test_formats(void) {
	fmt_case_t c;

	// --- undeclared format: Unknown, and only the capability actually used ------
	fmt_compile(&c, k_write_only);
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(c.blob_count == 1);
	TEST_CHECK(first_image_format(&c.blobs[0]) == SpvImageFormatUnknown);
	TEST_CHECK(count_caps(&c.blobs[0], SpvCapabilityStorageImageWriteWithoutFormat) == 1);
	TEST_CHECK(count_caps(&c.blobs[0], SpvCapabilityStorageImageReadWithoutFormat)  == 0);
	// the sks bit is joint, so a write-only shader still reports it — the SPIR-V
	// capability list above is what distinguishes read from write
	{
		uint64_t f = sks_features(&c);
		TEST_CHECK(  f & (1ull << SKS_FEAT_FORMATLESS));
		TEST_CHECK(!(f & (1ull << 63))); // known bits, not "unknown capability"
	}
	svsl_arena_free(&c.arena);

	// --- read and write: both capabilities, both bits ---------------------------
	fmt_compile(&c,
		"RWTexture2D<float4> dst : register(u0);\n"
		"RWTexture2D<float4> src : register(u1);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { dst[id.xy] = src[id.xy] * 2; }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(first_image_format(&c.blobs[0]) == SpvImageFormatUnknown);
	TEST_CHECK(count_caps(&c.blobs[0], SpvCapabilityStorageImageWriteWithoutFormat) == 1);
	TEST_CHECK(count_caps(&c.blobs[0], SpvCapabilityStorageImageReadWithoutFormat)  == 1);
	TEST_CHECK(sks_features(&c) & (1ull << SKS_FEAT_FORMATLESS));
	svsl_arena_free(&c.arena);

	// --- an explicit format still wins, and stays off the capabilities -----------
	fmt_compile(&c,
		"[[vk::image_format(\"rgba8\")]] RWTexture2DArray<float4> dst : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { dst[id] = float4(1, 0, 0, 1); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(first_image_format(&c.blobs[0]) == SpvImageFormatRgba8);
	TEST_CHECK(count_caps(&c.blobs[0], SpvCapabilityStorageImageWriteWithoutFormat) == 0);
	TEST_CHECK(!(sks_features(&c) & (1ull << SKS_FEAT_FORMATLESS)));
	svsl_arena_free(&c.arena);

	// --- 'unknown' is spellable both ways, and matches the undeclared form -------
	fmt_compile(&c,
		"Image2DArray<float4, unknown> dst : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { dst[id] = float4(1, 0, 0, 1); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(first_image_format(&c.blobs[0]) == SpvImageFormatUnknown);
	TEST_CHECK(count_caps(&c.blobs[0], SpvCapabilityStorageImageWriteWithoutFormat) == 1);
	svsl_arena_free(&c.arena);

	fmt_compile(&c,
		"[[vk::image_format(\"unknown\")]] RWTexture2DArray<float4> dst : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { dst[id] = float4(1, 0, 0, 1); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(first_image_format(&c.blobs[0]) == SpvImageFormatUnknown);
	svsl_arena_free(&c.arena);

	// a name that is neither a format nor 'unknown' is still an error
	fmt_compile(&c,
		"Image2D<float4, rgba9> dst : register(u0);\n"
		"[numthreads(1, 1, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { dst[id.xy] = 1; }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	// --- image atomics need a real format ----------------------------------------
	// OpImageTexelPointer's image format must not be Unknown under Vulkan's
	// standalone SPIR-V rules, so this has to fail in sema rather than emit
	fmt_compile(&c,
		"Image2D<uint> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { counter.InterlockedAdd(id.xy, 1u); }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	fmt_compile(&c,
		"Image2D<uint, unknown> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { counter.InterlockedAdd(id.xy, 1u); }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	fmt_compile(&c,
		"Image2D<uint, r32ui> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { counter.InterlockedAdd(id.xy, 1u); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(first_image_format(&c.blobs[0]) == SpvImageFormatR32ui);
	svsl_arena_free(&c.arena);

	// a declared format that isn't r32f/r32i/r32ui is rejected too: Vulkan's
	// OpImageTexelPointer rule is narrower than "not Unknown", and emitting one
	// anyway fails spirv-val (VUID-StandaloneSpirv-OpImageTexelPointer-04658)
	fmt_compile(&c,
		"Image2D<uint, r8ui> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { counter.InterlockedAdd(id.xy, 1u); }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	fmt_compile(&c,
		"Image2D<uint, r16ui> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { InterlockedAdd(counter[id.xy], 1u); }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	fmt_compile(&c,
		"Image2D<float, r32f> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { counter.InterlockedExchange(id.xy, 1.0f); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	svsl_arena_free(&c.arena);

	// --- HLSL's subscript spelling takes the same image-atomic path ---------------
	// InterlockedAdd(img[coord], v) has no pointer to lower through; it used to
	// reach lower_expr with a null value and crash
	fmt_compile(&c,
		"Image2D<uint, r32ui> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { InterlockedAdd(counter[id.xy], 1u); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(c.blob_count == 1);
	TEST_CHECK(count_insts(&c.blobs[0], SpvOpImageTexelPointer) == 1);
	TEST_CHECK(count_insts(&c.blobs[0], SpvOpAtomicIAdd)        == 1);
	svsl_arena_free(&c.arena);

	// the out-argument form, and the agnostic-format rejection, reach it too
	fmt_compile(&c,
		"Image2D<uint, r32ui> counter : register(u0);\n"
		"RWStructuredBuffer<uint> result : register(u1);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint prior; InterlockedExchange(counter[id.xy], 42u, prior); result[0] = prior; }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(count_insts(&c.blobs[0], SpvOpAtomicExchange) == 1);
	svsl_arena_free(&c.arena);

	fmt_compile(&c,
		"Image2D<uint> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { InterlockedAdd(counter[id.xy], 1u); }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	// the third argument means different things per spelling: the HLSL alias
	// takes an out-param, the native form a memory-order name. 0x808 is
	// AcquireRelease | ImageMemory — the semantics mask an ordered image atomic
	// gets, and proof the order reached emit instead of being read as a store
	// destination
	fmt_compile(&c,
		"Image2D<uint, r32ui> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { atomic_add(counter[id.xy], 1u, acq_rel); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(c.blob_count == 1);
	TEST_CHECK(count_insts(&c.blobs[0], SpvOpAtomicIAdd) == 1);
	TEST_CHECK(has_u32_const(&c.blobs[0], SpvMemorySemanticsAcquireReleaseMask |
	                                      SpvMemorySemanticsImageMemoryMask));
	svsl_arena_free(&c.arena);

	// relaxed stays None, bit-identical to the unordered spelling
	fmt_compile(&c,
		"Image2D<uint, r32ui> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) { atomic_add(counter[id.xy], 1u); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(!has_u32_const(&c.blobs[0], SpvMemorySemanticsAcquireReleaseMask |
	                                       SpvMemorySemanticsImageMemoryMask));
	svsl_arena_free(&c.arena);

	// an image atomic carries (image, coord, value) with no comparator slot
	fmt_compile(&c,
		"Image2D<uint, r32ui> counter : register(u0);\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	uint p; InterlockedCompareExchange(counter[id.xy], 0u, 1u, p); }\n");
	TEST_CHECK(c.diags.error_count > 0);
	svsl_arena_free(&c.arena);

	// buffer and groupshared destinations still go through the pointer path
	fmt_compile(&c,
		"RWStructuredBuffer<uint> buf : register(u0);\n"
		"groupshared uint tile[4];\n"
		"[numthreads(8, 8, 1)]\n"
		"void cs(uint3 id : SV_DispatchThreadID) {\n"
		"	InterlockedAdd(buf[0], 1u); InterlockedAdd(tile[0], 1u); }\n");
	TEST_CHECK(c.diags.error_count == 0);
	TEST_CHECK(count_insts(&c.blobs[0], SpvOpImageTexelPointer) == 0);
	TEST_CHECK(count_insts(&c.blobs[0], SpvOpAtomicIAdd)        == 2);
	svsl_arena_free(&c.arena);

	// --- one target language per container ---------------------------------------
	{
		svsl_source_t src = { .text = k_write_only, .filename = "targets.hlsl" };
		svsl_result_t *r  = svsl_compile(&src, &(svsl_options_t){
			.targets = svsl_target_spirv | svsl_target_wgsl });
		TEST_CHECK(r != NULL);
		TEST_CHECK(!r->ok); // both bits: nothing to preprocess against
		svsl_result_free(r);

		r = svsl_compile(&src, &(svsl_options_t){ .targets = svsl_target_spirv });
		TEST_CHECK(r->ok);
		svsl_result_free(r);
	}

	// --- TARGET_* predefines ------------------------------------------------------
	// exactly one is defined, and the caller's -D still wins: predefines are
	// seeded ahead of them and macro lookup takes the last definition
	{
		static const char *k_probe =
			"RWTexture2D<float4> dst : register(u0);\n"
			"[numthreads(1, 1, 1)]\n"
			"void cs(uint3 id : SV_DispatchThreadID) {\n"
			"#if defined(TARGET_SPIRV) && !defined(TARGET_WGSL)\n"
			"	dst[id.xy] = float4(1, 0, 0, 1);\n"
			"#else\n"
			"	#error \"expected TARGET_SPIRV only\"\n"
			"#endif\n"
			"}\n";
		svsl_source_t  src = { .text = k_probe, .filename = "predef.hlsl" };
		svsl_result_t *r   = svsl_compile(&src, NULL); // default target is SPIR-V
		TEST_CHECK(r->ok);
		svsl_result_free(r);

		static const char *k_override =
			"RWTexture2D<float4> dst : register(u0);\n"
			"[numthreads(1, 1, 1)]\n"
			"void cs(uint3 id : SV_DispatchThreadID) {\n"
			"#if TARGET_SPIRV == 7\n"
			"	dst[id.xy] = float4(1, 0, 0, 1);\n"
			"#else\n"
			"	#error \"caller -D should override the predefine\"\n"
			"#endif\n"
			"}\n";
		src = (svsl_source_t){ .text = k_override, .filename = "override.hlsl" };
		r   = svsl_compile(&src, &(svsl_options_t){
			.defines      = (svsl_define_t[]){ { "TARGET_SPIRV", "7" } },
			.define_count = 1 });
		TEST_CHECK(r->ok);
		svsl_result_free(r);
	}
}

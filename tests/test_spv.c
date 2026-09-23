// the SPIR-V headers' has-result/has-type table (C99 inline: the plain
// declaration below makes this translation unit hold its external definition)
#define SPV_ENABLE_UTILITY_CODE
#include "test_spv.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls" // intentional: see above
void SpvHasResultAndType(SpvOp opcode, bool *hasResult, bool *hasResultType);
#pragma GCC diagnostic pop

#include "front/lexer.h"
#include "front/parser.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void test_spv_compile(svsl_arena_t *arena, const char *src, const char *opt_path,
                      const svsl_pp_options_t *opt_pp, svsl_opt_level_ level, test_spv_t *out) {
	memset(out, 0, sizeof(*out));
	const char       *path = opt_path ? opt_path : "test_spv.hlsl";
	svsl_pp_result_t  pp;
	svsl_token_list_t tokens = {0};
	svsl_pp_run(arena, src, path, opt_pp, &pp, &out->diags);
	svsl_lex(arena, &pp, &tokens, &out->diags);
	svsl_ast_t *ast = svsl_parse(arena, &tokens, &out->diags);
	svsl_sema_run(arena, ast, &pp, path, NULL, &out->prog, &out->diags);
	if (out->diags.error_count == 0)
		svsl_ir_build(arena, &out->prog, level, &out->ir, &out->diags);
	if (out->diags.error_count == 0)
		for (int32_t i = 0; i < out->ir.func_count && i < 4; i++)
			if (svsl_spirv_emit(arena, &out->prog, &out->ir.funcs[i], &out->blobs[out->blob_count], &out->diags))
				out->blob_count++;
	for (int32_t i = 0; i < out->diags.count; i++)
		if (out->diags.items[i].severity == svsl_severity_warning) out->warnings++;
	out->ok = out->diags.error_count == 0 && out->blob_count == out->ir.func_count;
}

int32_t test_spv_count(const svsl_spirv_blob_t *b, SpvOp op, int64_t op_a, int64_t op_b) {
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

int32_t test_spv_find_def(const svsl_spirv_blob_t *b, uint32_t id) {
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16, op = b->words[i] & 0xFFFF;
		if (wc == 0) break;
		bool has_result, has_type;
		SpvHasResultAndType((SpvOp)op, &has_result, &has_type);
		int32_t at = has_type ? 2 : 1; // the result id follows the result type, when there is one
		if (has_result && (int32_t)wc > at && b->words[i + at] == id) return i;
		i += (int32_t)wc;
	}
	return -1;
}

bool test_spv_validate(const test_spv_t *spv) {
	static int32_t has_spirv_val = -1;
	if (has_spirv_val < 0)
		has_spirv_val = system("spirv-val --version > /dev/null 2>&1") == 0 ? 1 : 0;
	if (!has_spirv_val) return true;
	bool ok = true;
	for (int32_t i = 0; i < spv->blob_count; i++) {
		const svsl_spirv_blob_t *b = &spv->blobs[i];
		FILE *f = fopen("svsl_spv_test_tmp.spv", "wb");
		if (!f) return false;
		fwrite(b->words, 4, (size_t)b->word_count, f);
		fclose(f);
		bool v14 = b->word_count > 1 && b->words[1] >= 0x00010400u;
		char cmd[160];
		snprintf(cmd, sizeof(cmd), "spirv-val --target-env %s%s svsl_spv_test_tmp.spv",
		         v14 ? "vulkan1.1spv1.4" : "vulkan1.1", spv->prog.needs_scalar_layout ? " --scalar-block-layout" : "");
		if (system(cmd) != 0) ok = false;
	}
	remove("svsl_spv_test_tmp.spv");
	return ok;
}

int32_t test_spv_loop_exits(const svsl_spirv_blob_t *b, int32_t *opt_true_to_merge) {
	uint32_t merges[256];
	int32_t  merge_count = 0, exits = 0, true_to_merge = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == SpvOpLoopMerge && merge_count < 256) merges[merge_count++] = b->words[i + 1];
		i += (int32_t)wc;
	}
	uint32_t prev = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16, op = b->words[i] & 0xFFFF;
		if (wc == 0) break;
		if (op == SpvOpBranchConditional && prev != SpvOpSelectionMerge && prev != SpvOpLoopMerge) {
			for (int32_t m = 0; m < merge_count; m++) {
				if (b->words[i + 2] != merges[m] && b->words[i + 3] != merges[m]) continue;
				exits++;
				if (b->words[i + 2] == merges[m]) true_to_merge++;
				break;
			}
		}
		prev = op;
		i += (int32_t)wc;
	}
	if (opt_true_to_merge) *opt_true_to_merge = true_to_merge;
	return exits;
}

int32_t test_spv_array_loads(const svsl_spirv_blob_t *b) {
	int32_t n = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == SpvOpLoad) {
			int32_t def = test_spv_find_def(b, b->words[i + 1]); // the result type
			if (def >= 0 && (b->words[def] & 0xFFFF) == SpvOpTypeArray) n++;
		}
		i += (int32_t)wc;
	}
	return n;
}

int32_t test_spv_const_bitcasts(const svsl_spirv_blob_t *b) {
	int32_t n = 0;
	for (int32_t i = 5; i < b->word_count; ) {
		uint32_t wc = b->words[i] >> 16;
		if (wc == 0) break;
		if ((b->words[i] & 0xFFFF) == SpvOpBitcast) {
			int32_t def = test_spv_find_def(b, b->words[i + 3]); // the operand
			if (def >= 0 && (b->words[def] & 0xFFFF) == SpvOpConstant) n++;
		}
		i += (int32_t)wc;
	}
	return n;
}

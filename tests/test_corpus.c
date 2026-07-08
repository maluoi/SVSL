// Corpus tests: every StereoKit + ported prototype shader in tests/shaders/
// must go through the whole pipeline cleanly — pp + lex + parse + sema + IR +
// SPIR-V emission — and, when spirv-val is on PATH, every stage must validate
// against the Vulkan 1.1 environment.

#include "test.h"

#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "back/emit_spirv.h"
#include "ir/ir.h"
#include "sema/sema.h"
#include "util/arena.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SVSL_TEST_DIR
#define SVSL_TEST_DIR "."
#endif

static char *read_file(svsl_arena_t *arena, const char *path, int32_t *out_len) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *data = svsl_arena_alloc(arena, (size_t)size + 1);
	size_t read = fread(data, 1, (size_t)size, f);
	fclose(f);
	data[read] = '\0';
	if (out_len) *out_len = (int32_t)read;
	return data;
}

// Resolves includes against the requesting file's directory, then shaders/include/.
static svsl_include_src_t corpus_include(void *user, const char *path, const char *requester) {
	svsl_arena_t *arena = user;
	char          full[1024];

	const char *slash = requester ? strrchr(requester, '/') : NULL;
	if (slash) {
		snprintf(full, sizeof(full), "%.*s/%s", (int)(slash - requester), requester, path);
		int32_t len     = 0;
		char   *content = read_file(arena, full, &len);
		if (content) {
			return (svsl_include_src_t){
				.content = content,
				.length  = len,
				.path    = svsl_arena_strndup(arena, full, strlen(full)) };
		}
	}
	snprintf(full, sizeof(full), "%s/shaders/include/%s", SVSL_TEST_DIR, path);
	int32_t len     = 0;
	char   *content = read_file(arena, full, &len);
	if (!content) return (svsl_include_src_t){0};
	return (svsl_include_src_t){
		.content = content,
		.length  = len,
		.path    = svsl_arena_strndup(arena, full, strlen(full)) };
}

static bool corpus_check_file(const char *path) {
	svsl_arena_t arena = {0};
	char        *src   = read_file(&arena, path, NULL);
	if (!src) {
		printf("  cannot read %s\n", path);
		svsl_arena_free(&arena);
		return false;
	}

	svsl_pp_options_t opt = { .include_cb = corpus_include, .include_user = &arena };
	svsl_pp_result_t  pp;
	svsl_diag_list_t  diags = {0};
	bool ok = svsl_pp_run(&arena, src, path, &opt, &pp, &diags);

	svsl_token_list_t tokens = {0};
	ok = svsl_lex(&arena, &pp, &tokens, &diags) && ok;

	svsl_ast_t *ast = svsl_parse(&arena, &tokens, &diags);

	svsl_program_t program;
	svsl_sema_run(&arena, ast, &pp, path, NULL, &program, &diags);

	svsl_ir_module_t ir = {0};
	if (diags.error_count == 0)
		svsl_ir_build(&arena, &program, svsl_opt_default, &ir, &diags);

	if (diags.error_count == 0) {
		static int32_t has_spirv_val = -1;
		if (has_spirv_val < 0)
			has_spirv_val = system("spirv-val --version > /dev/null 2>&1") == 0 ? 1 : 0;
		for (int32_t i = 0; i < ir.func_count; i++) {
			svsl_spirv_blob_t blob = {0};
			if (!svsl_spirv_emit(&arena, &program, &ir.funcs[i], &blob, &diags)) break;
			if (!has_spirv_val) continue;
			FILE *f = fopen("svsl_corpus_tmp.spv", "wb");
			if (!f) continue;
			fwrite(blob.words, 4, (size_t)blob.word_count, f);
			fclose(f);
			if (system("spirv-val --target-env vulkan1.1 svsl_corpus_tmp.spv") != 0) {
				printf("  spirv-val failed: %s (entry %d)\n", path, i);
				svsl_diag_add(&arena, &diags, svsl_severity_error,
				              (svsl_loc_t){ .file = path }, "SPIR-V validation failed");
			}
		}
		remove("svsl_corpus_tmp.spv");
	}
	ok = diags.error_count == 0;

	if (!ok) {
		printf("  %s:\n", path);
		for (int32_t i = 0; i < diags.count; i++) {
			if (diags.items[i].severity != svsl_severity_error) continue;
			printf("    %s:%d: %s\n", diags.items[i].loc.file, diags.items[i].loc.line, diags.items[i].message);
		}
	}
	svsl_arena_free(&arena);
	return ok;
}

static int compare_names(const void *a, const void *b) {
	return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void corpus_check_dir(const char *subdir, int32_t *ref_count) {
	char dir_path[1024];
	snprintf(dir_path, sizeof(dir_path), "%s/shaders/%s", SVSL_TEST_DIR, subdir);

	DIR *dir = opendir(dir_path);
	TEST_CHECK(dir != NULL);
	if (!dir) return;

	// collect + sort for deterministic order
	char  *names[256];
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
	qsort(names, (size_t)name_count, sizeof(names[0]), compare_names);

	for (int32_t i = 0; i < name_count; i++) {
		char path[2048];
		snprintf(path, sizeof(path), "%s/%s", dir_path, names[i]);
		TEST_CHECK(corpus_check_file(path));
		free(names[i]);
		(*ref_count)++;
	}
}

static void test_corpus_metadata(void) {
	// spot-check that a known shader's metadata comes through
	char path[1024];
	snprintf(path, sizeof(path), "%s/shaders/builtin/shader_builtin_unlit.hlsl", SVSL_TEST_DIR);

	svsl_arena_t arena = {0};
	char        *src   = read_file(&arena, path, NULL);
	TEST_CHECK(src != NULL);
	if (!src) { svsl_arena_free(&arena); return; }

	svsl_pp_options_t opt = { .include_cb = corpus_include, .include_user = &arena };
	svsl_pp_result_t  pp;
	svsl_diag_list_t  diags = {0};
	TEST_CHECK(svsl_pp_run(&arena, src, path, &opt, &pp, &diags));

	bool found_name = false;
	for (int32_t i = 0; i < pp.meta_count; i++) {
		if (svsl_str_eq_cstr(pp.metas[i].name, "name") && svsl_str_eq_cstr(pp.metas[i].value, "sk/unlit"))
			found_name = true;
	}
	TEST_CHECK(found_name);
	svsl_arena_free(&arena);
}

void test_corpus(void) {
	int32_t count = 0;
	corpus_check_dir("builtin",   &count);
	corpus_check_dir("examples",  &count);
	corpus_check_dir("ported",    &count);
	corpus_check_dir("morrowind", &count);
	corpus_check_dir("checks",    &count);
	printf("  corpus: %d shaders\n", count);
	TEST_CHECK(count >= 121); // 17 builtin + 11 examples + 72 ported + 19 morrowind + checks

	test_corpus_metadata();
}

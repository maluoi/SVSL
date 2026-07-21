// svsl_bench — compiler performance harness. Compiles the whole shader corpus
// many times, timing each pipeline phase, and reports the payoff metrics
// (live IR instructions and emitted SPIR-V words) so the cost *and* benefit of
// each optimization pass can be tracked. Not a correctness test — the corpus
// and IR golden tests cover that. See docs/OPTIMIZATION_PLAN.md §5.
//
//   ./svsl_bench            full report at -O0 and -O1 (and -O2)
//   ./svsl_bench -O1 -n 400  pin a level and iteration count

#define _POSIX_C_SOURCE 199309L // clock_gettime / CLOCK_MONOTONIC under -std=c11

#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "back/emit_spirv.h"
#ifdef SVSL_HAS_WGSL
#include "back/emit_wgsl.h"
#endif
#include "ir/ir.h"
#include "sema/sema.h"
#include "util/arena.h"

#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef SVSL_TEST_DIR
#define SVSL_TEST_DIR "."
#endif

// --- corpus loading ---------------------------------------------------------

typedef struct { char *path; char *src; } shader_t;

static char *read_file(svsl_arena_t *arena, const char *path) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *data = svsl_arena_alloc(arena, (size_t)size + 1);
	size_t n = fread(data, 1, (size_t)size, f);
	fclose(f);
	data[n] = '\0';
	return data;
}

static svsl_include_src_t bench_include(void *user, const char *path, const char *requester) {
	svsl_arena_t *arena = user;
	char          full[1024];
	const char   *slash = requester ? strrchr(requester, '/') : NULL;
	if (slash) {
		snprintf(full, sizeof(full), "%.*s/%s", (int)(slash - requester), requester, path);
		char *c = read_file(arena, full);
		if (c) return (svsl_include_src_t){ .content = c, .length = (int32_t)strlen(c),
		                                    .path = svsl_arena_strndup(arena, full, strlen(full)) };
	}
	snprintf(full, sizeof(full), "%s/shaders/include/%s", SVSL_TEST_DIR, path);
	char *c = read_file(arena, full);
	if (!c) return (svsl_include_src_t){0};
	return (svsl_include_src_t){ .content = c, .length = (int32_t)strlen(c),
	                            .path = svsl_arena_strndup(arena, full, strlen(full)) };
}

static int32_t collect(svsl_arena_t *arena, shader_t *out, int32_t cap) {
	static const char *dirs[] = { "builtin", "examples", "ported", "morrowind", "checks" };
	int32_t n = 0;
	for (size_t d = 0; d < sizeof(dirs) / sizeof(dirs[0]); d++) {
		char dir_path[1024];
		snprintf(dir_path, sizeof(dir_path), "%s/shaders/%s", SVSL_TEST_DIR, dirs[d]);
		DIR *dir = opendir(dir_path);
		if (!dir) continue;
		struct dirent *e;
		while ((e = readdir(dir)) != NULL && n < cap) {
			const char *ext = strrchr(e->d_name, '.');
			if (!ext || (strcmp(ext, ".hlsl") != 0 && strcmp(ext, ".svsl") != 0)) continue;
			char path[2048];
			snprintf(path, sizeof(path), "%s/%s", dir_path, e->d_name);
			char *src = read_file(arena, path);
			if (!src) continue;
			out[n].path = svsl_arena_strndup(arena, path, strlen(path));
			out[n].src  = src;
			n++;
		}
		closedir(dir);
	}
	return n;
}

// --- timing -----------------------------------------------------------------

static double now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

typedef struct {
	double front_ms, ir_ms, emit_ms; // best-of phase times over all shaders, one iteration
	double wgsl_ms;                  // WGSL text emission (0 unless built with SVSL_ENABLE_WGSL)
	int64_t live_insts, spirv_words; // payoff metrics (deterministic, measured once)
	int64_t wgsl_bytes;
} run_t;

// Compile every shader once, accumulating per-phase time and output metrics.
static run_t compile_corpus(const shader_t *shaders, int32_t count, svsl_opt_level_ level,
                            bool measure_metrics) {
	run_t r = {0};
	for (int32_t s = 0; s < count; s++) {
		svsl_arena_t     arena = {0};
		svsl_diag_list_t diags = {0};

		double t0 = now_ms();
		svsl_pp_options_t popt = { .include_cb = bench_include, .include_user = &arena };
		svsl_pp_result_t  pp;
		svsl_pp_run(&arena, shaders[s].src, shaders[s].path, &popt, &pp, &diags);
		svsl_token_list_t tokens = {0};
		svsl_lex(&arena, &pp, &tokens, &diags);
		svsl_ast_t *ast = svsl_parse(&arena, &tokens, &diags);
		svsl_program_t program;
		svsl_sema_run(&arena, ast, &pp, shaders[s].path, NULL, &program, &diags);
		double t1 = now_ms();

		svsl_ir_module_t ir = {0};
		if (diags.error_count == 0)
			svsl_ir_build(&arena, &program, level, &ir, &diags);
		double t2 = now_ms();

		svsl_spirv_blob_t blobs[16] = {0};
		if (diags.error_count == 0) {
			for (int32_t i = 0; i < ir.func_count && i < 16; i++) {
				svsl_spirv_emit(&arena, &program, &ir.funcs[i], &blobs[i], &diags);
				if (measure_metrics) r.spirv_words += blobs[i].word_count;
			}
		}
		double t3 = now_ms();

#ifdef SVSL_HAS_WGSL
		if (diags.error_count == 0) {
			for (int32_t i = 0; i < ir.func_count && i < 16; i++) {
				svsl_wgsl_blob_t blob = {0};
				svsl_wgsl_emit(&arena, &program, &ir.funcs[i], blobs[i].vs_input_locations,
				               &blob, &diags);
				if (measure_metrics && blob.text) r.wgsl_bytes += blob.length;
			}
		}
#endif
		double t4 = now_ms();

		r.front_ms += t1 - t0;
		r.ir_ms    += t2 - t1;
		r.emit_ms  += t3 - t2;
		r.wgsl_ms  += t4 - t3;

		if (measure_metrics && diags.error_count == 0) {
			for (int32_t i = 0; i < ir.func_count; i++) {
				const svsl_ir_func_t *fn = &ir.funcs[i];
				for (int32_t k = 0; k < fn->insts.count; k++)
					if (fn->insts.items[k].op != svsl_ir_nop) r.live_insts++;
			}
		}
		svsl_arena_free(&arena);
	}
	return r;
}

static run_t best_of(const shader_t *shaders, int32_t count, svsl_opt_level_ level, int32_t iters) {
	run_t best = compile_corpus(shaders, count, level, true); // first run also grabs metrics
	for (int32_t it = 1; it < iters; it++) {
		run_t r = compile_corpus(shaders, count, level, false);
		if (r.front_ms < best.front_ms) best.front_ms = r.front_ms;
		if (r.ir_ms    < best.ir_ms)    best.ir_ms    = r.ir_ms;
		if (r.emit_ms  < best.emit_ms)  best.emit_ms  = r.emit_ms;
		if (r.wgsl_ms  < best.wgsl_ms)  best.wgsl_ms  = r.wgsl_ms;
	}
	return best;
}

static void report(const char *label, run_t r) {
	double total = r.front_ms + r.ir_ms + r.emit_ms + r.wgsl_ms;
	printf("  %-4s | front %6.2f  ir %6.2f  emit %6.2f  wgsl %6.2f  total %6.2f ms | "
	       "live_insts %6lld  spirv_words %7lld  wgsl_bytes %7lld\n",
	       label, r.front_ms, r.ir_ms, r.emit_ms, r.wgsl_ms, total,
	       (long long)r.live_insts, (long long)r.spirv_words, (long long)r.wgsl_bytes);
}

int main(int argc, char **argv) {
	int32_t         iters = 200;
	svsl_opt_level_ only  = (svsl_opt_level_)-1; // -1 = all levels
	for (int i = 1; i < argc; i++) {
		if      (strcmp(argv[i], "-O0") == 0) only = svsl_opt_none;
		else if (strcmp(argv[i], "-O1") == 0) only = svsl_opt_default;
		else if (strcmp(argv[i], "-O2") == 0) only = svsl_opt_aggressive;
		else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
	}

	svsl_arena_t arena = {0};
	shader_t     shaders[512];
	int32_t      count = collect(&arena, shaders, 512);
	printf("svsl_bench: %d shaders, best-of-%d\n", count, iters);

	if (only == (svsl_opt_level_)-1) {
		report("-O0", best_of(shaders, count, svsl_opt_none,       iters));
		report("-O1", best_of(shaders, count, svsl_opt_default,    iters));
		report("-O2", best_of(shaders, count, svsl_opt_aggressive, iters));
	} else {
		const char *l = only == svsl_opt_none ? "-O0" : only == svsl_opt_default ? "-O1" : "-O2";
		report(l, best_of(shaders, count, only, iters));
	}
	svsl_arena_free(&arena);
	return 0;
}

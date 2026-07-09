// svsl_fuzz — dependency-free mutation fuzzer for the compiler front end.
//
// Seeds from the shader corpus, mutates deterministically (xorshift PRNG), and
// runs pp → lex → parse → sema → IR → SPIR-V emission in-process. Diagnostics
// are expected on garbage input; crashes, hangs, and sanitizer reports are the
// findings. Build with ASAN/UBSAN for real coverage.
//
//   svsl_fuzz [-seed N] [-runs N] [-dump ITER] [-file path]
//
// Every case is reproducible: `-seed N -dump ITER` regenerates the exact input
// of iteration ITER (mutation only, no compiling) and writes fuzz_case.hlsl.
// `-file path` compiles one saved case directly.

#include "front/lexer.h"
#include "front/parser.h"
#include "front/pp.h"
#include "back/emit_spirv.h"
#include "ir/ir.h"
#include "sema/sema.h"
#include "util/arena.h"

#include <dirent.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef SVSL_TEST_DIR
#define SVSL_TEST_DIR "."
#endif

#define MAX_SEEDS 256
#define MAX_INPUT (256 * 1024)

typedef struct seed_t {
	char   *data;
	int32_t len;
} seed_t;

static seed_t  seeds[MAX_SEEDS];
static int32_t seed_count = 0;
static char   *include_src;     // stereokit.hlsli, served for every #include
static int32_t include_len = 0;

static uint64_t rng_state = 1;
static uint64_t rnd(void) { // xorshift64*
	rng_state ^= rng_state >> 12;
	rng_state ^= rng_state << 25;
	rng_state ^= rng_state >> 27;
	return rng_state * 0x2545F4914F6CDD1DULL;
}
static uint32_t rnd_below(uint32_t n) { return n ? (uint32_t)(rnd() % n) : 0; }

// tokens that steer mutation toward interesting compiler paths
static const char *dictionary[] = {
	"float4", "float4x4", "half3", "float16_t", "uint64_t", "cbuffer", "struct",
	"[numthreads(8,8,1)]", "[[vk::binding(1, 2)]]", "[[vk::push_constant]]",
	"[[vk::image_format(\"rgba8\")]]", "[unroll]", "register(t0, space1)",
	": SV_Position", ": SV_Target", ": SV_DispatchThreadID",
	"#define A(x) A(x) A(x)", "#define B B B", "#if 1", "#elif defined(A)",
	"#endif", "#include \"stereokit.hlsli\"", "#pragma pack_matrix(row_major)",
	"0x7fffffff", "4294967296", "1e38f", "1e999", "0.5h", "'", "\"", "\\",
	"((((((((((((((((((((((((((((((((", "))))))))))))))))",
	"{{{{{{{{{{{{{{{{", "[[[[[[[[[[[[[[[[", "a.xxxx.yyyy.zzzz.wwww",
	"f(f(f(f(f(f(f(f(f(f(1))))))))))", "?:", "-2147483648", "u--", "++",
	"tex.Sample(tex_s,", "GetDimensions(", "mul(", "discard;", "return",
	"specialization", "pushconstant", "storagebuffer", "groupshared",
};

// --- mutation ------------------------------------------------------------------

static int32_t mutate(char *buf, int32_t len) {
	uint32_t kind = rnd_below(10);
	switch (kind) {
	case 0: { // flip a bit
		if (len == 0) return len;
		buf[rnd_below((uint32_t)len)] ^= (char)(1 << rnd_below(8));
		return len;
	}
	case 1: { // random byte
		if (len == 0) return len;
		buf[rnd_below((uint32_t)len)] = (char)rnd_below(256);
		return len;
	}
	case 2: { // delete a span
		if (len < 2) return len;
		int32_t at = (int32_t)rnd_below((uint32_t)len);
		int32_t n  = 1 + (int32_t)rnd_below(64);
		if (at + n > len) n = len - at;
		memmove(buf + at, buf + at + n, (size_t)(len - at - n));
		return len - n;
	}
	case 3: { // duplicate a span
		if (len == 0 || len >= MAX_INPUT - 256) return len;
		int32_t at = (int32_t)rnd_below((uint32_t)len);
		int32_t n  = 1 + (int32_t)rnd_below(128);
		if (at + n > len) n = len - at;
		if (len + n > MAX_INPUT) return len;
		memmove(buf + at + n, buf + at, (size_t)(len - at));
		return len + n;
	}
	case 4: { // insert random ASCII
		if (len >= MAX_INPUT - 64) return len;
		int32_t at = (int32_t)rnd_below((uint32_t)len + 1);
		int32_t n  = 1 + (int32_t)rnd_below(32);
		memmove(buf + at + n, buf + at, (size_t)(len - at));
		for (int32_t i = 0; i < n; i++) buf[at + i] = (char)(32 + rnd_below(95));
		return len + n;
	}
	case 5: { // splice from another seed
		if (seed_count == 0 || len >= MAX_INPUT - 512) return len;
		const seed_t *s = &seeds[rnd_below((uint32_t)seed_count)];
		if (s->len == 0) return len;
		int32_t from = (int32_t)rnd_below((uint32_t)s->len);
		int32_t n    = 1 + (int32_t)rnd_below(256);
		if (from + n > s->len) n = s->len - from;
		int32_t at = (int32_t)rnd_below((uint32_t)len + 1);
		memmove(buf + at + n, buf + at, (size_t)(len - at));
		memcpy(buf + at, s->data + from, (size_t)n);
		return len + n;
	}
	case 6: case 7: { // insert a dictionary token (weighted up)
		const char *tok = dictionary[rnd_below(sizeof(dictionary) / sizeof(dictionary[0]))];
		int32_t     n   = (int32_t)strlen(tok);
		if (len + n >= MAX_INPUT) return len;
		int32_t at = (int32_t)rnd_below((uint32_t)len + 1);
		memmove(buf + at + n, buf + at, (size_t)(len - at));
		memcpy(buf + at, tok, (size_t)n);
		return len + n;
	}
	case 8: { // truncate
		if (len == 0) return len;
		return (int32_t)rnd_below((uint32_t)len);
	}
	default: { // hammer one character many times (lexer/pp loops)
		if (len >= MAX_INPUT - 4096) return len;
		int32_t at = (int32_t)rnd_below((uint32_t)len + 1);
		int32_t n  = 16 + (int32_t)rnd_below(2048);
		char    ch = "({[<#\"'/*.,;x0 \n"[rnd_below(16)];
		memmove(buf + at + n, buf + at, (size_t)(len - at));
		memset(buf + at, ch, (size_t)n);
		return len + n;
	}
	}
}

static int32_t build_case(char *buf) {
	int32_t len = 0;
	if (seed_count > 0 && rnd_below(100) != 0) { // 1%: start from nothing
		const seed_t *s = &seeds[rnd_below((uint32_t)seed_count)];
		len = s->len < MAX_INPUT ? s->len : MAX_INPUT;
		memcpy(buf, s->data, (size_t)len);
	}
	int32_t rounds = 1 + (int32_t)rnd_below(12);
	for (int32_t i = 0; i < rounds; i++)
		len = mutate(buf, len);
	buf[len] = '\0';
	return len;
}

// --- one compile ------------------------------------------------------------------

static svsl_include_src_t fuzz_include(void *user, const char *path, const char *requester) {
	(void)user; (void)path; (void)requester;
	// serve the real stereokit.hlsli for every include; pp's nesting
	// limit guards against include recursion
	return (svsl_include_src_t){ .content = include_src, .length = include_len,
	                             .path = "stereokit.hlsli" };
}

static void compile_one(const char *src) {
	svsl_arena_t arena = {0};

	svsl_pp_options_t opt   = { .include_cb = fuzz_include };
	svsl_pp_result_t  pp;
	svsl_diag_list_t  diags = {0};
	svsl_pp_run(&arena, src, "fuzz.hlsl", &opt, &pp, &diags);

	svsl_token_list_t tokens = {0};
	svsl_lex(&arena, &pp, &tokens, &diags);

	svsl_ast_t *ast = svsl_parse(&arena, &tokens, &diags);

	svsl_program_t program;
	svsl_sema_run(&arena, ast, &pp, "fuzz.hlsl", NULL, &program, &diags);

	svsl_ir_module_t ir = {0};
	if (diags.error_count == 0)
		svsl_ir_build(&arena, &program, svsl_opt_default, &ir, &diags);

	if (diags.error_count == 0) {
		for (int32_t i = 0; i < ir.func_count; i++) {
			svsl_spirv_blob_t blob = {0};
			if (!svsl_spirv_emit(&arena, &program, &ir.funcs[i], &blob, &diags)) break;
		}
	}
	svsl_arena_free(&arena);
}

// --- seed loading ------------------------------------------------------------------

static char *read_file(const char *path, int32_t *out_len) {
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

static int compare_names(const void *a, const void *b) {
	return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static void load_seed_dir(const char *subdir) {
	char dir_path[1024];
	snprintf(dir_path, sizeof(dir_path), "%s/shaders/%s", SVSL_TEST_DIR, subdir);
	DIR *dir = opendir(dir_path);
	if (!dir) return;

	char          *names[MAX_SEEDS];
	int32_t        name_count = 0;
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL && name_count < MAX_SEEDS) {
		const char *dot = strrchr(entry->d_name, '.');
		if (!dot || (strcmp(dot, ".hlsl") != 0 && strcmp(dot, ".svsl") != 0)) continue;
		char *copy = malloc(strlen(entry->d_name) + 1);
		strcpy(copy, entry->d_name);
		names[name_count++] = copy;
	}
	closedir(dir);
	qsort(names, (size_t)name_count, sizeof(char *), compare_names); // determinism

	for (int32_t i = 0; i < name_count && seed_count < MAX_SEEDS; i++) {
		char path[1300];
		snprintf(path, sizeof(path), "%s/%s", dir_path, names[i]);
		int32_t len  = 0;
		char   *data = read_file(path, &len);
		if (data) seeds[seed_count++] = (seed_t){ .data = data, .len = len };
		free(names[i]);
	}
}

// --- driver ------------------------------------------------------------------

static volatile int64_t current_iter = -1;

static void on_alarm(int sig) {
	(void)sig;
	fprintf(stderr, "\nfuzz: HANG at iteration %lld — reproduce with -dump %lld\n",
	        (long long)current_iter, (long long)current_iter);
	_exit(2);
}

int main(int argc, char **argv) {
	uint64_t seed     = 1;
	int64_t  runs     = 100000;
	int64_t  dump_at  = -1;
	const char *file  = NULL;

	for (int i = 1; i < argc; i++) {
		if      (strcmp(argv[i], "-seed") == 0 && i + 1 < argc) seed    = strtoull(argv[++i], NULL, 10);
		else if (strcmp(argv[i], "-runs") == 0 && i + 1 < argc) runs    = strtoll (argv[++i], NULL, 10);
		else if (strcmp(argv[i], "-dump") == 0 && i + 1 < argc) dump_at = strtoll (argv[++i], NULL, 10);
		else if (strcmp(argv[i], "-file") == 0 && i + 1 < argc) file    = argv[++i];
		else { fprintf(stderr, "usage: svsl_fuzz [-seed N] [-runs N] [-dump ITER] [-file path]\n"); return 1; }
	}

	include_src = read_file(SVSL_TEST_DIR "/shaders/include/stereokit.hlsli", &include_len);
	bool include_owned = include_src != NULL; // "" fallback is a literal, must not be freed
	if (!include_src) { include_src = ""; include_len = 0; }

	if (file) { // reproduce a saved case
		int32_t len = 0;
		char   *src = read_file(file, &len);
		if (!src) { fprintf(stderr, "cannot read '%s'\n", file); return 1; }
		compile_one(src);
		printf("fuzz: '%s' compiled without crashing\n", file);
		return 0;
	}

	const char *dirs[] = { "builtin", "examples", "ported", "morrowind", "checks" };
	for (size_t i = 0; i < sizeof(dirs) / sizeof(dirs[0]); i++)
		load_seed_dir(dirs[i]);
	printf("fuzz: %d seeds, seed=%llu, runs=%lld\n",
	       seed_count, (unsigned long long)seed, (long long)runs);

	rng_state = seed ? seed : 1;
	char *buf = malloc(MAX_INPUT + 1);

	signal(SIGALRM, on_alarm);

	for (int64_t iter = 0; iter < runs || dump_at >= 0; iter++) {
		current_iter = iter;
		int32_t len  = build_case(buf);

		if (dump_at >= 0) {
			if (iter < dump_at) continue; // fast-forward the PRNG, no compiling
			FILE *f = fopen("fuzz_case.hlsl", "wb");
			fwrite(buf, 1, (size_t)len, f);
			fclose(f);
			printf("fuzz: iteration %lld written to fuzz_case.hlsl (%d bytes)\n",
			       (long long)iter, len);
			return 0;
		}

		alarm(10); // a single case taking 10s is a hang finding
		compile_one(buf);
		alarm(0);

		if ((iter + 1) % 10000 == 0) {
			printf("fuzz: %lld/%lld\n", (long long)(iter + 1), (long long)runs);
			fflush(stdout);
		}
	}
	printf("fuzz: done, %lld runs clean (crash-free; reproduce any case with -seed %llu -dump ITER)\n",
	       (long long)runs, (unsigned long long)seed);
	free(buf);
	if (include_owned) free(include_src);
	return 0;
}

// svslc — SVSL command-line compiler. Default output is StereoKit's .sks
// container; -spv writes raw per-stage SPIR-V and -h an embeddable C header.
//
// The CLI talks to libsvsl only through <svsl/svsl.h>: it handles files, flags,
// include resolution, and shelling out to spirv-val, then hands source to
// svsl_compile and writes whatever products the flags asked for.

#include <svsl/svsl.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef S_ISDIR // MSVC's <sys/stat.h> has the mode bits but not the macro
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif

typedef struct cli_t {
	const char   *inputs[64];
	int32_t       input_count;
	const char   *include_dirs[16];
	int32_t       include_dir_count;
	svsl_define_t defines[32];
	int32_t       define_count;
	const char   *entry_vs, *entry_ps, *entry_cs;
	bool          reflect;
	bool          dump_ir;
	bool          sks;      // default when no output flag is given
	bool          spv;
	bool          header;
	uint32_t      targets;  // svsl_target_ bits from -t (0 = spirv only)
	bool          validate;
	bool          force;
	bool          half_strict16;
	bool          porting;
	svsl_opt_level_ opt_level;
	const char   *out_path; // file or directory
} cli_t;

// last path separator, or NULL; Windows paths may use either slash
static const char *last_sep(const char *path) {
	const char *sep = strrchr(path, '/');
#ifdef _WIN32
	const char *back = strrchr(path, '\\');
	if (back && (!sep || back > sep)) sep = back;
#endif
	return sep;
}

static bool is_dir(const char *path) {
	struct stat st;
	return path && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}
static long mtime_of(const char *path) {
	struct stat st;
	return stat(path, &st) == 0 ? (long)st.st_mtime : -1;
}

// output path: -o file, -o dir/<stem>.<ext>, or alongside the input
static void make_out_path(const cli_t *cli, const char *input, const char *ext,
                          char *out, size_t out_size) {
	if (cli->out_path && !is_dir(cli->out_path) && cli->input_count == 1 &&
	    !cli->spv) { // -spv writes several files; treat -o as dir then
		snprintf(out, out_size, "%s", cli->out_path);
		return;
	}
	const char *slash = last_sep(input);
	const char *stem  = slash ? slash + 1 : input;
	const char *dot   = strrchr(stem, '.');
	int32_t     len   = dot ? (int32_t)(dot - stem) : (int32_t)strlen(stem);
	if (cli->out_path)
		snprintf(out, out_size, "%s/%.*s%s", cli->out_path, len, stem, ext);
	else
		snprintf(out, out_size, "%.*s%.*s%s", (int32_t)(stem - input), input, len, stem, ext);
}

static char *read_file(const char *path, int32_t *out_len) {
	FILE *f = fopen(path, "rb");
	if (!f) return NULL;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *data = malloc((size_t)size + 1);
	if (!data) { fclose(f); return NULL; }
	size_t got = fread(data, 1, (size_t)size, f);
	fclose(f);
	data[got] = '\0';
	if (out_len) *out_len = (int32_t)got;
	return data;
}

// The include callback runs during svsl_compile, which copies whatever it keeps;
// we hold each buffer we hand out until the compile returns, then free them all.
typedef struct include_ctx_t {
	const cli_t *cli;
	void        *allocs[512];
	int32_t      alloc_count;
} include_ctx_t;

static void *track(include_ctx_t *ctx, void *p) {
	if (p && ctx->alloc_count < 512) ctx->allocs[ctx->alloc_count++] = p;
	return p;
}
static void free_tracked(include_ctx_t *ctx) {
	for (int32_t i = 0; i < ctx->alloc_count; i++) free(ctx->allocs[i]);
	ctx->alloc_count = 0;
}
static char *dup_str(const char *s) {
	size_t n = strlen(s) + 1;
	char  *c = malloc(n);
	if (c) memcpy(c, s, n);
	return c;
}

static svsl_include_src_t make_src(include_ctx_t *ctx, char *content, int32_t len, const char *path) {
	return (svsl_include_src_t){ .content = track(ctx, content), .length = len,
	                             .path = track(ctx, dup_str(path)) };
}

static svsl_include_src_t cli_include(void *user, const char *path, const char *requester) {
	include_ctx_t *ctx = user;
	char           full[1024];

	// requester's directory first, then -i search paths
	const char *slash = requester ? last_sep(requester) : NULL;
	if (slash) {
		snprintf(full, sizeof(full), "%.*s/%s", (int32_t)(slash - requester), requester, path);
		int32_t len;
		char   *content = read_file(full, &len);
		if (content) return make_src(ctx, content, len, full);
	}
	for (int32_t i = 0; i < ctx->cli->include_dir_count; i++) {
		snprintf(full, sizeof(full), "%s/%s", ctx->cli->include_dirs[i], path);
		int32_t len;
		char   *content = read_file(full, &len);
		if (content) return make_src(ctx, content, len, full);
	}
	return (svsl_include_src_t){0};
}

static void print_diags(const svsl_diag_t *diags, int32_t count) {
	static const char *severities[] = { "error", "warning", "porting", "info" };
	for (int32_t i = 0; i < count; i++) {
		const svsl_diag_t *d = &diags[i];
		fprintf(stderr, "%s:%d:%d: %s: %s\n",
		        d->loc.file ? d->loc.file : "<unknown>", d->loc.line, d->loc.col,
		        severities[d->severity], d->message);
	}
}

static const char *stage_ext_for(svsl_stage_ stage) {
	return stage == svsl_stage_vertex ? ".vert.spv" :
	       stage == svsl_stage_pixel  ? ".frag.spv" : ".comp.spv";
}

static bool write_bytes(const char *out_path, const void *data, size_t size) {
	FILE *f = fopen(out_path, "wb");
	if (!f) { fprintf(stderr, "svslc: cannot write '%s'\n", out_path); return false; }
	fwrite(data, 1, size, f);
	fclose(f);
	return true;
}

static bool compile_file(const cli_t *cli, const char *path) {
	int32_t source_len;
	char   *source = read_file(path, &source_len);
	if (!source) {
		fprintf(stderr, "svslc: cannot open '%s'\n", path);
		return false;
	}

	include_ctx_t  ictx = { .cli = cli };
	svsl_options_t opts = {
		.defines = cli->defines, .define_count = cli->define_count,
		.include_cb = cli_include, .include_user = &ictx,
		.entry_vertex = cli->entry_vs, .entry_pixel = cli->entry_ps, .entry_compute = cli->entry_cs,
		.opt_level = cli->opt_level, .half_strict16 = cli->half_strict16,
		.targets = cli->targets,
		.porting_hints = cli->porting };
	svsl_result_t *r = svsl_compile(&(svsl_source_t){ .text = source, .length = source_len,
	                                                  .filename = path }, &opts);
	free(source);
	free_tracked(&ictx);
	if (!r) { fprintf(stderr, "svslc: out of memory\n"); return false; }

	print_diags(r->diagnostics, r->diagnostic_count);
	bool ok = r->ok;

	if (ok && cli->reflect)
		printf("%s", svsl_result_reflection(r));
	if (ok && cli->dump_ir)
		printf("%s", svsl_result_ir(r));

	if (ok && (cli->spv || cli->validate)) {
		for (int32_t i = 0; i < r->stage_count; i++) {
			const svsl_stage_output_t *st = &r->stages[i];
			const char *stage_ext = stage_ext_for(st->stage);
			char        out_path[1024];
			if (cli->spv) {
				// an explicit `-o name.spv` is honored exactly (stage-suffixed
				// when several entry points each need a file)
				const char *op  = cli->out_path;
				const char *ext = op && !is_dir(op) && cli->input_count == 1 ? strrchr(op, '.') : NULL;
				if (ext && strcmp(ext, ".spv") == 0 && r->stage_count == 1)
					snprintf(out_path, sizeof(out_path), "%s", op);
				else if (ext && strcmp(ext, ".spv") == 0)
					snprintf(out_path, sizeof(out_path), "%.*s%s", (int32_t)(ext - op), op, stage_ext);
				else
					make_out_path(cli, path, stage_ext, out_path, sizeof(out_path));
			} else { // validation-only temp file, next to the real output
				char base[1000];
				make_out_path(cli, path, ".sks", base, sizeof(base));
				snprintf(out_path, sizeof(out_path), "%.*s%s",
				         (int32_t)(sizeof(out_path) - 16), base, stage_ext);
			}

			if (!write_bytes(out_path, st->spirv, (size_t)st->spirv_word_count * 4)) { ok = false; continue; }

			if (cli->spv && (cli->targets & svsl_target_wgsl) && st->wgsl) { // debugging aid: raw WGSL text beside the .spv
				char wgsl_path[1040];
				snprintf(wgsl_path, sizeof(wgsl_path), "%.*s.wgsl", (int32_t)strlen(out_path) - 4, out_path);
				if (!write_bytes(wgsl_path, st->wgsl, (size_t)st->wgsl_length)) ok = false;
			}

			// WGSL stages validate through naga when it's on PATH — optional,
			// like the corpus tests, since naga is a cargo tool not a SDK one
			if (cli->validate && st->wgsl) {
				static int32_t has_naga = -1;
				if (has_naga < 0) {
					has_naga = system("naga --version > /dev/null 2>&1") == 0 ? 1 : 0;
					if (!has_naga)
						fprintf(stderr, "svslc: naga not on PATH; WGSL stages not validated\n");
				}
				if (has_naga) {
					char wgsl_path[1040], cmd[1200];
					snprintf(wgsl_path, sizeof(wgsl_path), "%.*s.wgsl", (int32_t)strlen(out_path) - 4, out_path);
					bool have_file = cli->spv && (cli->targets & svsl_target_wgsl);
					if (!have_file && !write_bytes(wgsl_path, st->wgsl, (size_t)st->wgsl_length)) continue;
					snprintf(cmd, sizeof(cmd), "naga %s", wgsl_path);
					if (system(cmd) != 0) {
						fprintf(stderr, "svslc: WGSL validation failed for '%s'\n", wgsl_path);
						ok = false;
					}
					if (!have_file) remove(wgsl_path);
				}
			}

			if (cli->validate) {
				char cmd[1200];
				snprintf(cmd, sizeof(cmd), "spirv-val --target-env vulkan1.1%s %s",
				         svsl_result_needs_scalar_layout(r) ? " --scalar-block-layout" : "",
				         out_path);
				if (system(cmd) != 0) {
					fprintf(stderr, "svslc: validation failed for '%s'\n", out_path);
					ok = false;
				}
				if (!cli->spv) remove(out_path);
			}
		}
	}

	if (ok && cli->sks) {
		svsl_bytes_t sks = svsl_result_sks(r);
		char         out_path[1024];
		make_out_path(cli, path, ".sks", out_path, sizeof(out_path));
		if (!sks.data || !write_bytes(out_path, sks.data, (size_t)sks.size)) ok = false;
		else printf("svslc: %s -> %s\n", path, out_path);
	}
	if (ok && cli->header) {
		const char *slash = last_sep(path);
		const char *stem  = slash ? slash + 1 : path;
		const char *text  = svsl_result_header(r, stem);
		char        out_path[1024];
		make_out_path(cli, path, ".sks.h", out_path, sizeof(out_path));
		if (!text || !write_bytes(out_path, text, strlen(text))) ok = false;
		else printf("svslc: %s -> %s\n", path, out_path);
	}

	svsl_result_free(r);
	return ok;
}

// like skshaderc: skip recompiling when the output is newer than the input.
// Only the top-level input's mtime is checked — edits to #include'd files won't
// trigger a rebuild. Matches skshaderc; use -f after touching a header.
static bool output_is_fresh(const cli_t *cli, const char *input) {
	if (cli->force || !cli->sks || cli->spv || cli->header || cli->reflect ||
	    cli->dump_ir || cli->validate)
		return false;
	char out_path[1024];
	make_out_path(cli, input, ".sks", out_path, sizeof(out_path));
	long in_time  = mtime_of(input);
	long out_time = mtime_of(out_path);
	return in_time >= 0 && out_time >= 0 && out_time >= in_time;
}

static void usage(void) {
	fprintf(stderr,
		"usage: svslc [options] <files...>\n"
		"  -o <path>          output file or directory\n"
		"  -sks | -spv | -h   output format (default -sks)\n"
		"  -t <s|w|sw>        container languages: SPIR-V, WGSL, or both (default s;\n"
		"                     WGSL needs an SVSL_ENABLE_WGSL build)\n"
		"  -vs/-ps/-cs <name> entry point names (default vs/ps/cs)\n"
		"  -i <dir>           include search path\n"
		"  -D NAME[=VALUE]    preprocessor define\n"
		"  -r                 print reflection\n"
		"  --dump-ir          print the IR\n"
		"  --validate         run spirv-val on the output\n"
		"  --half=strict16    treat every `half` as an exact float16\n"
		"  -Wporting          hint on legacy HLSL spellings (off by default)\n"
		"  -O0/-O1/-O2        optimization level (default -O1; -O2 enables float algebra)\n"
		"  -f                 recompile even when the output is newer\n");
}

int main(int argc, char **argv) {
	cli_t cli = { .opt_level = svsl_opt_default };
	bool  format_set = false;

	for (int32_t i = 1; i < argc; i++) {
		const char *arg = argv[i];
		if (strcmp(arg, "--version") == 0) {
			printf("svslc %d.%d.%d\n", SVSL_VERSION_MAJOR, SVSL_VERSION_MINOR, SVSL_VERSION_PATCH);
			return 0;
		}
		if (strcmp(arg, "-r") == 0) { cli.reflect = true; continue; }
		if (strcmp(arg, "--dump-ir") == 0) { cli.dump_ir = true; continue; }
		if (strcmp(arg, "-sks") == 0) { cli.sks = true; format_set = true; continue; }
		if (strcmp(arg, "-spv") == 0) { cli.spv = true; format_set = true; continue; }
		if (strcmp(arg, "-h") == 0) { cli.header = true; format_set = true; continue; }
		if (strcmp(arg, "-t") == 0 && i + 1 < argc) { // skshaderc-style language set
			for (const char *c = argv[++i]; *c; c++) {
				if      (*c == 's') cli.targets |= svsl_target_spirv;
				else if (*c == 'w') cli.targets |= svsl_target_wgsl;
				else { fprintf(stderr, "svslc: unknown target '%c' in -t (use s and/or w)\n", *c); return 1; }
			}
			continue;
		}
		if (strcmp(arg, "--validate") == 0) { cli.validate = true; continue; }
		if (strcmp(arg, "--half=strict16") == 0) { cli.half_strict16 = true; continue; }
		if (strcmp(arg, "-Wporting") == 0)       { cli.porting = true;       continue; }
		if (strcmp(arg, "-O0") == 0) { cli.opt_level = svsl_opt_none;       continue; }
		if (strcmp(arg, "-O1") == 0) { cli.opt_level = svsl_opt_default;    continue; }
		if (strcmp(arg, "-O2") == 0) { cli.opt_level = svsl_opt_aggressive; continue; }
		if (strcmp(arg, "-f") == 0) { cli.force = true; continue; }
		if (strcmp(arg, "-vs") == 0 && i + 1 < argc) { cli.entry_vs = argv[++i]; continue; }
		if (strcmp(arg, "-ps") == 0 && i + 1 < argc) { cli.entry_ps = argv[++i]; continue; }
		if (strcmp(arg, "-cs") == 0 && i + 1 < argc) { cli.entry_cs = argv[++i]; continue; }
		if (strcmp(arg, "-o") == 0 && i + 1 < argc) { cli.out_path = argv[++i]; continue; }
		if (strcmp(arg, "-i") == 0 && i + 1 < argc) {
			const char *dir = argv[++i]; // consume the operand even at the cap
			if (cli.include_dir_count < 16) cli.include_dirs[cli.include_dir_count++] = dir;
			else fprintf(stderr, "svslc: too many -i include dirs (max 16), ignoring '%s'\n", dir);
			continue;
		}
		if (strcmp(arg, "-D") == 0 && i + 1 < argc) {
			char *def = argv[++i]; // consume the operand even at the cap
			if (cli.define_count < 32) {
				char *eq = strchr(def, '=');
				if (eq) *eq = '\0';
				cli.defines[cli.define_count++] = (svsl_define_t){ .name = def, .value = eq ? eq + 1 : NULL };
			} else fprintf(stderr, "svslc: too many -D defines (max 32), ignoring '%s'\n", def);
			continue;
		}
		if (arg[0] == '-') {
			fprintf(stderr, "svslc: unknown option '%s'\n", arg);
			usage();
			return 1;
		}
		if (cli.input_count < 64) cli.inputs[cli.input_count++] = arg;
	}
	// default output: -r/--dump-ir/--validate alone are inspect/check modes with
	// no file output, but an explicit -o always produces one; the format follows
	// the -o extension so `-o shader.spv` never silently writes an SKS container
	if (!format_set && (cli.out_path || (!cli.reflect && !cli.dump_ir && !cli.validate))) {
		const char *ext = cli.out_path ? strrchr(cli.out_path, '.') : NULL;
		if      (ext && strcmp(ext, ".spv") == 0) cli.spv    = true;
		else if (ext && strcmp(ext, ".h")   == 0) cli.header = true;
		else                                      cli.sks    = true;
	}

	if (cli.input_count == 0) {
		usage();
		return 1;
	}

	bool all_ok = true;
	for (int32_t i = 0; i < cli.input_count; i++) {
		if (output_is_fresh(&cli, cli.inputs[i])) continue;
		all_ok = compile_file(&cli, cli.inputs[i]) && all_ok;
	}
	return all_ok ? 0 : 1;
}

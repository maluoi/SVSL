// svsl_view — batch visual comparison of svslc against skshaderc.
//   svsl_view -test <dir> [<dir>...]   compare every shader, exit nonzero on regression
//   svsl_view -file <shader>           compare a single shader, with SPIR-V diff
//   svsl_view -output <dir>            image directory (default: compare/ next to the binary)
//   svsl_view -no-diff                 suppress the SPIR-V diff in -file mode
//
// Every comparison writes <shader>_ref.png / <shader>_svsl.png to the image
// directory, so mismatches can be inspected by eye.

#include "view.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int compare_names(const void *a, const void *b) {
	return strcmp(*(const char *const *)a, *(const char *const *)b);
}

typedef struct stats_t {
	int32_t pass, fail, skip;
} stats_t;

static void run_one(scene_t *scene, const char *path, const char *include_dir,
                    const char *out_dir, bool spirv_diff, stats_t *stats) {
	compare_result_t r = compare_shader(scene, path, include_dir, out_dir, spirv_diff);
	const char *base = strrchr(path, '/') ? strrchr(path, '/') + 1 : path;
	if (r.skipped) {
		printf("  skip  %-44s %s\n", base, r.message);
		stats->skip++;
	} else if (r.ok) {
		printf("  ok    %-44s %s\n", base, r.message);
		stats->pass++;
	} else {
		printf("  FAIL  %-44s %s\n", base, r.message);
		stats->fail++;
	}
	fflush(stdout);
}

static void run_dir(scene_t *scene, const char *dir_path, const char *include_dir,
                    const char *out_dir, stats_t *stats) {
	DIR *dir = opendir(dir_path);
	if (!dir) {
		printf("cannot open directory '%s'\n", dir_path);
		stats->fail++;
		return;
	}
	char *names[256];
	char *subdirs[64];
	int32_t count = 0, subdir_count = 0;
	struct dirent *entry;
	while ((entry = readdir(dir)) != NULL && count < 256) {
		if (entry->d_name[0] == '.') continue;
		char entry_path[2048];
		snprintf(entry_path, sizeof(entry_path), "%s/%s", dir_path, entry->d_name);
		struct stat st;
		if (stat(entry_path, &st) == 0 && S_ISDIR(st.st_mode)) { // recurse; 'include' holds headers, not shaders
			if (strcmp(entry->d_name, "include") != 0 && subdir_count < 64) {
				subdirs[subdir_count] = malloc(strlen(entry->d_name) + 1);
				strcpy(subdirs[subdir_count], entry->d_name);
				subdir_count++;
			}
			continue;
		}
		const char *ext = strrchr(entry->d_name, '.');
		if (!ext || (strcmp(ext, ".hlsl") != 0 && strcmp(ext, ".svsl") != 0)) continue;
		names[count] = malloc(strlen(entry->d_name) + 1);
		strcpy(names[count], entry->d_name);
		count++;
	}
	closedir(dir);
	qsort(names,   (size_t)count,        sizeof(names[0]),   compare_names);
	qsort(subdirs, (size_t)subdir_count, sizeof(subdirs[0]), compare_names);

	for (int32_t i = 0; i < count; i++) {
		char path[2048];
		snprintf(path, sizeof(path), "%s/%s", dir_path, names[i]);
		run_one(scene, path, include_dir, out_dir, false, stats);
		free(names[i]);
	}
	for (int32_t i = 0; i < subdir_count; i++) {
		char path[2048];
		snprintf(path, sizeof(path), "%s/%s", dir_path, subdirs[i]);
		run_dir(scene, path, include_dir, out_dir, stats);
		free(subdirs[i]);
	}
}

int main(int argc, char **argv) {
	const char *test_dirs[8];
	int32_t     test_dir_count = 0;
	const char *single_file    = NULL;
	const char *out_dir        = NULL;
	bool        spirv_diff     = true; // -file mode only
	char        include_dir[1024];
	snprintf(include_dir, sizeof(include_dir), "%s/include", SVSL_SHADER_DIR);

	for (int32_t i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-test") == 0 && i + 1 < argc) {
			while (i + 1 < argc && argv[i + 1][0] != '-' && test_dir_count < 8)
				test_dirs[test_dir_count++] = argv[++i];
		}
		else if (strcmp(argv[i], "-file") == 0 && i + 1 < argc)   single_file = argv[++i];
		else if (strcmp(argv[i], "-output") == 0 && i + 1 < argc) out_dir = argv[++i];
		else if (strcmp(argv[i], "-no-diff") == 0)                spirv_diff = false;
		else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
			snprintf(include_dir, sizeof(include_dir), "%s", argv[++i]);
		else {
			fprintf(stderr, "usage: svsl_view [-test <dir>...] [-file <shader>] [-output <dir>] [-no-diff] [-i <include>]\n");
			return 1;
		}
	}
	if (test_dir_count == 0 && !single_file) { // default: the pinned corpus
		static char builtin_dir[1024], examples_dir[1024], morrowind_dir[1024], checks_dir[1024];
		snprintf(builtin_dir, sizeof(builtin_dir), "%s/builtin", SVSL_SHADER_DIR);
		snprintf(examples_dir, sizeof(examples_dir), "%s/examples", SVSL_SHADER_DIR);
		snprintf(morrowind_dir, sizeof(morrowind_dir), "%s/morrowind", SVSL_SHADER_DIR);
		snprintf(checks_dir, sizeof(checks_dir), "%s/checks", SVSL_SHADER_DIR);
		test_dirs[test_dir_count++] = builtin_dir;
		test_dirs[test_dir_count++] = examples_dir;
		test_dirs[test_dir_count++] = morrowind_dir;
		test_dirs[test_dir_count++] = checks_dir;
	}
	if (!out_dir) { // default: compare/ next to the binary, so results are easy to find
		static char default_out[1024];
		const char *slash = strrchr(argv[0], '/');
		if (slash)
			snprintf(default_out, sizeof(default_out), "%.*s/compare",
			         (int32_t)(slash - argv[0]), argv[0]);
		else
			snprintf(default_out, sizeof(default_out), "compare");
		out_dir = default_out;
	}
	mkdir(out_dir, 0755); // fine if it already exists

	// headless renderer: no window, no swapchain. StereoKit's binding slots:
	// material $Global at b0, system stereokit_buffer at b1, sk_inst at t12.
	static const skr_bind_settings_t binds = {
		.material_slot = 0,
		.system_slot   = 1,
		.instance_slot = 12,
	};
	skr_settings_t settings = {
		.app_name          = "svsl_view",
		.app_version       = 1,
		.enable_validation = false,
		.bind_settings     = &binds,
	};
	if (!skr_init(settings)) {
		fprintf(stderr, "svsl_view: sk_renderer init failed (no Vulkan device?)\n");
		return 1;
	}

	scene_t scene;
	if (!scene_init(&scene)) {
		fprintf(stderr, "svsl_view: scene setup failed\n");
		return 1;
	}

	stats_t stats = {0};
	if (single_file)
		run_one(&scene, single_file, include_dir, out_dir, spirv_diff, &stats);
	for (int32_t i = 0; i < test_dir_count; i++) {
		printf("%s:\n", test_dirs[i]);
		run_dir(&scene, test_dirs[i], include_dir, out_dir, &stats);
	}

	printf("\n%d passed, %d failed, %d skipped\n", stats.pass, stats.fail, stats.skip);
	printf("images: %s\n", out_dir);

	scene_destroy(&scene);
	skr_shutdown();
	return stats.fail == 0 ? 0 : 1;
}

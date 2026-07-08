// svsl_tests — unit + corpus test runner.
// Runs all suites, or only those named on the command line.

#include "test.h"

#include <stdbool.h>
#include <string.h>

int32_t test_checks = 0;
int32_t test_fails  = 0;

void test_util  (void);
void test_pp    (void);
void test_lexer (void);
void test_parser(void);
void test_layout(void);
void test_sema  (void);
void test_ir    (void);
void test_sks   (void);
void test_api   (void);
void test_corpus(void);

typedef struct {
	const char *name;
	void      (*run)(void);
} test_suite_t;

static const test_suite_t suites[] = {
	{ "util",   test_util   },
	{ "pp",     test_pp     },
	{ "lexer",  test_lexer  },
	{ "parser", test_parser },
	{ "layout", test_layout },
	{ "sema",   test_sema   },
	{ "ir",     test_ir     },
	{ "sks",    test_sks    },
	{ "api",    test_api    },
	{ "corpus", test_corpus },
};

int main(int argc, char **argv) {
	int32_t suite_count = (int32_t)(sizeof(suites) / sizeof(suites[0]));

	// unknown suite names are an error, not a silent pass
	for (int32_t a = 1; a < argc; a++) {
		bool known = false;
		for (int32_t i = 0; i < suite_count; i++)
			if (strcmp(argv[a], suites[i].name) == 0) known = true;
		if (!known) {
			printf("unknown test suite '%s'\n", argv[a]);
			return 1;
		}
	}

	for (int32_t i = 0; i < suite_count; i++) {
		bool run = argc <= 1;
		for (int32_t a = 1; a < argc; a++)
			if (strcmp(argv[a], suites[i].name) == 0) run = true;
		if (!run) continue;

		int32_t fails_before = test_fails;
		suites[i].run();
		printf("%-8s %s\n", suites[i].name, test_fails == fails_before ? "ok" : "FAILED");
	}
	printf("%d checks, %d failures\n", test_checks, test_fails);
	return test_fails == 0 ? 0 : 1;
}

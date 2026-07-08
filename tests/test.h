// Minimal test harness: plain C asserts that report and count failures.

#pragma once

#include <stdint.h>
#include <stdio.h>

extern int32_t test_checks;
extern int32_t test_fails;

#define TEST_CHECK(cond) do { \
	test_checks++; \
	if (!(cond)) { \
		test_fails++; \
		printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
	} \
} while (0)

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

// One entry point per suite, dispatched by the table in main.c.
void test_util   (void);
void test_pp     (void);
void test_lexer  (void);
void test_parser (void);
void test_layout (void);
void test_sema   (void);
void test_ir     (void);
void test_sks    (void);
void test_api    (void);
void test_qcom   (void);
void test_formats(void);
void test_wgsl   (void);
void test_corpus (void);

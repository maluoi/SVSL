add_executable(svsl_tests
	main.c
	test_util.c
	test_pp.c
	test_lexer.c
	test_parser.c
	test_layout.c
	test_sema.c
	test_ir.c
	test_sks.c
	test_api.c
	test_corpus.c
)
target_link_libraries     (svsl_tests PRIVATE svsl)
target_include_directories(svsl_tests PRIVATE ../src)
target_compile_definitions(svsl_tests PRIVATE SVSL_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# dev-only mutation fuzzer; build with -DCMAKE_C_FLAGS="-fsanitize=address,undefined"
# in a separate build dir for real coverage (see fuzz_main.c header)
add_executable            (svsl_fuzz fuzz_main.c)
target_link_libraries     (svsl_fuzz PRIVATE svsl)
target_include_directories(svsl_fuzz PRIVATE ../src)
target_compile_definitions(svsl_fuzz PRIVATE SVSL_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# dev-only performance harness (see bench_main.c). Not wired into ctest.
add_executable            (svsl_bench bench_main.c)
target_link_libraries     (svsl_bench PRIVATE svsl)
target_include_directories(svsl_bench PRIVATE ../src)
target_compile_definitions(svsl_bench PRIVATE SVSL_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_test(NAME util   COMMAND svsl_tests util)
add_test(NAME pp     COMMAND svsl_tests pp)
add_test(NAME lexer  COMMAND svsl_tests lexer)
add_test(NAME parser COMMAND svsl_tests parser)
add_test(NAME layout COMMAND svsl_tests layout)
add_test(NAME sema   COMMAND svsl_tests sema)
add_test(NAME ir     COMMAND svsl_tests ir)
add_test(NAME sks    COMMAND svsl_tests sks)
add_test(NAME api    COMMAND svsl_tests api)
add_test(NAME corpus COMMAND svsl_tests corpus)

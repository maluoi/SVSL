cmake_minimum_required(VERSION 3.21)

# Version lives in include/svsl/svsl.h; parse it rather than duplicating it here.
file(STRINGS include/svsl/svsl.h _svsl_ver_lines REGEX "#define SVSL_VERSION_(MAJOR|MINOR|PATCH)")
string(REGEX MATCHALL "[0-9]+" _svsl_ver "${_svsl_ver_lines}")
list(JOIN _svsl_ver "." _svsl_ver)

project(svsl VERSION ${_svsl_ver} LANGUAGES C)

# Dev targets default on only when SVSL is the top-level project, so consuming
# it via add_subdirectory/FetchContent builds just the library.
option(SVSL_BUILD_CLI   "Build the svslc CLI compiler"                 ${PROJECT_IS_TOP_LEVEL})
option(SVSL_BUILD_TESTS "Build and register the unit/corpus tests"     ${PROJECT_IS_TOP_LEVEL})
option(SVSL_INSTALL     "Generate install rules"                       ${PROJECT_IS_TOP_LEVEL})
option(SVSL_BUILD_VIEW  "Build the svsl_view visual test app"          OFF)
option(SVSL_BUILD_ISA   "Build the svsl_isa GPU measurement tool"     OFF)
option(SVSL_ENABLE_WGSL "Compile the WGSL text backend (WebGPU)"       OFF)

# Directory-scoped: applies to every target here, leaks nothing into a parent.
set(CMAKE_C_STANDARD          11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS        OFF)

# Warning set: portability is a feature, so the bar is "clean on MSVC /W4, GCC,
# Clang, and mingw". Deliberately excluded, with reasons:
#   -Wswitch-enum      thousands of hits; the switches use `default:` on purpose
#   -Wcast-qual        the AST/const-expr paths cast const away by design
#   -Wfloat-equal      constant folding compares bit-exact values on purpose
#   -Wconversion       narrowing to int32_t/uint32_t is the house style
# The GNU/Clang list is calibrated to cover the MSVC /W4 classes that would
# otherwise only surface on Windows: -Wnested-externs is C4210, -Wshadow is
# C4456/C4457, -Wconditional-uninitialized is C4701.
#
# Sources are plain ASCII on purpose, so /utf-8 (and C4819) never come up: MSVC
# reads non-ASCII source in the local codepage, which mangles the text of the
# diagnostics we emit. Keep new comments and string literals ASCII-only.
if (MSVC)
	set(SVSL_WARNINGS /W4)
elseif (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
	set(SVSL_WARNINGS
		-Wall -Wextra
		-Wshadow                # MSVC C4456/C4457
		-Wnested-externs        # MSVC C4210
		-Wmissing-prototypes -Wmissing-declarations -Wstrict-prototypes -Wold-style-definition
		-Wredundant-decls
		-Wundef                 # #if on an undefined macro is always a typo
		-Wpointer-arith -Wwrite-strings -Wcast-align -Winit-self
		-Wvla                   # not portable to MSVC
		-Wformat-security       # the diag/emit helpers carry format attributes (util/attr.h)
		-Wnull-dereference
		-Wimplicit-fallthrough)
	if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
		list(APPEND SVSL_WARNINGS -Wlogical-op -Wduplicated-cond -Wduplicated-branches)
	else()
		list(APPEND SVSL_WARNINGS -Wconditional-uninitialized -Wcomma -Wloop-analysis
		                          -Wmissing-variable-declarations -Wstring-conversion)
	endif()
endif()

option(SVSL_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
if (SVSL_WARNINGS_AS_ERRORS)
	list(APPEND SVSL_WARNINGS $<IF:$<BOOL:${MSVC}>,/WX,-Werror>)
endif()

# --- libsvsl: zero-dependency core library --------------------------------

add_library(svsl STATIC
	vendor/smolv.c
	src/util/arena.c
	src/util/array.c
	src/util/str.c
	src/diag.c
	src/front/pp.c
	src/front/lexer.c
	src/front/parser.c
	src/front/ast.c
	src/back/spirv_builder.c
	src/back/emit_spirv.c
	src/tables/formats.c
	src/tables/keywords.c
	src/tables/semantics.c
	src/tables/spirv_opcodes.c
	src/tables/types_builtin.c
	src/sema/types.c
	src/sema/layout.c
	src/sema/const_eval.c
	src/sema/sema.c
	src/sema/check.c
	src/tables/intrinsics.c
	src/ir/ir_build.c
	src/ir/ir_dump.c
	src/ir/ir_operands.c
	src/ir/ir_edit.c
	src/ir/ir_cf.c
	src/ir/ir_const.c
	src/ir/ir_verify.c
	src/ir/passes/optimize.c
	src/ir/passes/fold.c
	src/ir/passes/combine.c
	src/ir/passes/cfg.c
	src/ir/passes/mem.c
	src/ir/passes/cse.c
	src/ir/passes/dce.c
	src/ir/passes/unroll.c
	src/ir/passes/sroa.c
	src/ir/usage.c
	src/out/reflect.c
	src/out/sks_write.c
	src/out/sks_read.c
	src/out/header_write.c
	src/api.c
)
add_library(svsl::svsl ALIAS svsl)
target_include_directories(svsl PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
)
target_include_directories(svsl PRIVATE src vendor)
target_compile_options    (svsl PRIVATE ${SVSL_WARNINGS})

# WGSL backend: pure C text emission, still zero-dependency - the option only
# gates code size. svsl_options_t.emit_wgsl errors without it.
if (SVSL_ENABLE_WGSL)
	target_sources            (svsl PRIVATE src/back/emit_wgsl.c)
	target_compile_definitions(svsl PRIVATE SVSL_HAS_WGSL)
endif()

# --- svslc: CLI compiler ----------------------------------------------------

if (SVSL_BUILD_CLI)
	add_executable(svslc cli/main.c)
	target_link_libraries (svslc PRIVATE svsl)
	target_compile_options(svslc PRIVATE ${SVSL_WARNINGS})
endif()

# --- install ------------------------------------------------------------------

if (SVSL_INSTALL)
	include(GNUInstallDirs)
	include(CMakePackageConfigHelpers)
	install(TARGETS svsl EXPORT svslTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
	install(DIRECTORY include/svsl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
	install(EXPORT svslTargets
		FILE        svslConfig.cmake
		NAMESPACE   svsl::
		DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svsl)
	write_basic_package_version_file(svslConfigVersion.cmake COMPATIBILITY SameMajorVersion)
	install(FILES ${CMAKE_CURRENT_BINARY_DIR}/svslConfigVersion.cmake
		DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/svsl)
	if (SVSL_BUILD_CLI)
		install(TARGETS svslc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
	endif()
endif()

# --- tests ------------------------------------------------------------------

if (SVSL_BUILD_TESTS)
	enable_testing()
	add_subdirectory(tests)
endif()

# --- svsl_view: dev-only visual test app ------------------------------------

if (SVSL_BUILD_VIEW)
	add_subdirectory(app)
endif()

# --- svsl_isa: dev-only GPU measurement tool --------------------------------
# Builds .spv stages into real driver pipelines (desktop or Android) and reports
# the driver's statistics, cold pipeline-creation time, or a timed dispatch - the
# hardware cost SPIR-V word count only loosely proxies. Links libvulkan directly
# (measurement tool, not part of the zero-dependency core). See app/isa.c.

if (SVSL_BUILD_ISA)
	find_package(Vulkan REQUIRED)
	add_executable(svsl_isa app/isa.c)
	target_link_libraries (svsl_isa PRIVATE Vulkan::Vulkan)
	target_compile_options(svsl_isa PRIVATE ${SVSL_WARNINGS})
endif()

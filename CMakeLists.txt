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
option(SVSL_BUILD_ISA   "Build the svsl_isa RDNA ISA measurement tool" OFF)
option(SVSL_ENABLE_WGSL "Compile the WGSL text backend (WebGPU)"       OFF)

# Directory-scoped: applies to every target here, leaks nothing into a parent.
set(CMAKE_C_STANDARD          11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS        OFF)

if (MSVC)
	set(SVSL_WARNINGS /W4)
elseif (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
	set(SVSL_WARNINGS -Wall -Wextra)
endif()

# --- libsvsl: zero-dependency core library --------------------------------

add_library(svsl STATIC
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
	src/ir/passes/optimize.c
	src/ir/passes/fold.c
	src/ir/passes/peephole.c
	src/ir/passes/mem.c
	src/ir/passes/cse.c
	src/ir/passes/dce.c
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

# WGSL backend: pure C text emission, still zero-dependency — the option only
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

# --- svsl_isa: dev-only RDNA ISA measurement tool ---------------------------
# Compiles a .spv stage into a real RADV pipeline and reports VGPR/SGPR/code
# size — the hardware cost SPIR-V word count only loosely proxies. Links
# libvulkan directly (measurement tool, not part of the zero-dependency core).

if (SVSL_BUILD_ISA)
	find_package(Vulkan REQUIRED)
	add_executable(svsl_isa app/isa.c)
	target_link_libraries(svsl_isa PRIVATE Vulkan::Vulkan)
endif()

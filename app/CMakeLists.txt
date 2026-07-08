# svsl_view — dev-only visual test app. Renders every corpus shader with both
# skshaderc (reference) and svslc, and pixel-diffs the results offscreen.
# Only this target may use FetchContent dependencies (SVSL_BUILD_VIEW=ON).

include(FetchContent)

set(SKR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Prefer the local checkout (the reference this project verifies against);
# override with -DFETCHCONTENT_SOURCE_DIR_SK_RENDERER=<path>.
if (NOT DEFINED FETCHCONTENT_SOURCE_DIR_SK_RENDERER AND EXISTS "$ENV{HOME}/SK/sk_renderer/CMakeLists.txt")
	set(FETCHCONTENT_SOURCE_DIR_SK_RENDERER "$ENV{HOME}/SK/sk_renderer" CACHE PATH "" FORCE)
	message(STATUS "svsl_view: using local sk_renderer at $ENV{HOME}/SK/sk_renderer")
endif()

FetchContent_Declare(
	sk_renderer
	GIT_REPOSITORY https://github.com/StereoKit/sk_renderer.git
	GIT_TAG        fe10447d # SKS v9 (feature mask, resource shape/format, per-stage wave_size)
)
FetchContent_MakeAvailable(sk_renderer)

add_executable(svsl_view
	src/main.c
	src/scene.c
	src/compare.c
)
target_link_libraries     (svsl_view PRIVATE svsl sk_renderer m)
target_include_directories(svsl_view PRIVATE
	../src
	${sk_renderer_SOURCE_DIR}/example/tools
)
target_compile_definitions(svsl_view PRIVATE
	SVSL_SKSHADERC_PATH="$<TARGET_FILE:skshaderc>"
	SVSL_SHADER_DIR="${CMAKE_SOURCE_DIR}/tests/shaders"
)
add_dependencies(svsl_view skshaderc)

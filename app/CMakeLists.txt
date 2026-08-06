# svsl_view — dev-only visual test app. Renders every corpus shader with both
# skshaderc (reference) and svslc, and pixel-diffs the results offscreen.
# Only this target may use FetchContent dependencies (SVSL_BUILD_VIEW=ON).

include(FetchContent)

set(SKR_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# skshaderc's optional embedded-SVSL backend would FetchContent this very
# project (target collision) — and the reference must stay glslang regardless.
set(SKSHADERC_ENABLE_SVSL OFF CACHE BOOL "" FORCE)

# sk_renderer for rendering test images, and for skshaderc, our reference compiler.
FetchContent_Declare(
	sk_renderer
	GIT_REPOSITORY https://github.com/StereoKit/sk_renderer.git
	GIT_TAG        v2026.8.6
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

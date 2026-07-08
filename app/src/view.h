// svsl_view internals: scene setup and the shader comparison engine.

#pragma once

#include <sk_renderer.h>
#include <float_math.h>

#include <stdbool.h>
#include <stdint.h>

#define VIEW_SIZE 256 // offscreen render target dimensions

// StereoKit's system buffer (stereokit.hlsli's stereokit_buffer, 1920 bytes)
#define SK_MAX_VIEWS 6
typedef struct sk_system_buffer_t {
	float4x4 sk_view       [SK_MAX_VIEWS];
	float4x4 sk_proj       [SK_MAX_VIEWS];
	float4x4 sk_proj_inv   [SK_MAX_VIEWS];
	float4x4 sk_viewproj   [SK_MAX_VIEWS];
	float4   sk_lighting_sh[7];
	float4   sk_camera_pos [SK_MAX_VIEWS];
	float4   sk_camera_dir [SK_MAX_VIEWS];
	float4   sk_fingertip  [2];
	float4   sk_cubemap_i;
	float4   sk_screen_size;
	float    sk_time;
	uint32_t sk_view_count;
	uint32_t sk_eye_offset;
	uint32_t _pad;
} sk_system_buffer_t;

// StereoKit's per-instance data (sk_inst element, 80 bytes)
typedef struct sk_inst_t {
	float4x4 world;
	float4   color;
} sk_inst_t;

typedef struct scene_t {
	skr_vert_type_t    vert_type;
	skr_mesh_t         sphere;
	skr_tex_t          checker;    // 2D pattern for diffuse-style textures
	skr_tex_t          cubemap;    // small colored cubemap
	skr_tex_t          tex_array;  // 1-layer 2D array (shadow maps)
	skr_tex_t          tex_3d;     // RGB gradient volume (GI voxels, SDFs)
	skr_tex_t          tex_3d_rw;  // storage-capable volume (RWTexture3D bindings)
	skr_tex_t          color_a, depth_a; // render targets, one pair per compiler
	skr_tex_t          color_b, depth_b;
	skr_tex_t          color_mv_a, depth_mv_a; // 2-layer targets for multiview passes
	skr_tex_t          color_mv_b, depth_mv_b;
	skr_tex_t          color_pf;   // postfx intermediate (SubpassInput source)
	skr_tex_t          color_ms, depth_ms; // 4x MSAA pair for shader-resolve passes
	skr_render_list_t  list;
	sk_system_buffer_t system;    // single view (sk_view_count = 1)
	sk_system_buffer_t system_mv; // stereo pair  (sk_view_count = 2)
	sk_inst_t          instance;
} scene_t;

bool scene_init(scene_t *scene);
void scene_destroy(scene_t *scene);

typedef struct compare_result_t {
	bool   ok;            // compiled + rendered + diffed under threshold
	bool   compiled_ref;  // skshaderc succeeded
	bool   compiled_ours; // svslc succeeded
	bool   rendered;
	bool   skipped;       // compute-only shader with no compute test config
	double avg_error;     // average per-channel error, 0..1
	double ref_variance;  // reference image variance (0 = blank render warning)
	char   message[256];
} compare_result_t;

// Compiles `shader_path` with both compilers, then renders both and diffs the
// pixels — or, for compute-only shaders with a test config, dispatches both on
// identical inputs and diffs the outputs. When out_dir is non-NULL, writes
// <name>_ref.ppm / <name>_svsl.ppm there. show_spirv_diff prints a spirv-dis
// unified diff of every stage (used by -file mode).
compare_result_t compare_shader(scene_t *scene, const char *shader_path,
                                const char *include_dir, const char *opt_out_dir,
                                bool show_spirv_diff);

// Scene setup: sphere, textures, render targets, and StereoKit-style system data.

#include "view.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static skr_tex_sampler_t linear_sampler(void) {
	return (skr_tex_sampler_t){
		.sample  = skr_tex_sample_linear,
		.address = skr_tex_address_wrap };
}

// StereoKit's fixed vertex layout: pos, norm, uv, color
static bool make_vert_type(skr_vert_type_t *out_type) {
	skr_vert_component_t components[4] = {
		{ .format = skr_vertex_fmt_f32, .count = 3, .semantic = skr_semantic_position },
		{ .format = skr_vertex_fmt_f32, .count = 3, .semantic = skr_semantic_normal },
		{ .format = skr_vertex_fmt_f32, .count = 2, .semantic = skr_semantic_texcoord },
		{ .format = skr_vertex_fmt_ui8_normalized, .count = 4, .semantic = skr_semantic_color },
	};
	return skr_vert_type_create(components, 4, out_type) == skr_err_success;
}

typedef struct vertex_t {
	float    pos[3];
	float    norm[3];
	float    uv[2];
	uint32_t color;
} vertex_t;

// UV sphere with poles on +Y/-Y
static bool make_sphere(const skr_vert_type_t *type, skr_mesh_t *out_mesh) {
	const int32_t segs = 32, rings = 24;
	const int32_t vert_count = (segs + 1) * (rings + 1);
	const int32_t ind_count  = segs * rings * 6;

	vertex_t *verts = malloc((size_t)vert_count * sizeof(vertex_t));
	uint32_t *inds  = malloc((size_t)ind_count * sizeof(uint32_t));
	int32_t   v = 0;
	for (int32_t r = 0; r <= rings; r++) {
		float phi = (float)r / rings * 3.14159265f;
		for (int32_t s2 = 0; s2 <= segs; s2++) {
			float theta = (float)s2 / segs * 6.2831853f;
			float x = sinf(phi) * cosf(theta), y = cosf(phi), z = sinf(phi) * sinf(theta);
			verts[v] = (vertex_t){
				.pos   = { x, y, z },
				.norm  = { x, y, z },
				.uv    = { (float)s2 / segs, (float)r / rings },
				.color = 0xFFFFFFFF };
			v++;
		}
	}
	int32_t i = 0;
	for (int32_t r = 0; r < rings; r++)
		for (int32_t s2 = 0; s2 < segs; s2++) {
			uint32_t a = (uint32_t)(r * (segs + 1) + s2);
			uint32_t b = a + (uint32_t)segs + 1;
			inds[i++] = a;     inds[i++] = a + 1; inds[i++] = b;
			inds[i++] = a + 1; inds[i++] = b + 1; inds[i++] = b;
		}

	bool ok = skr_mesh_create(type, skr_index_fmt_u32, verts, (uint32_t)vert_count,
	                          inds, (uint32_t)ind_count, out_mesh) == skr_err_success;
	free(verts);
	free(inds);
	return ok;
}

// 8x8 checker in a friendly palette
static bool make_checker(skr_tex_t *out_tex) {
	static uint32_t pixels[64 * 64];
	for (int32_t y = 0; y < 64; y++)
		for (int32_t x = 0; x < 64; x++) {
			bool on = ((x / 8) + (y / 8)) & 1;
			pixels[y * 64 + x] = on ? 0xFFE0A860 : 0xFF404048; // ABGR
		}
	skr_tex_data_t data = { .data = pixels, .mip_count = 1, .layer_count = 1, .row_pitch = 64 * 4 };
	return skr_tex_create(skr_tex_fmt_rgba32, skr_tex_flags_none, linear_sampler(),
	                      (skr_vec3i_t){ 64, 64, 1 }, 1, 1, &data, out_tex) == skr_err_success;
}

// tiny cubemap: one color per face so reflections show orientation
static bool make_cubemap(skr_tex_t *out_tex) {
	static uint32_t faces[6][16 * 16];
	const uint32_t colors[6] = { 0xFF5050D0, 0xFF3030A0, 0xFF60C060, 0xFF306030,
	                             0xFFD05050, 0xFFA03030 };
	for (int32_t f = 0; f < 6; f++)
		for (int32_t i = 0; i < 16 * 16; i++)
			faces[f][i] = colors[f];
	skr_tex_data_t data = { .data = faces, .mip_count = 1, .layer_count = 6 };
	return skr_tex_create(skr_tex_fmt_rgba32, skr_tex_flags_cubemap, linear_sampler(),
	                      (skr_vec3i_t){ 16, 16, 6 }, 1, 1, &data, out_tex) == skr_err_success;
}

// RGBA volume with one gradient per axis, so 3D sampling shows direction
static bool make_tex_3d(skr_tex_t *out_tex) {
	static uint32_t voxels[16 * 16 * 16];
	for (int32_t z = 0; z < 16; z++)
		for (int32_t y = 0; y < 16; y++)
			for (int32_t x = 0; x < 16; x++) {
				uint32_t r = (uint32_t)x * 17, g = (uint32_t)y * 17, b = (uint32_t)z * 17;
				voxels[(z * 16 + y) * 16 + x] = 0xFF000000 | (b << 16) | (g << 8) | r; // ABGR
			}
	skr_tex_data_t data = { .data = voxels, .mip_count = 1, .layer_count = 1 };
	return skr_tex_create(skr_tex_fmt_rgba32, skr_tex_flags_3d, linear_sampler(),
	                      (skr_vec3i_t){ 16, 16, 16 }, 1, 1, &data, out_tex) == skr_err_success;
}

// storage-capable volume for RWTexture3D bindings (written, never diffed)
static bool make_tex_3d_rw(skr_tex_t *out_tex) {
	return skr_tex_create(skr_tex_fmt_rgba32_linear, skr_tex_flags_3d | skr_tex_flags_compute,
	                      linear_sampler(),
	                      (skr_vec3i_t){ 16, 16, 16 }, 1, 1, NULL, out_tex) == skr_err_success;
}

static bool make_tex_array(skr_tex_t *out_tex) {
	static uint32_t pixels[16 * 16];
	for (int32_t i = 0; i < 16 * 16; i++) pixels[i] = 0xFFFFFFFF;
	skr_tex_data_t data = { .data = pixels, .mip_count = 1, .layer_count = 1, .row_pitch = 16 * 4 };
	return skr_tex_create(skr_tex_fmt_rgba32, skr_tex_flags_array, linear_sampler(),
	                      (skr_vec3i_t){ 16, 16, 1 }, 1, 1, &data, out_tex) == skr_err_success;
}

static bool make_target(skr_tex_t *out_color, skr_tex_t *out_depth, int32_t layers) {
	skr_tex_flags_ array = layers > 1 ? skr_tex_flags_array : skr_tex_flags_none;
	if (skr_tex_create(skr_tex_fmt_rgba32,
	                   skr_tex_flags_writeable | skr_tex_flags_readable | array,
	                   linear_sampler(),
	                   (skr_vec3i_t){ VIEW_SIZE, VIEW_SIZE, layers }, 1, 1, NULL, out_color) != skr_err_success)
		return false;
	return skr_tex_create(skr_tex_fmt_depth32, skr_tex_flags_writeable | array, linear_sampler(),
	                      (skr_vec3i_t){ VIEW_SIZE, VIEW_SIZE, layers }, 1, 1, NULL, out_depth) == skr_err_success;
}

bool scene_init(scene_t *scene) {
	memset(scene, 0, sizeof(*scene));

	#define STEP(x) if (!(x)) { fprintf(stderr, "scene_init failed: %s\n", #x); return false; }
	STEP(make_vert_type(&scene->vert_type));
	STEP(make_sphere(&scene->vert_type, &scene->sphere));
	STEP(make_checker(&scene->checker));
	STEP(make_cubemap(&scene->cubemap));
	STEP(make_tex_array(&scene->tex_array));
	STEP(make_tex_3d(&scene->tex_3d));
	STEP(make_tex_3d_rw(&scene->tex_3d_rw));
	STEP(make_target(&scene->color_a, &scene->depth_a, 1));
	STEP(make_target(&scene->color_b, &scene->depth_b, 1));
	STEP(make_target(&scene->color_mv_a, &scene->depth_mv_a, 2));
	STEP(make_target(&scene->color_mv_b, &scene->depth_mv_b, 2));
	// geometry color that a postfx subpass reads as its input attachment
	STEP(skr_tex_create(skr_tex_fmt_rgba32,
	                    skr_tex_flags_writeable | skr_tex_flags_input_attachment,
	                    linear_sampler(),
	                    (skr_vec3i_t){ VIEW_SIZE, VIEW_SIZE, 1 }, 1, 1, NULL,
	                    &scene->color_pf) == skr_err_success);
	// 4x MSAA pair for manual shader-resolve passes (SubpassInputMS reads the
	// samples in-tile; input_attachment is what makes that legal)
	STEP(skr_tex_create(skr_tex_fmt_rgba32,
	                    skr_tex_flags_writeable | skr_tex_flags_input_attachment,
	                    linear_sampler(),
	                    (skr_vec3i_t){ VIEW_SIZE, VIEW_SIZE, 1 }, 4, 1, NULL,
	                    &scene->color_ms) == skr_err_success);
	STEP(skr_tex_create(skr_tex_fmt_depth32,
	                    skr_tex_flags_writeable | skr_tex_flags_input_attachment,
	                    linear_sampler(),
	                    (skr_vec3i_t){ VIEW_SIZE, VIEW_SIZE, 1 }, 4, 1, NULL,
	                    &scene->depth_ms) == skr_err_success);
	STEP(skr_render_list_create(&scene->list) == skr_err_success);
	#undef STEP

	// camera: slightly off-axis so lighting/normals show gradients
	float3   eye  = { 0.9f, 0.7f, -2.0f };
	float3   at   = { 0, 0, 0 };
	float4x4 view = float4x4_lookat(eye, at, (float3){ 0, 1, 0 });
	float4x4 proj = float4x4_perspective(1.0f, 1.0f, 0.1f, 50.0f);
	float4x4 vp   = float4x4_mul(proj, view); // proj·view, like the sk_renderer examples

	sk_system_buffer_t *sys = &scene->system;
	for (int32_t v = 0; v < SK_MAX_VIEWS; v++) {
		sys->sk_view[v]     = view;
		sys->sk_proj[v]     = proj;
		sys->sk_proj_inv[v] = float4x4_identity(); // not exercised by the corpus visuals
		sys->sk_viewproj[v] = vp;
		sys->sk_camera_pos[v] = (float4){ eye.x, eye.y, eye.z, 1 };
		float3 dir = float3_norm(float3_sub(at, eye));
		sys->sk_camera_dir[v] = (float4){ dir.x, dir.y, dir.z, 0 };
	}
	// plausible L2 spherical harmonics: warm key light from the upper left
	sys->sk_lighting_sh[0] = (float4){ 0.55f, 0.52f, 0.50f, 1 };
	sys->sk_lighting_sh[1] = (float4){ 0.20f, 0.18f, 0.12f, 1 };
	sys->sk_lighting_sh[2] = (float4){ 0.12f, 0.12f, 0.15f, 1 };
	sys->sk_lighting_sh[3] = (float4){ -0.10f, -0.08f, -0.06f, 1 };
	for (int32_t i = 4; i < 7; i++)
		sys->sk_lighting_sh[i] = (float4){ 0.03f, 0.02f, 0.02f, 1 };
	sys->sk_fingertip[0] = (float4){ 0.4f, 0.2f, -0.5f, 1 };
	sys->sk_fingertip[1] = (float4){ -0.4f, 0.1f, -0.5f, 1 };
	sys->sk_cubemap_i    = (float4){ 16, 16, 1, 0 };
	sys->sk_screen_size  = (float4){ VIEW_SIZE, VIEW_SIZE, 1.0f / VIEW_SIZE, 1.0f / VIEW_SIZE };
	sys->sk_time         = 1.25f; // fixed, so animated shaders stay deterministic
	sys->sk_view_count   = 1;
	sys->sk_eye_offset   = 0;

	// multiview variant: a stereo pair. View 0 matches the single-view camera,
	// view 1 is offset like a right eye so SV_ViewID differences show in pixels.
	scene->system_mv = *sys;
	sk_system_buffer_t *mv    = &scene->system_mv;
	float3              eye_r = { eye.x + 0.12f, eye.y, eye.z };
	float4x4            view_r = float4x4_lookat(eye_r, at, (float3){ 0, 1, 0 });
	mv->sk_view[1]       = view_r;
	mv->sk_proj[1]       = proj;
	mv->sk_proj_inv[1]   = float4x4_identity();
	mv->sk_viewproj[1]   = float4x4_mul(proj, view_r);
	mv->sk_camera_pos[1] = (float4){ eye_r.x, eye_r.y, eye_r.z, 1 };
	float3 dir_r = float3_norm(float3_sub(at, eye_r));
	mv->sk_camera_dir[1] = (float4){ dir_r.x, dir_r.y, dir_r.z, 0 };
	mv->sk_view_count    = 2;

	scene->instance.world = float4x4_identity();
	scene->instance.color = (float4){ 1, 1, 1, 1 };
	return true;
}

void scene_destroy(scene_t *scene) {
	skr_render_list_destroy(&scene->list);
	skr_mesh_destroy(&scene->sphere);
	skr_tex_destroy(&scene->checker);
	skr_tex_destroy(&scene->cubemap);
	skr_tex_destroy(&scene->tex_array);
	skr_tex_destroy(&scene->tex_3d);
	skr_tex_destroy(&scene->tex_3d_rw);
	skr_tex_destroy(&scene->color_a);
	skr_tex_destroy(&scene->depth_a);
	skr_tex_destroy(&scene->color_b);
	skr_tex_destroy(&scene->depth_b);
	skr_tex_destroy(&scene->color_mv_a);
	skr_tex_destroy(&scene->depth_mv_a);
	skr_tex_destroy(&scene->color_mv_b);
	skr_tex_destroy(&scene->depth_mv_b);
	skr_tex_destroy(&scene->color_pf);
	skr_tex_destroy(&scene->color_ms);
	skr_tex_destroy(&scene->depth_ms);
}

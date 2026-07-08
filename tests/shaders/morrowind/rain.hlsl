#include <stereokit.hlsli>

///////////////////////////////////////////

//--rain_head_pos       = {0, 0, 0, 0}
//--rain_params         = {0, 20, 9, 0.15}
//--rain_color          = {0.7, 0.75, 0.85, 0.3}
//--rain_settings       = {1, 1, 0, 0}
//--rain_occl_transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
float4   rain_head_pos;       // xyz = head world position
float4   rain_params;         // x = time, y = cube_size, z = fall_speed, w = drop_length
float4   rain_color;          // xyz = tint from sky, w = opacity
float4   rain_settings;       // x = intensity (0-1), y = drop_size multiplier
float4x4 rain_occl_transform; // world -> occlusion map clip space

//--rain_occl_map = white
Texture2D              rain_occl_map;
SamplerComparisonState rain_occl_map_s;

///////////////////////////////////////////

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;  // x = vertex index (0-3), y = per-droplet variation
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos       : SV_POSITION;
	float3 world_pos : TEXCOORD0;
	float  alpha     : TEXCOORD1;
};

///////////////////////////////////////////

// Wrap x into [0, size) — always positive, unlike fmod
float wrap(float x, float size) {
	return x - floor(x / size) * size;
}

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;

	// Discard droplets above intensity threshold
	float intensity = rain_settings.x;
	float drop_size = rain_settings.y;
	if (input.uv.y > intensity) {
		o.pos = 0.0 / 0.0;
		return o;
	}

	float  time       = rain_params.x;
	float  cube_size  = rain_params.y;
	float  fall_speed = rain_params.z;
	float  drop_len   = rain_params.w * 2 * drop_size;
	float  half_cube  = cube_size * 0.5;
	float3 head       = rain_head_pos.xyz;

	// Wrap model positions within the rain cube.
	// Head XZ offset in unit-cube space keeps droplets world-fixed
	// while the cube travels with the head via the world matrix.
	float3 head_offset = head / cube_size;

	float3 local;
	local.x = wrap(input.pos.x - head_offset.x, 1.0) - 0.5;
	local.z = wrap(input.pos.z - head_offset.z, 1.0) - 0.5;
	local.y = wrap(input.pos.y - head_offset.y - fall_speed * time / cube_size, 1.0) - 0.5;

	float3 model_pos = local * cube_size;

	// Expand 4 vertices into a diamond shape:
	//   0 = top tip (skinny), 1 = wide left, 2 = wide right, 3 = bottom tip (blunt)
	// Triangles via index buffer: (0,1,2) upper, (2,1,3) lower.
	// Wide point sits 70% down — skinny top, blunt bottom.
	// Streak length scaled by per-droplet speed variation for
	// frame-to-frame continuity.
	float  width    = 0.002 * drop_size;
	int    vtx      = (int)input.uv.x;
	float  y_offset = (vtx == 0) ?  drop_len * 0.5
	                : (vtx == 3) ? -drop_len * 0.5
	                :              -drop_len * 0.2;
	float  x_offset = (vtx == 1) ? -width
	                : (vtx == 2) ?  width
	                :               0;

	// Camera-facing billboard (horizontal axis only — rain stays vertical)
	// In model space, camera is at the origin
	float3 to_cam = -model_pos;
	to_cam.y      = 0;
	float  len_xz = length(to_cam);
	float3 right  = (len_xz > 0.001)
		? float3(-to_cam.z / len_xz, 0, to_cam.x / len_xz)
		: float3(1, 0, 0);

	model_pos    += right * x_offset;
	model_pos.y  += y_offset;

	// Standard StereoKit transform: model -> world -> clip
	float4 world = mul(float4(model_pos, 1), sk_inst[ids.inst].world);
	o.world_pos  = world.xyz;
	o.pos        = mul(world, sk_viewproj[ids.view]);

	// Fade at cube edges to prevent visible wrap popping
	float dist_y = abs(local.y * 2);
	float dist_h = length(float2(local.x, local.z)) * 2;
	float edge   = max(dist_y, dist_h);
	o.alpha      = saturate(1.0 - smoothstep(0.8, 1.0, edge));

	return o;
}

///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	// Occlusion check: transform world position into occlusion map space
	float4 occl_clip = mul(float4(input.world_pos, 1), rain_occl_transform);
	float2 occl_uv   = occl_clip.xy;
	float  occl_z    = occl_clip.z / occl_clip.w;

#ifdef SK_OPENGL
	occl_uv = occl_uv * 0.5 + 0.5;
#else
	occl_uv = occl_uv * float2(0.5, -0.5) + 0.5;
#endif

	// If within occlusion map bounds, test depth
	if (occl_uv.x >= 0 && occl_uv.x <= 1 && occl_uv.y >= 0 && occl_uv.y <= 1) {
		float visible = rain_occl_map.SampleCmpLevelZero(rain_occl_map_s, occl_uv, occl_z);
		if (visible < 0.5)
			discard;
	}

	return float4(rain_color.rgb, input.alpha * rain_color.a);
}

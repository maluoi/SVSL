#include <stereokit.hlsli>

///////////////////////////////////////////

//--cloud_scroll    = {0, 0, 0, 0}
//--cloud_coverage  = {0.5, 0, 0, 0}
//--cloud_sun_dir   = {0.5, 0.7, 0.5, 0}
//--cloud_color_lit = {1, 1, 1, 0}
//--cloud_color_shadow = {0.2, 0.2, 0.3, 0}
float4 cloud_scroll;       // xy = UV scroll offset
float4 cloud_coverage;     // x = coverage (0 = clear, 1 = overcast)
float4 cloud_sun_dir;      // xyz = sun direction
float4 cloud_color_lit;    // xyz = sky color sampled toward sun
float4 cloud_color_shadow; // xyz = sky color sampled downward

//--cloud_map = white
Texture2D    cloud_map;
SamplerState cloud_map_s;

///////////////////////////////////////////

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos       : SV_POSITION;
	float4 uv        : TEXCOORD0;
	float  edge_fade : TEXCOORD2;
};

///////////////////////////////////////////

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;

	// Skybox trick: take the camera's view matrix with translation zeroed,
	// treat input.pos as already in camera space, then project. The disc
	// visually follows the camera and stays "infinite distance" via
	// `o.pos.z = o.pos.w`.
	float4x4 rot_view = sk_view[ids.view];
	rot_view._m30 = 0;
	rot_view._m31 = 0;
	rot_view._m32 = 0;
	o.pos   = mul(mul(float4(input.pos.xyz, 1), rot_view), sk_proj[ids.view]);
	o.pos.z = o.pos.w; // pin to far plane so the disc never far-clips

	// UVs derived from world XZ (not mesh-local XZ) so the cloud pattern stays
	// anchored as the player walks — without this the disc's per-frame
	// Matrix.T(camera.xz) translation drags the UVs along with it and clouds
	// appear glued to the camera. The shadow caster does the same so silhouettes
	// match.
	float4 world = mul(input.pos, sk_inst[ids.inst].world);
	o.uv.xy =  world.xz * 0.0015 + 0.5 + cloud_scroll.xy * 0.0015;
	o.uv.zw = (world.xz * 0.00075 + cloud_scroll.xy) * 3.0 + 0.5;
	o.edge_fade = input.uv.x; // 0 at center, 1 at rim

	return o;
}

///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	// Two-scale sampling: large shapes + smaller detail
	float large  = cloud_map.Sample(cloud_map_s, input.uv.xy).r;
	float detail = cloud_map.Sample(cloud_map_s, input.uv.zw).r;

	float thickness = large * 0.7 + detail * 0.3;
	float cloud = (thickness - (1-cloud_coverage.x));

	if (cloud*12 < 0.01)
		discard;

	// Edge fade at disc rim
	float fade = 1.0 - smoothstep(0.7, 1.0, input.edge_fade);

	// Thin clouds lit from above, thick clouds show shadow underneath
	float3 color = lerp(cloud_color_lit.rgb, cloud_color_shadow.rgb, 1- pow(1-cloud, 4));

	return float4(color, min(fade,saturate(cloud * 12)));
}

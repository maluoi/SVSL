#include <stereokit.hlsli>

//--name = app/gi_voxel_debug
// Debug viz for the voxelized GI volume. One cube per voxel cell, sampling
// voxel_tex at the cube center. Empty cells (alpha < 0.5) are discarded so
// only filled voxels show.

cbuffer GIBuffer : register(b4) {
	float3 gi_volume_min;
	float  gi_voxel_res;
	float3 gi_volume_inv;
	float  gi_intensity;
	float4 _gi_pad0;
	// .x = voxel storage range scale; voxel.rgb × this = real radiance.
	float4 voxel_range_pad;
	float4 _gi_pad2;
};

Texture3D<float4> voxel_tex   : register(t0);
SamplerState      voxel_tex_s : register(s0);

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos                          : SV_POSITION;
	float3 normal                       : NORMAL0;
	nointerpolation float3 cube_center  : TEXCOORD0;
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;
	float4 world  = mul(input.pos, sk_inst[ids.inst].world);
	o.pos         = mul(world, sk_viewproj[ids.view]);
	o.normal      = normalize(mul(input.norm, (float3x3)sk_inst[ids.inst].world));
	// World-space center of this cube = its world transform applied to the origin.
	o.cube_center = mul(float4(0, 0, 0, 1), sk_inst[ids.inst].world).xyz;
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	// Volume is non-cubic, so both the bounds discard and the UVW lookup are
	// per-axis. The voxel texture is sized to VoxelRes_* per axis (256×128×256),
	// so a single world position maps to a different UVW per axis — dividing
	// by any one axis's extent would compress the others into the wrong slot
	// range (Y-stretched cubes were the visible symptom).
	float3 volume_extent = 1.0 / gi_volume_inv;
	float3 to_min        = input.cube_center - gi_volume_min;
	if (any(to_min < 0) || any(to_min > volume_extent)) discard;
	float3 uvw = input.cube_center / volume_extent;

	float4 v = voxel_tex.SampleLevel(voxel_tex_s, uvw, 0);
	if (v.a < 0.5) discard;

	// Voxels store lit / voxel_range; unscale to display the real radiance.
	float3 rgb = v.rgb * voxel_range_pad.x;

	// Cheap NdotL on a fixed light so cube faces have some shading.
	float ndotl = saturate(dot(input.normal, normalize(float3(0.5, 1.0, 0.3)))) * 0.7 + 0.3;
	return float4(rgb * ndotl, 1);
}

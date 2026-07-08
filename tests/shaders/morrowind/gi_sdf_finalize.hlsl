//--name = app/gi_sdf_finalize
// Convert JFA's seed-coordinate grid into an R8 distance field. Each cell's
// SDF value is the Euclidean *slot-local* distance to its nearest in-window
// seed — which equals the true world distance, since both the seed pass
// (gi_sdf_seed.hlsl) and the JFA pass (gi_sdf_jfa.hlsl) operate in slot-
// local coordinates and reject neighbour reads that would cross the world-
// discontinuity seam in the toroidal slot mapping.
//
// Consumers use this as the skip-distance for a hybrid sphere-trace / DDA
// march: when SDF is large, jump the full distance through empty space;
// when SDF drops below a near-geometry threshold (~2 voxels), switch to
// DDA — step exactly to the next voxel boundary along the ray and check
// voxel_tex.a. That gives precise visit-every-voxel-on-the-ray-path
// coverage near surfaces (no thin-feature skips) while still amortizing
// long empty stretches into single jumps.
//
// Encoding: distance / SDF_MAX_VOXELS → R8 unorm. floor-quantize on write
// so stored ≤ true; sphere tracing the stored value never overshoots.
//
// SDF_MAX_VOXELS = 128 (half the longest axis). R8 step is 128/255 ≈ 0.5
// voxels per stored value — enough to distinguish cardinal-1 / diagonal-√2
// / corner-√3, which was flattening to the same R8 value at MAX=256 and
// driving the trilinear-interpolated level set to a caltrop shape near
// thin features. Anything past 128 voxels saturates to "very far" — fine
// for sphere tracing, since one 128-voxel jump crosses half the volume
// and the bounds check picks up the rest.

cbuffer GIBuffer : register(b4) {
	float3 gi_volume_min;
	float  gi_voxel_res;
	float3 gi_volume_inv;
	float  gi_intensity;
	float4 gi_prev_origin_probe_res;
	float4 gi_voxel_range_pad;
	float4 _gi_pad2;
};

Texture3D<float4>                            jfa_in  : register(t0);
[[vk::image_format("r8")]] RWTexture3D<unorm float> sdf_out : register(u0);

// Per-axis voxel resolution. Must match gi.hlsli + gi_voxel_to_sh.hlsl.
#define VOXEL_RES_X     256
#define VOXEL_RES_Y     128
#define VOXEL_RES_Z     256
static const int3 VOXEL_RES      = int3(VOXEL_RES_X,     VOXEL_RES_Y,     VOXEL_RES_Z);
static const int3 VOXEL_RES_MASK = int3(VOXEL_RES_X - 1, VOXEL_RES_Y - 1, VOXEL_RES_Z - 1);
#define SDF_MAX_VOXELS 128.0

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	// Volume can be non-cubic but voxel spacing is uniform — derive voxel_size
	// from any axis (X as canonical).
	float voxel_size = (1.0 / gi_volume_inv.x) / (float)VOXEL_RES_X;
	int3  origin_off = int3(floor(gi_volume_min / voxel_size)) & VOXEL_RES_MASK;
	int3  curr_local = (int3(id) - origin_off + VOXEL_RES) & VOXEL_RES_MASK;

	float4 s = jfa_in.Load(int4(id, 0));
	float dist_voxels;
	if (s.a < 0.5) {
		// Cell never received a seed — no occluder reachable inside the
		// active window. Report max so the consumer treats this cell as
		// "open space, jump as far as you can".
		dist_voxels = SDF_MAX_VOXELS;
	} else {
		int3 seed_local = int3(round(s.rgb * 255.0));
		// Euclidean — no toroidal min(). Both curr_local and seed_local
		// live in [0, VOXEL_RES) under the active window's origin and were
		// produced by a JFA whose neighbour reads never crossed the seam, so
		// `curr_local - seed_local` is the true world-voxel displacement.
		int3 diff   = curr_local - seed_local;
		dist_voxels = length(float3(diff));
	}
	// Floor-quantize so stored ≤ true.
	sdf_out[id] = floor(saturate(dist_voxels / SDF_MAX_VOXELS) * 255.0) / 255.0;
}

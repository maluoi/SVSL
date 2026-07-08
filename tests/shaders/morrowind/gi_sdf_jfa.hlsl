//--name = app/gi_sdf_jfa
// One JFA (Jump Flood Algorithm) pass over the voxel volume in *slot-local*
// coordinates (the cell's offset from the active window's low corner, in
// [0, VOXEL_RES)). At each cell we read 27 neighbours at offset = step_size,
// adopt the seed minimising Euclidean slot-local distance to *this* cell.
// After log2(VOXEL_RES) = 7 passes (steps 64, 32, 16, 8, 4, 2, 1) every cell
// in the active window holds its true world-nearest seed.
//
// One Compute instance per pass with step_size baked at Init (sk_renderer
// replays queued dispatches with each Compute's CURRENT params; sharing one
// Compute would collapse all passes to the final step_size). Each Compute
// also has its ping/pong textures bound — pong is the read source on
// alternate passes so the chain is ping→pong→ping→pong→ping→pong→ping→pong.
//
// CRITICAL: neighbour reads are *bounded* in slot-local space. The voxel
// volume uses toroidal slot addressing (slot = world & VOXEL_RES_MASK), so
// slot 63 and slot 64 are adjacent in slot space but, whenever the world
// origin places its modular seam between them, represent world cells 128
// voxels apart. A simple toroidal-slot JFA propagates seeds across that
// seam, then reports a 1-voxel slot-distance for cells that are really 127
// voxels apart in world — the consumer sphere-tracer either over-eagerly
// stops (false near-geometry) or under-jumps and burns iterations.
//
// Working in slot-local fixes both halves of that bug:
//   - The seed payload is the seed cell's slot-local position (gi_sdf_seed),
//     so any two seeds in the active window have a meaningful Euclidean
//     distance (no toroidal min() trick needed).
//   - Neighbours whose slot-local lies outside [0, VOXEL_RES) are skipped,
//     so seeds never propagate across the world-discontinuity seam.
// Inside the active window slot-local Euclidean distance is exactly the
// true world distance — the slot mapping preserves contiguity except across
// the seam, and we never read across it.

cbuffer GIBuffer : register(b4) {
	float3 gi_volume_min;
	float  gi_voxel_res;
	float3 gi_volume_inv;
	float  gi_intensity;
	float4 gi_prev_origin_probe_res;
	float4 gi_voxel_range_pad;
	float4 _gi_pad2;
};

Texture3D<float4>                                jfa_in  : register(t0);
[[vk::image_format("rgba8")]] RWTexture3D<unorm float4> jfa_out : register(u0);

uint step_size;

// Per-axis voxel resolution. Must match gi.hlsli + gi_voxel_to_sh.hlsl.
#define VOXEL_RES_X     256
#define VOXEL_RES_Y     128
#define VOXEL_RES_Z     256
static const int3 VOXEL_RES      = int3(VOXEL_RES_X,     VOXEL_RES_Y,     VOXEL_RES_Z);
static const int3 VOXEL_RES_MASK = int3(VOXEL_RES_X - 1, VOXEL_RES_Y - 1, VOXEL_RES_Z - 1);

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	// Volume can be non-cubic but voxel spacing is uniform — derive voxel_size
	// from any axis (X as canonical).
	float voxel_size = (1.0 / gi_volume_inv.x) / (float)VOXEL_RES_X;
	int3  origin_off = int3(floor(gi_volume_min / voxel_size)) & VOXEL_RES_MASK;
	int3  curr_local = (int3(id) - origin_off + VOXEL_RES) & VOXEL_RES_MASK;

	float4 best     = float4(0, 0, 0, 0);
	float  best_d2  = 1e30;

	[unroll] for (int dz = -1; dz <= 1; dz++)
	[unroll] for (int dy = -1; dy <= 1; dy++)
	[unroll] for (int dx = -1; dx <= 1; dx++) {
		int3 n_local = curr_local + int3(dx, dy, dz) * (int)step_size;
		// Bounded read: skip neighbours that lie outside the active window.
		// This is what stops seed propagation across the world-discontinuity
		// seam — the slot at curr's slot + offset only represents a real
		// in-window neighbour if (curr_local + offset*step) ∈ [0, VOXEL_RES).
		if (any(n_local < 0) || any(n_local >= VOXEL_RES)) continue;
		// Storage lookup is still toroidal — the neighbour's slot is
		// `slot(curr) + offset` mod VOXEL_RES (equivalent to
		// (n_local + origin_off) & MASK). The +VOXEL_RES bias keeps the
		// mask correct for negative intermediate values.
		int3   n_slot = (int3(id) + int3(dx, dy, dz) * (int)step_size + VOXEL_RES) & VOXEL_RES_MASK;
		float4 s      = jfa_in.Load(int4(n_slot, 0));
		if (s.a < 0.5) continue;
		int3  seed_local = int3(round(s.rgb * 255.0));
		// Pure Euclidean — no toroidal min() — because both curr_local and
		// seed_local live in [0, VOXEL_RES) under the active window's origin,
		// and the bounded read above guarantees the seed's seed_local makes
		// sense in the current frame's origin.
		int3  diff = curr_local - seed_local;
		float d2   = dot(float3(diff), float3(diff));
		if (d2 < best_d2) {
			best    = s;
			best_d2 = d2;
		}
	}

	jfa_out[id] = best;
}

#include <stereokit.hlsli>

//--name = app/gi_sdf_debug
// Debug viz of the SDF + voxel volume. The C# side draws a single cube at the
// volume's world AABB (centered at volume_min + half_extent, scale = volume_size)
// with FaceCull.Front + DepthTest.Always + QueueOffset late, so we always
// render the cube's back-side from inside or behind any scene geometry.
//
// The fragment runs a sphere-tracing raymarch through the SDF starting at the
// camera (or the cube's near intersection if the camera is outside) toward the
// backface position; on hit, the voxel color is the output. This is the same
// march the SH compute does, just spotlighted: when the viz looks right, the
// SH compute is reading the right data.
//
// Renders late (QueueOffset > 0) so it overlays the scene wherever the volume
// covers — handy for spotting holes in the voxelization or chunks of SDF that
// disagree with the visible geometry.

cbuffer GIBuffer : register(b4) {
	float3 gi_volume_min;
	float  gi_voxel_res;
	float3 gi_volume_inv;
	float  gi_intensity;
	float4 gi_prev_origin_probe_res;
	float4 gi_voxel_range_pad;
	float4 _gi_pad2;
};

Texture3D<float4> voxel_tex   : register(t0);
Texture3D<float>  sdf_tex     : register(t1);
SamplerState      sdf_tex_s   : register(s1);

// Per-axis voxel resolution. Must match gi.hlsli + gi_voxel_to_sh.hlsl.
#define VOXEL_RES_X       256
#define VOXEL_RES_Y       128
#define VOXEL_RES_Z       256
static const int3 VOXEL_RES      = int3(VOXEL_RES_X,     VOXEL_RES_Y,     VOXEL_RES_Z);
static const int3 VOXEL_RES_MASK = int3(VOXEL_RES_X - 1, VOXEL_RES_Y - 1, VOXEL_RES_Z - 1);
// SDF_MAX_VOXELS = 128 (half the longest axis). Matches gi_sdf_finalize.hlsl's
// R8 scale — see the docs there for why this beats MAX=256.
#define SDF_MAX_VOXELS    128.0
// Sphere-trace stop epsilon, in voxel units. The rendered surface is the
// SDF level set at this distance — the geometry "inflated" by this much.
// Smaller = tighter to the voxel grid (visible cubes). Larger = blobbier.
// Must be > 0; at 0 the trace can never satisfy `sdf < 0` and never hits.
#define HIT_THICKNESS       0.3
// When sphere-trace stops inside the hit thickness, the current slot is
// often a "near" cell rather than the actual filled one — the seed is up
// to HIT_THICKNESS voxels away in some direction. We first jump forward by
// HIT_THICKNESS along the ray so the slot lookup lands past the inflation
// shell into the filled cell. For axis-aligned surfaces the level set sits
// exactly on a slot boundary — without this pre-step `floor()` flips to
// the empty slot on the wrong side at half the hits and the picker reads
// air.
//
// RECOVERY_MODE picks how the hit branch resolves the surface color:
//   RECOVERY_NONE      — single voxel.Load at the forward-step position; if
//                        the slot is empty, discard. Cheapest path; reveals
//                        exactly where the inflation-shell jump alone is
//                        sufficient and where it leaves gaps.
//   RECOVERY_DIR_STEP  — one extra 1-voxel jump along the ray direction,
//                        then a single voxel.Load. Same cost as GRAD_STEP
//                        but no gradient computation; great on grazing rays
//                        where the next slot along the path is the filled
//                        one. Direction-blind, so worse than GRAD_STEP at
//                        head-on hits where the gradient points truly inward.
//   RECOVERY_DDA       — forward-step then walk along the ray direction one
//                        voxel boundary at a time (up to COLOR_RECOVERY
//                        hops) until a filled cell is hit. Robust to
//                        grazing rays; samples every voxel along the path.
//   RECOVERY_GRAD_STEP — forward-step then sample once at one voxel along
//                        -∇SDF (the shortest direction to the nearest
//                        seed). One voxel.Load; clean on axis-aligned
//                        surfaces but can wobble near corners or thin
//                        features where the gradient is noisy.
//   RECOVERY_GRAD_VIZ  — diagnostic: bypass voxel lookups entirely. Map
//                        the gradient normal directly to RGB (each axis
//                        rescaled from [-1, 1] to [0, 1]) so colour =
//                        surface-normal orientation. Flat surfaces show
//                        as solid tones (e.g. up-facing floor → (0.5, 1,
//                        0.5)); noisy gradient zones show as dappled patches.
#define RECOVERY_NONE       0
#define RECOVERY_DIR_STEP   1
#define RECOVERY_DDA        2
#define RECOVERY_GRAD_STEP  3
#define RECOVERY_GRAD_VIZ   4
#define RECOVERY_MODE       RECOVERY_DDA
#define COLOR_RECOVERY      3

// Continuous SDF sample at a world-voxel-coord position. The texture uses
// TexAddress.Wrap + TexSample.Linear, so hardware filtering blends across
// the toroidal slot seam — texel 127 and texel 0 are adjacent under both
// the slot mapping (world & MASK) and the wrap-addressed texture, so the
// blend mixes the right neighbouring world cells.
float sample_sdf_smooth(float3 ray_pos_w) {
	// UVW = ray_pos_w / RES per-axis — texel centres at integer + 0.5 in voxel
	// coords correspond to UVW = (integer + 0.5) / RES, which is the natural
	// texel-centre sampling position under HLSL's filtering convention. Volume
	// is non-cubic so each axis divides by its own resolution.
	float3 uvw = ray_pos_w / float3(VOXEL_RES);
	return sdf_tex.SampleLevel(sdf_tex_s, uvw, 0) * SDF_MAX_VOXELS;
}

// SDF gradient. Both variants return a vector pointing away from the nearest
// geometry (outward normal of the level set); negating it gives the
// direction to walk *toward* the geometry — the shortest path through the
// inflation shell to the actual filled voxel. The recovery + viz paths
// below pick which one to call.
//
// 4-tap tetrahedral (iq). Cheaper — half the SDF samples — and a single
// finite-difference plane per axis. Slightly noisier on coarse SDFs because
// the four corner samples don't fully cancel under linear filtering.
float3 sdf_gradient_4(float3 ray_pos_w) {
	const float  h = 1.0;
	const float2 k = float2(1, -1);
	return k.xyy * sample_sdf_smooth(ray_pos_w + k.xyy * h) +
	       k.yyx * sample_sdf_smooth(ray_pos_w + k.yyx * h) +
	       k.yxy * sample_sdf_smooth(ray_pos_w + k.yxy * h) +
	       k.xxx * sample_sdf_smooth(ray_pos_w + k.xxx * h);
}

// 6-tap central differences. Classical per-axis ∂f/∂axis = f(p + h·axis) -
// f(p - h·axis). Twice the samples of the tetrahedral version but each
// axis derivative is computed from a clean ±h pair, so the result is less
// affected by trilinear cross-talk on coarsely-quantised SDFs.
float3 sdf_gradient_6(float3 ray_pos_w) {
	const float  h = 1.0;
	const float2 k = float2(h, 0);
	return float3(sample_sdf_smooth(ray_pos_w + k.xyy) - sample_sdf_smooth(ray_pos_w - k.xyy),
	              sample_sdf_smooth(ray_pos_w + k.yxy) - sample_sdf_smooth(ray_pos_w - k.yxy),
	              sample_sdf_smooth(ray_pos_w + k.yyx) - sample_sdf_smooth(ray_pos_w - k.yyx));
}

// Currently active gradient — alias so the recovery / viz paths don't care
// which version is in use. Flip to sdf_gradient_4 to A/B compare.
#define sdf_gradient  sdf_gradient_4

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos       : SV_POSITION;
	float3 world_pos : TEXCOORD0;
	uint   view_idx  : TEXCOORD1;
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;
	float4 world = mul(input.pos, sk_inst[ids.inst].world);
	o.pos        = mul(world, sk_viewproj[ids.view]);
	o.world_pos  = world.xyz;
	o.view_idx   = ids.view;
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	float3 cam_pos     = sk_camera_pos[input.view_idx].xyz;
	float3 ray_end_w   = input.world_pos;       // backface = far side of volume.
	float3 dir         = normalize(ray_end_w - cam_pos);

	// Volume is non-cubic; voxel spacing is uniform so derive voxel_size
	// from any axis (X as canonical). vol_extent is per-axis for the AABB
	// slab test below.
	float  voxel_size  = (1.0 / gi_volume_inv.x) / (float)VOXEL_RES_X;
	float3 vol_extent  = 1.0 / gi_volume_inv;
	float3 volume_max  = gi_volume_min + vol_extent;
	int3   origin_vox  = int3(floor(gi_volume_min / voxel_size));

	// Pick a ray start: inside the volume → camera itself; outside → near
	// intersection with the AABB (slab test). Backface culling means the PS
	// always runs even when the camera is inside the cube; the ray-AABB test
	// here handles the outside case so we don't start the march outside the
	// volume and immediately discard.
	float3 start_w;
	if (all(cam_pos >= gi_volume_min) && all(cam_pos <= volume_max)) {
		start_w = cam_pos;
	} else {
		// Standard ray-slab AABB intersection. Divide-by-zero on axis-aligned
		// rays produces ±inf which behaves correctly under min/max.
		float3 inv_dir = 1.0 / dir;
		float3 t0      = (gi_volume_min - cam_pos) * inv_dir;
		float3 t1      = (volume_max    - cam_pos) * inv_dir;
		float3 tn      = min(t0, t1);
		float3 tf      = max(t0, t1);
		float  t_near  = max(max(tn.x, tn.y), tn.z);
		float  t_far   = min(min(tf.x, tf.y), tf.z);
		if (t_far < t_near || t_far <= 0) discard;
		start_w = cam_pos + dir * max(t_near, 0.0);
	}

	// March in world-voxel coordinates. ray_pos_w is the position in voxel
	// units (1 unit = 1 voxel side), so SDF jumps land at voxel-aligned steps
	// and DDA t-values are directly voxel-distances.
	float3 ray_pos_w = start_w / voxel_size;

	// Cap is roughly the volume's body diagonal in voxel units (~377 for the
	// non-cubic 256×128×256 grid), but rays in open space cross in 1–3 SDF
	// jumps so the cap rarely matters.
	[loop] for (int i = 0; i < 256; i++) {
		int3 vox_w = int3(floor(ray_pos_w));
		int3 local = vox_w - origin_vox;
		if (any(local < 0) || any(local >= VOXEL_RES)) discard;

		int3  slot  = vox_w & VOXEL_RES_MASK;
		// Smooth (HW-filtered) SDF sample for a continuous distance field —
		// the level set is then a smooth surface instead of snapping to
		// voxel boundaries.
		float sdf_v = sample_sdf_smooth(ray_pos_w);

		if (sdf_v < HIT_THICKNESS) {
#if RECOVERY_MODE == RECOVERY_GRAD_VIZ
			// Pure gradient visualization — skip voxel lookups entirely
			// and write the surface normal directly as RGB. Gradient is
			// sampled at ray_pos_w (on the level set itself) where it
			// cleanly represents the outward surface normal; positions
			// past the forward step can fall into the SDF's flat zero-
			// region inside thick geometry and yield a degenerate
			// gradient. The [-1, 1] → [0, 1] remap matches the standard
			// "normal texture" colour convention so axis-aligned faces
			// hit the recognizable solid tones (up = (0.5, 1, 0.5) etc).
			float3 grad   = sdf_gradient(ray_pos_w);
			float3 normal = length(grad) > 1e-4 ? normalize(grad) : float3(0, 1, 0);
			return float4(normal * 0.5 + 0.5, 1.0);
#endif
			//dir = sdf_gradient(ray_pos_w);
			//dir = length(dir) > 1e-4 ? -normalize(dir) : dir;

			// Hit the SDF level set — jump forward by HIT_THICKNESS so the
			// slot lookup lands past the inflation shell into the actual
			// filled cell (otherwise axis-aligned hits sit on the slot
			// boundary and `floor()` flips to the empty side at half the
			// hits). Then fall back to a short DDA walk if that lookup
			// missed (thin shell, grazing ray, or feature smaller than
			// HIT_THICKNESS) before giving up and letting the fragment
			// pass through to the scene behind.
			float3 walk      = ray_pos_w;
			int3   walk_vox  = int3(floor(walk));
			int3   walk_slot = walk_vox & VOXEL_RES_MASK;

			float4 v = voxel_tex.Load(int4(walk_slot, 0));
			if (v.a > 0) {
				// Voxel-storage scale: voxels store lit / voxel_range_pad.x
				// (so unorm rgba8 covers [0, voxel_range_pad.x]). Multiply
				// back for the displayed radiance.
				return float4(v.rgb * gi_voxel_range_pad.x, 1.0);
			}

#if RECOVERY_MODE == RECOVERY_DIR_STEP
			// One extra 1-voxel jump along the ray direction, then a
			// single voxel.Load. Gradient-free counterpart to GRAD_STEP
			// — same cost shape, no SDF normal computation needed.
			walk += dir;
			walk_vox = int3(floor(walk));
			if (!any(uint3(walk_vox - origin_vox) >= uint3(VOXEL_RES))) {
				walk_slot = walk_vox & VOXEL_RES_MASK;
				float4 wv = voxel_tex.Load(int4(walk_slot, 0));
				if (wv.a > 0) return float4(wv.rgb * gi_voxel_range_pad.x, 1.0);
			}
#elif RECOVERY_MODE == RECOVERY_GRAD_STEP
			// Single 1-voxel step along -∇SDF — the shortest direction
			// through the inflation shell toward the actual filled cell.
			// Falls back to `dir` if the gradient is degenerate (≈0 length,
			// e.g. deep inside a solid where all 4 taps return 0).
			float3 grad   = sdf_gradient(walk);
			float3 inward = length(grad) > 1e-4 ? -normalize(grad) : dir;
			walk += inward;
			walk_vox = int3(floor(walk));
			if (!any(uint3(walk_vox - origin_vox) >= uint3(VOXEL_RES))) {
				walk_slot = walk_vox & VOXEL_RES_MASK;
				float4 wv = voxel_tex.Load(int4(walk_slot, 0));
				if (wv.a > 0) return float4(wv.rgb * gi_voxel_range_pad.x, 1.0);
			}
#elif RECOVERY_MODE == RECOVERY_DDA
			[unroll] for (int j = 0; j < COLOR_RECOVERY; j++) {
				float3 boundary = floor(walk) + step(0.0, dir);
				float3 t_axis   = (boundary - walk) / dir;
				float  t_step   = min(min(t_axis.x, t_axis.y), t_axis.z) + 1e-4;
				walk += dir * t_step;
				walk_vox = int3(floor(walk));
				if (any(uint3(walk_vox - origin_vox) >= uint3(VOXEL_RES))) break;
				walk_slot = walk_vox & VOXEL_RES_MASK;
				float4 wv = voxel_tex.Load(int4(walk_slot, 0));
				if (wv.a > 0) return float4(wv.rgb * gi_voxel_range_pad.x, 1.0);
			}
#endif
			// RECOVERY_NONE falls straight through to the discard below: the
			// forward-step lookup already missed, no fallback walk happens.
			// Recovery exhausted without finding a filled cell — transparent
			// so the scene behind the volume shows through instead of an
			// opaque black smudge.
			discard;
		}

		// Classic iq-style sphere trace — step by the full SDF distance.
		// No `max(..., 1.0)` clamp: with smooth-filtered SDF the step
		// shrinks naturally as we approach the surface, and the
		// HIT_THICKNESS check catches us before sdf_v ever reaches 0.
		// A 1-voxel minimum would overshoot past the level set when
		// sdf_v sits between HIT_THICKNESS and 1, fattening the shell
		// asymmetrically by the leftover overshoot distance.
		ray_pos_w += dir * sdf_v;
	}
	discard;
	return float4(0, 0, 0, 0);
}

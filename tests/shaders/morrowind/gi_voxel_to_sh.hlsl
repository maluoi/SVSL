#include "lighting.hlsli"

//--name = app/gi_voxel_to_sh
// Voxel→SH compute. Each thread is one probe; each frame casts ray_count
// stratified Fibonacci-sphere rays through the voxel volume, projects to
// L1 SH, and stores the result in a ring-buffer history slot. The probe's
// final SH is the *exact* mean of all history slots — no EMA decay tail,
// equal 1/N weight per frame, bounded error.
//
// Additionally performs next-event light sampling: for each of the 4 nearest
// dynamic point lights, walks a shadow ray through the voxel volume to that
// light. If unoccluded, the light's directional radiance is added to the
// probe's SH directly — bypasses the Monte-Carlo-hits-light lottery, which
// is effectively zero for sub-meter point lights at 16 rays per probe.
// This is what lets probes carry direct-lighting-with-visibility, enabling
// soft bounce shadows around fixtures.
//
// Adapted from the StereoKit GI demo with three changes:
//   - GIBuffer slot moved to b4 (LightingBuffer is at b12); padded to 80 B.
//   - Ray-miss sampling reads from our SkyLightingBuffer at b5 (the L2 SH
//     mirror that GI.cs repopulates from Renderer.SkyLight each frame).
//     We don't pull in <stereokit.hlsli> because its declarations (b1
//     stereokit_buffer, t12 sk_inst, t11 sk_cubemap) aren't reliably bound
//     for compute dispatches.

cbuffer GIBuffer : register(b4) {
	float3 gi_volume_min;
	float  gi_voxel_res;
	float3 gi_volume_inv;
	float  gi_intensity;
	// .xyz = previous frame's origin probe-cell (integer-as-float). Used by
	// the toroidal history invalidator below to detect probes whose active
	// world cell rotated since last frame.
	float4 prev_origin_cell;
	// .x = voxel storage range scale; voxel.rgb × this = real radiance.
	float4 voxel_range_pad;
	float4 _gi_pad2;
};

// L2 SH mirror of Renderer.SkyLight. .xyz = RGB coefficient; .w unused.
// Standard real-spherical-harmonics basis (matches SphericalHarmonics.ToArray):
//   c0      L0 = Y_00
//   c1..c3  L1 in (y, z, x) order
//   c4..c8  L2 in (xy, yz, zz-norm, xz, xx-yy) order
cbuffer SkyLightingBuffer : register(b5) {
	float4 sky_sh[9];
};

// Evaluate the sky SH in direction `dir`. Returns a low-frequency directional
// color — used as the radiance estimate for rays that exit the voxel volume.
float3 gi_sky_lighting(float3 dir) {
	float x = dir.x, y = dir.y, z = dir.z;
	float3 col =
		sky_sh[0].rgb * 0.282095 +
		sky_sh[1].rgb * (0.488603 * y) +
		sky_sh[2].rgb * (0.488603 * z) +
		sky_sh[3].rgb * (0.488603 * x) +
		sky_sh[4].rgb * (1.092548 * x * y) +
		sky_sh[5].rgb * (1.092548 * y * z) +
		sky_sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) +
		sky_sh[7].rgb * (1.092548 * x * z) +
		sky_sh[8].rgb * (0.546274 * (x * x - y * y));
	return max(0, col);
}

// Live SH (Stage-1 ZH3 compression — Roughton et al., I3D 2024):
//   sh_dc       — Rgba16f, HDR DC RGB per probe in .rgb. .a holds the trap
//                 factor in [0,1] (Stage 2: packed here to free a UAV slot
//                 for the history buffer; the dedicated R8 sh_state UAV/SRV
//                 is gone). fp16 alpha gives ~4× the precision of the old R8.
//   sh_r/g/b_l1 — Rgba8 unorm, per-channel L1 ratios in (y, z, x, _).
//                 Stored offset-binary in [0, 1]; decoded as
//                 `coef = (raw * 2 - 1) * √3 · DC`.
//
// History (Stage 3): single RWStructuredBuffer of 16-byte entries, one per
// (probe × ring slot). Hits the paper's ~13-byte target by packing each entry
// as [fp16 dc.r | fp16 dc.g | fp16 dc.b | 9× uint8 ratio | 1 byte pad] across
// 4 uints. Switching from 4 textures to a single buffer also frees three UAV
// slots — the live-SH ones can sit at their natural u0..u2 + u12 without
// playing slot-collision tetris.
//
// A previous attempt at structured-buffer history hit a "every 32nd frame
// correct, the rest black" flicker on this hardware. The likely cause was
// the buffer not persisting between frames (transient allocation per dispatch
// would explain that exact 1-in-32 pattern: only the current cycle slot has
// valid data, ring average reads garbage from the other 31 slots, every 32
// frames cycle_offset wraps back to 0 and re-stamps that slot, etc). This
// pass uses a single persistent ComputeBuffer<ShHistEntry> created once in
// GI.cs's Init — no per-frame reallocation, no per-frame clear (the
// fresh-seed logic stamps all 32 slots on the first frame via the -1e9
// origin-cell sentinel, so initial buffer contents never get read).
//
// === UAV slot allocation note ===
// sk_renderer's global_textures[16] (for storage images) and global_buffers[16]
// (for cbuffers + RWStructuredBuffers) are both indexed by raw slot number,
// and the "missing-global → use binds[i]" fallback only fires when that
// global slot is null. So:
//   - A compute storage image at u# is misdirected if any t# global is set
//     at the same number.
//   - A compute RWStructuredBuffer at u# is misdirected if any b# global is
//     set at the same number.
//
// global_textures claimed: 6..8, 10 (GI live-SH), 11 (SK sky cubemap), 13
// (shadow), 14..15 (lighting).
// global_buffers claimed: 1 (stereokit_buffer), 3 (FogBuffer), 4 (GIBuffer),
// 12 (LightingBuffer), 13 (ShadowBuffer).
//
// We use u0..u2 + u12 for storage images (free in global_textures), and u5
// for the RWStructuredBuffer (free in global_buffers — SkyLightingBuffer
// at b5 is bound locally on the compute via SetConstant, not via
// SetGlobalBuffer, so it doesn't occupy global_buffers[5]).
//
// === ShHistEntry — 16-byte packed ring slot ===
// One entry per (probe × ring slot). The packing layout:
//   p0 [0..15] fp16 dc.r          [16..31] fp16 dc.g
//   p1 [0..15] fp16 dc.b          [16..23] u8 r_ratio.x  [24..31] u8 r_ratio.y
//   p2 [0..7]  u8 r_ratio.z       [8..15]  u8 g_ratio.x
//      [16..23] u8 g_ratio.y      [24..31] u8 g_ratio.z
//   p3 [0..7]  u8 b_ratio.x       [8..15]  u8 b_ratio.y
//      [16..23] u8 b_ratio.z      [24..31] unused
// 3 × fp16 DC (6 B) + 9 × u8 ratios (9 B) + 1 B pad = 16 B. Matches the C#
// ShHistEntry struct verbatim — keep them in lockstep.
struct ShHistEntry { uint p0, p1, p2, p3; };

Texture3D<float4>                                   voxel_tex      : register(t0);
Texture3D<float>                                    sdf_tex        : register(t1);
[[vk::image_format("rgba16f")]] RWTexture3D<float4> sh_dc          : register(u0);
[[vk::image_format("rgba8")]]   RWTexture3D<float4> sh_r_l1        : register(u1);
[[vk::image_format("rgba8")]]   RWTexture3D<float4> sh_g_l1        : register(u2);
[[vk::image_format("rgba8")]]   RWTexture3D<float4> sh_b_l1        : register(u12);
RWStructuredBuffer<ShHistEntry>                     sh_hist        : register(u5);

// === u8 ratio (offset-binary) helpers ===
// Encode: [-1, 1] → uint [0, 255] with round-to-nearest.
// Decode: uint [0, 255] → [-1, 1].
uint  _u8q (float v) { return (uint)(clamp(v * 0.5 + 0.5, 0.0, 1.0) * 255.0 + 0.5); }
float _u8dq(uint  b) { return (float)b * (2.0 / 255.0) - 1.0; }

// === ShHistEntry encode / decode ===
// Round-trip is float → fp16(DC) + u8(ratio) → float, mirroring the live-SH
// ratio scheme but with DC kept in fp16 instead of full fp32. Ratios are
// already in [-1, 1] by Wyman-Sloan / encoder clamp.
ShHistEntry _hist_encode(float3 dc, float3 r_ratio, float3 g_ratio, float3 b_ratio) {
	ShHistEntry e;
	e.p0 = f32tof16(dc.r) | (f32tof16(dc.g) << 16);
	e.p1 = f32tof16(dc.b) | (_u8q(r_ratio.x) << 16) | (_u8q(r_ratio.y) << 24);
	e.p2 = _u8q(r_ratio.z) | (_u8q(g_ratio.x) << 8) | (_u8q(g_ratio.y) << 16) | (_u8q(g_ratio.z) << 24);
	e.p3 = _u8q(b_ratio.x) | (_u8q(b_ratio.y) << 8) | (_u8q(b_ratio.z) << 16);
	return e;
}

void _hist_decode(ShHistEntry e, out float3 dc, out float3 r_ratio, out float3 g_ratio, out float3 b_ratio) {
	dc.r        = f16tof32(e.p0 & 0xFFFF);
	dc.g        = f16tof32(e.p0 >> 16);
	dc.b        = f16tof32(e.p1 & 0xFFFF);
	r_ratio.x   = _u8dq((e.p1 >> 16) & 0xFF);
	r_ratio.y   = _u8dq((e.p1 >> 24) & 0xFF);
	r_ratio.z   = _u8dq( e.p2        & 0xFF);
	g_ratio.x   = _u8dq((e.p2 >> 8 ) & 0xFF);
	g_ratio.y   = _u8dq((e.p2 >> 16) & 0xFF);
	g_ratio.z   = _u8dq((e.p2 >> 24) & 0xFF);
	b_ratio.x   = _u8dq( e.p3        & 0xFF);
	b_ratio.y   = _u8dq((e.p3 >> 8 ) & 0xFF);
	b_ratio.z   = _u8dq((e.p3 >> 16) & 0xFF);
}

uint frame_seed;     // monotonic frame counter

// Compile-time-fixed per-axis sizing. The corresponding C# params in GI.Init
// assert these values so the shader and the ComputeBuffer sizing stay in
// lockstep — hardcoded so the toroidal `((x % res) + res) % res` modulo dance
// (which AMD lowers to ~13 instructions per axis on a runtime divisor)
// collapses to a single `x & MASK`. Voxel is locked at 4× probe per-axis to
// preserve the voxel/probe ratio that the volume sizing assumes.
//
// Grid is non-cubic: 64×32×64 probes (Y vertical, smaller because terrain
// extends mostly horizontally), voxels at 4× = 256×128×256.
#define PROBE_RES_X        64u
#define PROBE_RES_Y        32u
#define PROBE_RES_Z        64u
#define VOXEL_RES_X        256u
#define VOXEL_RES_Y        128u
#define VOXEL_RES_Z        256u
// int3 forms for componentwise toroidal AND and bounds checks. Must match
// gi.hlsli — these are duplicated here because gi_voxel_to_sh doesn't pull
// in gi.hlsli (it would drag in stereokit.hlsli, whose bindings collide).
static const int3 PROBE_RES      = int3(PROBE_RES_X,     PROBE_RES_Y,     PROBE_RES_Z);
static const int3 PROBE_RES_MASK = int3(PROBE_RES_X - 1, PROBE_RES_Y - 1, PROBE_RES_Z - 1);
static const int3 VOXEL_RES      = int3(VOXEL_RES_X,     VOXEL_RES_Y,     VOXEL_RES_Z);
static const int3 VOXEL_RES_MASK = int3(VOXEL_RES_X - 1, VOXEL_RES_Y - 1, VOXEL_RES_Z - 1);
#define HISTORY_SIZE      32u
#define RAY_COUNT         16u
// Next-event light sampling. When 0, the per-light shadow walk + SH
// accumulation below is compiled out entirely (no shadow-walk cost, no
// direct-light contribution baked into the probes). Useful for isolating
// the bounce term in profiling / A-B comparisons. Bounce rays still pick
// up direct light naturally via the voxelized albedo×sun term, just
// without the visibility-aware NEE refinement.
#define NEE_ENABLED       1
// Scale applied to bounce-ray radiance on voxel hits (not to sky-miss
// contributions, which keep their natural intensity). Lets you boost or
// attenuate the indirect bounce term independently of ambient sky. 1.0 =
// physical; <1 darkens bounce vs sky; >1 exaggerates colour bleed.
// Distinct from gi_voxelize.hlsl's BOUNCE_STRENGTH, which governs the
// multi-bounce feedback gain at voxelize time (this frame's input).
#define BOUNCE_STRENGTH   2.0

// Linear index into the ring buffer. Probes' HISTORY_SIZE ring slots are
// contiguous in memory so the per-probe ring-average loop reads adjacent
// entries — good for cache.
uint _hist_idx(uint3 id, uint slot_idx) {
	uint probe_linear = id.x + (id.y + id.z * PROBE_RES_Y) * PROBE_RES_X;
	return probe_linear * HISTORY_SIZE + slot_idx;
}

#define GOLDEN_RATIO 1.6180339887

// === Ray-walk backend ===
// Hybrid sphere-trace + DDA through the R8 SDF built each voxelize frame
// from the voxel volume (see gi_sdf_*.hlsl). Far from geometry, jump the
// full SDF distance — crosses open space in 1–2 steps. Below a near-
// geometry threshold (~2 voxels), DDA-step exactly to the next voxel
// boundary along the ray so every voxel the ray crosses is sampled. This
// fixes the "sphere tracing lands somewhere on the safe-radius sphere, not
// at the seed" failure mode that misses grazing rays through thin features.
// Bounds-checked against the active window so toroidal wrap can't alias
// OOB rays to wrong-cell slots.
// Max SDF distance encodable in R8 unorm. Set to half the longest axis (128
// voxels) so each R8 step is ~0.5 voxels — enough to keep cardinal-1 vs
// diagonal-√2 vs corner-√3 distinguishable, which the trilinear-filtered
// debug viz needs to avoid caltrop level sets near thin features. Anything
// past 128 voxels saturates; the SH compute can still cross half the volume
// in a single safe-step and the bounds check handles the rest.
#define SDF_MAX_VOXELS      128.0
// Sphere-trace stop epsilon, in voxel units — the threshold below which we
// drop out of pure SDF stepping and start sampling voxels directly. Matches
// gi_sdf_debug.hlsl's HIT_THICKNESS, so the bounce + NEE walks see the same
// surface the debug viz draws. No JFA margin / 1-voxel-min clamp: the world-
// correct slot-local JFA now produces honest distances (the inflated-
// estimate failure mode that motivated the margin is gone), so a plain
// `dir * sdf_v` jump is safe.
#define HIT_THICKNESS       0.35

uint _hash(uint x) { x ^= x >> 16; x *= 0x45d9f3bu; x ^= x >> 16; return x; }

float3 _fibonacci_dir(uint i, uint total, float offset) {
	float fi = (float)i + 0.5;
	float fn = (float)total;
	float z   = 1.0 - 2.0 * fi / fn;
	float r   = sqrt(max(0.0, 1.0 - z * z));
	float phi = 6.28318530718 * frac(fi * (1.0 / GOLDEN_RATIO) + offset);
	return float3(r * cos(phi), r * sin(phi), z);
}

// Force wave32 on RDNA. The AMD Vulkan driver defaults compute to wave64
// (legacy GCN behavior — wave32 was new in RDNA1), but for a memory-bound
// shader this dispatch is launch-size-limited: with 32³ probes at 64-thread
// workgroups, we only emit 512 wave64s across 96 SIMDs (≈5.3 waves/SIMD vs.
// the 16-wave/SIMD hw cap → 33% occupancy). Wave32 doubles the wave count
// per workgroup (2× wave32 per group of 64 threads), bringing per-SIMD
// average to ~10.7 waves and ~67% occupancy — twice the latency-hiding
// capacity for the same workload. The texture-fetch divergence cost is
// already real (each ray walks its own voxel column), so coalescing 64
// lanes vs. 32 lanes makes little difference here. See PAL metadata in
// the RGP capture for the wavefront_size confirmation.
//--wave_size = 32
[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (any(id >= uint3(PROBE_RES))) return;

	// Volume can be non-cubic; spacing is uniform because the probe count and
	// volume size scale together — derive scalar probe/voxel sizes from the
	// X-axis ratio, valid for any axis.
	float3 vol_size   = 1.0 / gi_volume_inv;
	float  probe_size = vol_size.x / (float)PROBE_RES_X;
	float  voxel_size = vol_size.x / (float)VOXEL_RES_X;

	// Toroidal slot → world cell mapping. A slot represents the world cell
	// `c` where `c mod probeRes == slot` and `c ∈ [origin, origin + probeRes)`.
	// As origin slides, most slots' active cells stay the same — only the
	// ones whose modular class just rotated change. PROBE_RES is a power of
	// two, so the canonical `((x % res) + res) % res` collapses to `x & MASK`
	// (two's-complement bottom-bits == positive remainder for any sign of x).
	int3 curr_origin = int3(floor(gi_volume_min / probe_size));
	int3 slot        = int3(id);
	int3 delta_curr  = (slot - curr_origin) & PROBE_RES_MASK;
	int3 active_cell = curr_origin + delta_curr;
	float3 probe_w   = (float3(active_cell) + 0.5) * probe_size;

	// Did this slot rotate to a different active cell this frame? If so, the
	// 31 non-current history entries refer to the OLD world cell and need to
	// be re-seeded with the current SH instead of averaged in.
	int3  prev_origin = int3(prev_origin_cell.xyz);
	int3  delta_prev  = (slot - prev_origin) & PROBE_RES_MASK;
	int3  prev_active = prev_origin + delta_prev;
	bool  slot_rotated = any(prev_active != active_cell);
	// Last frame's trap factor — used to detect a "trapped → free" transition
	// so we can re-seed the history ring along with the toroidal case.
	// Packed into sh_dc.a (Stage 2: the dedicated R8 sh_state texture went
	// away to free a UAV slot for the 4th history texture).
	float prev_trap   = sh_dc[id].a;

	// Per-probe angular offset. We measured wave-coherent variants (per-
	// workgroup, per-dispatch) hoping for L1 cache reuse from coordinated
	// voxel fetches across the wave. They were measurably WORSE: per-iter
	// wait time more than doubled (28.6 vs 11.4 clk/iter). The TMU address-
	// generator has parallel banks that thrive on scattered fetches; 32
	// addresses hashing to the same region (wave-coherent) serialize at
	// the bank arbitrator before L1 is even reached. Per-probe random keeps
	// the fetches spread across banks → best throughput in practice.
	// Bit-packing avoids a 3-deep dependent hash chain.
	uint  h     = _hash(id.x | (id.y << 10) | (id.z << 20));
	float p_off = (float)h / 4294967295.0;

	uint cycle_offset = frame_seed % HISTORY_SIZE;

	float w_per = 12.56637 / (float)RAY_COUNT; // 4π / N

	float4 acc_r = 0;
	float4 acc_g = 0;
	float4 acc_b = 0;

	// Count rays whose first hit lands within NEAR_T voxels of the probe.
	// A probe in open air has near_hits ≈ 0; one buried in a hillside has
	// near_hits ≈ ray_count. The fraction drives the per-probe trap factor.
	// 5 voxels ≈ 1.9 m at the default 0.375 m voxel size — small enough that
	// a probe in a normal 3 m room mostly hits at "far" range, big enough
	// that a probe wedged 0.5 m inside a thin slope still flags as trapped.
	// We compare against hit_t (voxel-distance walked) rather than a loop
	// iteration count — under hybrid SDF+DDA marching, a single SDF jump
	// can cover many voxels so iteration-count is no longer voxel-aligned.
	const float NEAR_T = 5.0;
	uint near_hits = 0;

	// Walk in world voxel coordinates (so the toroidal slot is just
	// `world_vox mod voxel_res`). origin_local is the volume-relative
	// position in [0, voxel_res) — used for the t_exit math.
	int3   origin_vox   = int3(floor(gi_volume_min / voxel_size));
	float3 origin_v     = probe_w / voxel_size;
	float3 origin_local = origin_v - float3(origin_vox);

	// Strided Fibonacci index per frame: 16 directions spread across the
	// whole 512-point sphere, one from each 32-direction "bin". Each frame
	// is a complete, near-uniform Monte Carlo sample on its own, so the SH
	// estimate is stable frame-to-frame even before the 32-frame ring fills.
	//
	// (Contiguous blocks [N..N+15] don't give "small cones" — the Fibonacci
	// sphere has slowly-varying z but wildly-varying phi between consecutive
	// indices, so 16 in a row form a thin ring near one pole. Over 32 frames
	// the ring would sweep z from +1 → −1, producing a visible rotating
	// directional bias.)
	[loop] for (uint ray = 0; ray < RAY_COUNT; ray++) {
		uint   ray_idx = ray * HISTORY_SIZE + cycle_offset;
		float3 dir     = _fibonacci_dir(ray_idx, RAY_COUNT * HISTORY_SIZE, p_off);

		// Sphere-trace cap. With SDF jumps the typical ray needs <16 steps to
		// exit / hit; cap stays conservative at the volume's longest axis so a
		// worst-case "1-voxel-per-step through dense geometry" sweep still
		// terminates before hitting the bounds check.
		const uint max_steps = VOXEL_RES_X;

		// Skip the immediate probe-containing voxel (start one step in). With
		// trap detection in place, probes whose rays would otherwise drown in
		// their own surrounding geometry get flagged and suppressed directly,
		// so we no longer need to push every ray's start further out — that
		// only cost contact bounce on legitimate near-surface probes.
		const float start_s = 1.0;

		float3 radiance = 0;
		bool   hit      = false;
		float  hit_t    = 0.0;       // voxel-distance from probe to hit, for trap

		// Hybrid sphere-trace + DDA. ray_pos_v is in voxel units (= world
		// position / voxel_size). t_walked tracks voxel-distance from the
		// probe so the trap detector can compare against NEAR_T directly
		// (a step count would mix big SDF jumps with 1-voxel DDA steps).
		float3 ray_pos_v = origin_v + dir * start_s;
		float  t_walked  = start_s;
		[loop] for (uint s = 0; s < max_steps; s++) {
			int3 vox_w = int3(floor(ray_pos_v));
			// In-window bounds check — out of the active window means the
			// ray has genuinely exited (toroidal wrap would otherwise alias
			// to a wrong-cell slot). Unsigned-cast trick folds <0 and
			// >=VOXEL_RES_axis into one compare per axis.
			if (any(uint3(vox_w - origin_vox) >= uint3(VOXEL_RES))) break;

			int3  slot  = vox_w & VOXEL_RES_MASK;
			float sdf_v = sdf_tex.Load(int4(slot, 0)) * SDF_MAX_VOXELS;

			if (sdf_v < HIT_THICKNESS) {
				// Hit the SDF level set — sample the voxel at the current
				// trace position (no forward step; the inflation-shell
				// inset that gi_sdf_debug.hlsl used was found to degrade
				// the result). If empty, fall through to a DDA boundary
				// step and keep marching: the SH compute can't terminate
				// on a single miss the way the debug viz does, or we'd
				// drop every grazing hit.
				float4 v = voxel_tex.Load(int4(slot, 0));
				if (v.a > 0) {
					radiance = v.rgb * voxel_range_pad.x * BOUNCE_STRENGTH;
					hit      = true;
					hit_t    = t_walked;
					break;
				}
				// Per-axis t to the next voxel boundary. step(0, dir.axis)
				// gives +1 for non-negative dir (boundary at floor+1) and
				// +0 for negative dir (boundary at floor). dir.axis==0
				// produces t.axis=inf which the min naturally ignores.
				// +1e-4 pushes a hair past the boundary so floor() lands
				// in the next cell rather than teetering on the integer plane.
				float3 boundary = floor(ray_pos_v) + step(0.0, dir);
				float3 t_axis   = (boundary - ray_pos_v) / dir;
				float  t_step   = min(min(t_axis.x, t_axis.y), t_axis.z) + 1e-4;
				ray_pos_v += dir * t_step;
				t_walked  += t_step;
			} else {
				// Far from geometry — full sphere-trace jump. Plain `dir * sdf_v`,
				// no margin or 1-voxel clamp: the slot-local JFA produces honest
				// world-distance SDFs so the inflated-estimate failure mode that
				// motivated the old margin is gone, and a clamp would overshoot
				// the level set whenever sdf_v sits between HIT_THICKNESS and 1.
				ray_pos_v += dir * sdf_v;
				t_walked  += sdf_v;
			}
		}

		// Ray exited the volume — read the sky SH along the ray direction.
		// Tracks day/night in exterior (Weather updates Renderer.SkyLight)
		// and the cell ambient in interior, automatically.
		if (!hit) radiance = gi_sky_lighting(dir);
		else if (hit_t < NEAR_T) near_hits++;

		float4 sh_dir = float4(0.28209, 0.48860 * dir.y, 0.48860 * dir.z, 0.48860 * dir.x) * w_per;
		acc_r += sh_dir * radiance.r;
		acc_g += sh_dir * radiance.g;
		acc_b += sh_dir * radiance.b;
	}

	// === Next-event light sampling ===
	// One light per frame, rotating through the cell's top-4 over 4 frames.
	// Per-frame contribution is scaled by 4 (the inverse sampling probability)
	// so the expected brightness after the 32-frame history ring fills equals
	// "sample all 4 every frame" — each light lands in 8 of 32 history slots,
	// ×4 scaling restores full contribution. Cuts NEE shadow-walk cost 4×.
	//
	// Trade-offs:
	//  - Slower warmup (full brightness only after ~32 frames, vs. 1 frame).
	//  - Slight slot-rotation jitter at volume slides (each light is in only
	//    8 of 32 slots, so re-seeding any of those biases the average until
	//    they refill).
	//
	// The DDA walk pattern mirrors the bounce ray loop above (same start_s,
	// bounds check, Load + alpha test, per-axis-boundary advance), plus a
	// "stop once we've walked past the light" cap.
	//
	// Skip NEE for thoroughly-trapped probes — their consumer weight is
	// (1 - trap), so the result is multiplied near-zero downstream. Avoids
	// a full shadow-walk per probe per frame for probes buried in geometry.
	// 0.85 is comfortably below the smoothstep top (0.95), so probes only
	// flicker their NEE work at the band edge.
#if NEE_ENABLED
	if (prev_trap < 0.85) {
	float2 light_cell_xz = floor(probe_w.xz * light_origin_inv.z);
	float2 light_grid_uv = (light_cell_xz + 0.5) * light_grid.zw;
	float4 light_idx     = light_grid_tex.SampleLevel(light_grid_s, light_grid_uv, 0);
	float  inv_table_w   = 1.0 / light_origin_inv.w;
	int    slots[4]      = { (int)light_idx.x, (int)light_idx.y, (int)light_idx.z, (int)light_idx.w };

	// Rotate through the 4 slots: slot 0 on frame 0, slot 1 on frame 1, etc.
	// Wrap in a `for (.. < 1)` so the existing per-light early-out `continue`s
	// fall through cleanly to the trap update without restructuring as nested ifs.
	int  nee_pick = (int)(frame_seed % 4);
	[loop] for (uint nee_iter = 0; nee_iter < 1; nee_iter++) {
		int li   = nee_pick;
		int slot = slots[li];
		if (slot == 0) continue;  // cell has fewer than 4 lights — empty sentinel

		float  lu0  = (slot * 2 + 0.5) * inv_table_w;
		float  lu1  = (slot * 2 + 1.5) * inv_table_w;
		float4 lpr  = light_table_tex.SampleLevel(light_table_s, float2(lu0, 0.5), 0);
		float4 lcp  = light_table_tex.SampleLevel(light_table_s, float2(lu1, 0.5), 0);

		float3 to_light    = lpr.xyz - probe_w;
		float  light_dist  = length(to_light);
		float  light_rad   = max(lpr.w, 1e-4);
		if (light_dist >= light_rad) continue;        // out of light's reach
		if (light_dist < 1e-4)        continue;       // divide-by-zero guard only

		float3 light_dir   = to_light / light_dist;
		float  dist_voxels = light_dist / voxel_size; // ray distance in voxel units

		// World voxel containing the light's actual position. Many MW LIGH
		// records sit inside their host fixture geometry (candle inside its
		// holder, lantern's flame inside the casing), which we voxelize
		// elsewhere in the scene. The shadow ray's last step lands right on
		// that voxel — if we treat it as a normal occluder, every NEE for
		// that light falsely reads as shadowed. Skip the alpha check for
		// the single voxel that contains the light.
		int3 light_voxel = int3(floor(lpr.xyz / voxel_size));

		// === SDF shadow walk (mirrors the bounce ray loop's structure) ===
		// Sphere trace through the same SDF the bounce loop uses, with an
		// extra distance cap at the light position. Voxels behind the light
		// aren't occluders for it. nee_max_steps tracks the bounce loop's
		// cap; in open space NEE finishes in 1–2 jumps.
		const uint  nee_max_steps = VOXEL_RES_X;
		const float nee_start_s   = 1.0;

		bool   occluded  = false;
		float3 nee_pos_v = origin_v + light_dir * nee_start_s;
		float  t_walked  = nee_start_s;
		[loop] for (uint i = 0; i < nee_max_steps; i++) {
			// Past the light — anything beyond it isn't an occluder for it.
			if (t_walked >= dist_voxels) break;
			int3 vox_n = int3(floor(nee_pos_v));
			if (any(uint3(vox_n - origin_vox) >= uint3(VOXEL_RES))) break;

			int3  slot  = vox_n & VOXEL_RES_MASK;
			float sdf_v = sdf_tex.Load(int4(slot, 0)) * SDF_MAX_VOXELS;

			if (sdf_v < HIT_THICKNESS) {
				// Near geometry — check voxel for occlusion at the current
				// trace position (no forward step; matches the bounce loop
				// and gi_sdf_debug.hlsl's RECOVERY_NONE behaviour). On a
				// miss, DDA-step to the next voxel boundary so thin walls
				// can't slip between sphere-trace samples.
				float4 v = voxel_tex.Load(int4(slot, 0));
				// Occlusion — unless this voxel is the light's own host
				// fixture (lantern body, candle holder); those self-voxels
				// would otherwise shadow every probe's NEE for that light.
				if (v.a > 0 && any(vox_n != light_voxel)) { occluded = true; break; }
				// DDA step (see bounce loop above for the per-axis t math).
				float3 boundary = floor(nee_pos_v) + step(0.0, light_dir);
				float3 t_axis   = (boundary - nee_pos_v) / light_dir;
				float  t_step   = min(min(t_axis.x, t_axis.y), t_axis.z) + 1e-4;
				t_step          = min(t_step, dist_voxels - t_walked);
				nee_pos_v      += light_dir * t_step;
				t_walked       += t_step;
			} else {
				// Far from geometry — full sphere-trace jump, capped at light
				// dist so we don't overshoot past the light position (voxels
				// beyond aren't occluders for it).
				float step_v = min(sdf_v, dist_voxels - t_walked);
				nee_pos_v   += light_dir * step_v;
				t_walked    += step_v;
			}
		}
		if (occluded) continue;

		// Visibility confirmed. Attenuation matches the formula in
		// lighting.hlsli so probe-encoded and surface-direct intensity agree.
		// NEE_SAMPLE_INV_PROB compensates for sampling 1 of 4 slots per frame
		// so the live SH average — over a full history cycle — matches what
		// "all 4 every frame" would produce.
		const float NEE_MAX_GAIN        = 30.0;
		const float NEE_SAMPLE_INV_PROB = 4.0;
		float  dr  = light_dist / light_rad;
		float  att = min(NEE_MAX_GAIN, 1.0 / (3.0 * max(dr, 1e-4)));
		float3 rad = lcp.rgb * att;

		float4 sh_dir = float4(0.28209, 0.48860 * light_dir.y, 0.48860 * light_dir.z, 0.48860 * light_dir.x) * NEE_SAMPLE_INV_PROB;
		acc_r += sh_dir * rad.r;
		acc_g += sh_dir * rad.g;
		acc_b += sh_dir * rad.b;
	}
	} // end if (prev_trap < 0.85)
#endif // NEE_ENABLED

	// Probe trap factor — soft transition between "obviously open" (≤75% of
	// rays hit close) and "obviously trapped" (≥95%). Probes near surfaces
	// (about half their hemisphere is occupied) stay well under the lower
	// threshold and contribute normally; only probes effectively buried in
	// thick geometry get fully zeroed.
	//
	// EMA-smooth the trap factor across frames so per-frame Fibonacci-phase
	// variation (different ray subsets sample different occluders) doesn't
	// oscillate the consumer-side (1-trap) weight or spuriously trigger the
	// state-revived history reset below. Time constant ~10 frames.
	float near_frac    = (float)near_hits / (float)RAY_COUNT;
	float instant_trap = smoothstep(0.75, 0.95, near_frac);
	float new_trap     = lerp(prev_trap, instant_trap, 0.1);

	// Re-seed the entire history ring when a probe transitions out of the
	// trapped state on a *sustained* basis — only fire when the smoothed
	// trap has actually descended through the deep band, not when single-
	// frame noise nibbles at the boundary. Toroidal rotation still triggers
	// re-seed unconditionally because that's a genuine world-cell change.
	bool state_revived = (prev_trap > 0.85) && (new_trap < 0.15);
	bool is_fresh_slot = slot_rotated || state_revived;

	// === Encode this frame's per-probe SH for the cycle-offset history slot ===
	// Per-frame DC is the per-slot ratio normalizer; later frames each have
	// their own DC. Clamp absorbs the Monte-Carlo overshoot a 16-ray estimate
	// can produce on highly directional probes (the Wyman-Sloan bound holds
	// for the true integral, not necessarily for a sparse estimate).
	const float INV_SQRT3 = 0.57735026919; // 1/√3
	float3 cur_dc        = max(float3(acc_r.x, acc_g.x, acc_b.x), 0);
	float3 cur_inv_norm  = INV_SQRT3 / max(cur_dc, 1e-6);
	float3 cur_r_ratio   = clamp(acc_r.yzw * cur_inv_norm.r, -1.0, 1.0);
	float3 cur_g_ratio   = clamp(acc_g.yzw * cur_inv_norm.g, -1.0, 1.0);
	float3 cur_b_ratio   = clamp(acc_b.yzw * cur_inv_norm.b, -1.0, 1.0);
	ShHistEntry cur_entry = _hist_encode(cur_dc, cur_r_ratio, cur_g_ratio, cur_b_ratio);

	// === Encode the sky-SH baseline for fresh-seed slots ===
	// Sky SH (from Renderer.SkyLight) is full-hemisphere average — neutral
	// baseline that the ring refills with real local data over the next
	// history_size frames. Encoded here in the same ratio form so its slot
	// layout matches the cycle-offset write. Scale matches: our (4π/N) × Y_00
	// integration weight cancels exactly with the sky-eval shaders' (1/Y_00)
	// factor, so sky_sh[0..3] is directly comparable to acc_r.xyzw.
	float4 sky_r4        = float4(sky_sh[0].r, sky_sh[1].r, sky_sh[2].r, sky_sh[3].r);
	float4 sky_g4        = float4(sky_sh[0].g, sky_sh[1].g, sky_sh[2].g, sky_sh[3].g);
	float4 sky_b4        = float4(sky_sh[0].b, sky_sh[1].b, sky_sh[2].b, sky_sh[3].b);
	float3 sky_dc        = max(float3(sky_r4.x, sky_g4.x, sky_b4.x), 0);
	float3 sky_inv_norm  = INV_SQRT3 / max(sky_dc, 1e-6);
	float3 sky_r_ratio   = clamp(sky_r4.yzw * sky_inv_norm.r, -1.0, 1.0);
	float3 sky_g_ratio   = clamp(sky_g4.yzw * sky_inv_norm.g, -1.0, 1.0);
	float3 sky_b_ratio   = clamp(sky_b4.yzw * sky_inv_norm.b, -1.0, 1.0);
	ShHistEntry sky_entry = _hist_encode(sky_dc, sky_r_ratio, sky_g_ratio, sky_b_ratio);

	// Write the current cycle slot with this frame's ray walk. Unconditional
	// single store — hoisted out of the loop so we don't burn 31 iterations
	// of skipped writes.
	sh_hist[_hist_idx(id, cycle_offset)] = cur_entry;

	// === Fresh-seed stamp + ring average, fused ===
	// On a fresh-seed (toroidal rotation or transition out of the trapped
	// state), the other 31 slots get the sky baseline so the averaged SH
	// starts at a neutral ambient and converges toward real local light
	// as later frames overwrite slots. The fresh-seed branch also covers
	// the first-ever frame: C# inits prev_origin_cell to -1e9, so
	// slot_rotated fires for every probe and stamps every ring slot — no
	// need to clear the ComputeBuffer on allocation.
	//
	// The per-slot read+decode runs in the same unrolled pass that writes
	// the fresh-seed entries. Special-cases avoid a buffer round-trip when
	// we already know the slot's contents:
	//   i == cycle_offset:               use the live cur_* floats.
	//   i != cycle_offset, is_fresh_slot: use the live sky_* floats.
	//   otherwise:                       decode from the buffer.
	// cycle_offset is uniform across the wave; is_fresh_slot is per-probe.
	const float GI_L1_NORM = 1.7320508; // √3
	float4 sum_r = 0, sum_g = 0, sum_b = 0;
	[unroll] for (uint i = 0; i < HISTORY_SIZE; i++) {
		uint   idx = _hist_idx(id, i);
		float3 slot_dc, slot_r_ratio, slot_g_ratio, slot_b_ratio;
		if (i == cycle_offset) {
			slot_dc      = cur_dc;
			slot_r_ratio = cur_r_ratio;
			slot_g_ratio = cur_g_ratio;
			slot_b_ratio = cur_b_ratio;
		} else if (is_fresh_slot) {
			sh_hist[idx] = sky_entry;
			slot_dc      = sky_dc;
			slot_r_ratio = sky_r_ratio;
			slot_g_ratio = sky_g_ratio;
			slot_b_ratio = sky_b_ratio;
		} else {
			_hist_decode(sh_hist[idx], slot_dc, slot_r_ratio, slot_g_ratio, slot_b_ratio);
		}
		float3 scale = GI_L1_NORM * slot_dc;
		sum_r += float4(slot_dc.r, slot_r_ratio * scale.r);
		sum_g += float4(slot_dc.g, slot_g_ratio * scale.g);
		sum_b += float4(slot_dc.b, slot_b_ratio * scale.b);
	}
	const float inv_n = 1.0 / (float)HISTORY_SIZE;
	float4 mean_r = sum_r * inv_n;
	float4 mean_g = sum_g * inv_n;
	float4 mean_b = sum_b * inv_n;

	// === Encode the averaged SH for the consumer-facing live read path ===
	// Same per-channel ratio scheme as the history write, applied to the
	// averaged SH. trap factor is packed into sh_dc.a (Stage 2 replacement
	// for the standalone R8 sh_state texture, which was eliminated to free
	// a UAV slot for the 4th history texture).
	float3 dc           = max(float3(mean_r.x, mean_g.x, mean_b.x), 0);
	float3 inv_norm     = INV_SQRT3 / max(dc, 1e-6);
	float3 r_l1         = clamp(mean_r.yzw * inv_norm.r, -1.0, 1.0);
	float3 g_l1         = clamp(mean_g.yzw * inv_norm.g, -1.0, 1.0);
	float3 b_l1         = clamp(mean_b.yzw * inv_norm.b, -1.0, 1.0);

	sh_dc  [id] = float4(dc,                    new_trap);
	sh_r_l1[id] = float4(r_l1 * 0.5 + 0.5,      0.0);
	sh_g_l1[id] = float4(g_l1 * 0.5 + 0.5,      0.0);
	sh_b_l1[id] = float4(b_l1 * 0.5 + 0.5,      0.0);
}

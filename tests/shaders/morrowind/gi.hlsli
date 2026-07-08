// Drop-in voxel-GI consumer header. Include this from any shader that wants
// to sample the indirect-light SH probe grid, and call
// `GISampleIndirect(world_pos, normal)` to get the diffuse irradiance.
//
// The system is driven CPU-side by SKMorrowind.GI (see src/GI/GI.cs): probe
// data is updated each frame by a compute shader that rays through a voxel
// volume containing the previous frame's "input light" (dyn-light bounces,
// in this project — sun term is left for future exterior work).
//
// Bindings (StereoKit reserves b1/t11/t12; project uses b3=fog, b12=lighting,
// b13/t13=shadow):
//   cbuffer b4    — GIBuffer (80 bytes — see CLAUDE.md > $Global sizing)
//   t6           — gi_sh_dc:   HDR DC RGB in .rgb (Rgba16f), trap factor in
//                  .a (Stage 2: packed here to free a UAV slot for the 4th
//                  history texture; the dedicated R8 sh_state texture is gone).
//   t7/t8/t10    — gi_sh_{r,g,b}_l1: per-channel L1 ratios in (y, z, x, _)
//                  (Rgba8 unorm, offset-binary).
//
// ZH3 compression (Roughton et al., I3D 2024 §"Bonus: Encoding SH").
// For any non-negative spherical function |L_l| ≤ √(2l+1)·|L0|, so the L1
// coefficient packs into [-1, 1] via `ratio = L1 / (√3 · DC)`, then offset-
// binary into rgba8 unorm. Decoded coefficient is `(raw·2 - 1) · √3 · DC`.
// Stage 1 applies this to the live SH; Stage 2 applies the same scheme to
// each ring slot (each slot's per-frame DC normalizes that slot's L1
// ratios). The ring-average loop in gi_voxel_to_sh decodes each slot back
// to float before summing, so per-slot 8-bit quantisation stays bounded
// per slot and doesn't accumulate through the average.
//
// Pulled in for sk_lighting(normal), the sky-SH fallback for samples outside
// the GI volume — keeps shader self-contained so consumers don't have to
// include stereokit.hlsli in a specific order.
#include "stereokit.hlsli"

cbuffer GIBuffer : register(b4) {
	float3 gi_volume_min;
	float  gi_voxel_res;
	float3 gi_volume_inv;
	float  gi_intensity;
	// .xyz = previous frame's origin probe-cell (used by the SH compute);
	// .w   = probe_res, plumbed through so the consumer-side toroidal
	//        slot math in the directional 8-tap works without a separate uniform.
	float4 gi_prev_origin_probe_res;
	// .x = voxel storage range scale (see GIBufferData.voxel_range_pad).
	float4 gi_voxel_range_pad;
	float4 _gi_pad2;
};

Texture3D<float4> gi_sh_dc    : register(t6);
Texture3D<float4> gi_sh_r_l1  : register(t7);
Texture3D<float4> gi_sh_g_l1  : register(t8);
Texture3D<float4> gi_sh_b_l1  : register(t10);
// Per-probe "trap factor" in [0, 1] — 0 = probe is in open space, 1 = probe is
// effectively buried in geometry (most of its rays hit close to itself).
// Computed in gi_voxel_to_sh.hlsl from the ray-walk statistics and packed into
// gi_sh_dc.a (Stage 2). Consumed by the directional 8-tap below to suppress
// contributions from probes that would otherwise leak dark interior-of-ground
// values onto nearby surfaces.

// √3 — Wyman-Sloan L1 normalisation constant (|L1| ≤ √3·DC for non-negative
// functions). Both encode (compute) and decode (here) multiply by this.
static const float GI_L1_NORM = 1.7320508;

// Compile-time-fixed per-axis probe / voxel resolution. Must match the
// matching defines in the other gi_*.hlsl shaders and the values asserted
// in GI.Init — hardcoded so the canonical toroidal `((cell % res) + res) % res`
// collapses to `cell & MASK`, which is ~13 instructions cheaper per axis
// than a runtime-divisor modulo on AMD. Reading the resolution from the
// cbuffer would force the slow form (the SSA optimizer can't prove power-
// of-two for a uniform).
//
// Grid is non-cubic: 64×32×64 probes (Y=32 because vertical needs less
// XZ coverage), voxels at 4× = 256×128×256. Volume sizing scales with the
// probe count so spacing stays uniform at the configured probeSpacing —
// voxel_size is therefore still a single scalar throughout the shaders.
#define PROBE_RES_X        64
#define PROBE_RES_Y        32
#define PROBE_RES_Z        64
#define VOXEL_RES_X        256
#define VOXEL_RES_Y        128
#define VOXEL_RES_Z        256
// int3 forms for componentwise toroidal AND and bounds checks. Static const
// (not #define) so they're a single value the optimizer can fold rather than
// re-constructing the int3 at each use site.
static const int3 PROBE_RES      = int3(PROBE_RES_X,     PROBE_RES_Y,     PROBE_RES_Z);
static const int3 PROBE_RES_MASK = int3(PROBE_RES_X - 1, PROBE_RES_Y - 1, PROBE_RES_Z - 1);
static const int3 VOXEL_RES      = int3(VOXEL_RES_X,     VOXEL_RES_Y,     VOXEL_RES_Z);
static const int3 VOXEL_RES_MASK = int3(VOXEL_RES_X - 1, VOXEL_RES_Y - 1, VOXEL_RES_Z - 1);

// Decode one probe slot back to the (L0, L1_y, L1_z, L1_x) layout the rest of
// this header expects. Loads from the 4 compressed textures and inverts the
// offset-binary + √3·DC normalisation done by gi_voxel_to_sh.hlsl. Also
// returns the trap factor that lives in gi_sh_dc.a — a single Load covers
// both, since DC and trap share the same texel.
void _gi_decode_probe(int3 slot,
                      out float4 r4, out float4 g4, out float4 b4,
                      out float trap)
{
	float4 dc_full  = gi_sh_dc.Load(int4(slot, 0));
	float3 dc       = dc_full.rgb;
	trap            = dc_full.a;
	float3 r_ratio  = gi_sh_r_l1.Load(int4(slot, 0)).rgb * 2.0 - 1.0;
	float3 g_ratio  = gi_sh_g_l1.Load(int4(slot, 0)).rgb * 2.0 - 1.0;
	float3 b_ratio  = gi_sh_b_l1.Load(int4(slot, 0)).rgb * 2.0 - 1.0;
	float3 scale    = GI_L1_NORM * dc;
	r4 = float4(dc.r, r_ratio * scale.r);
	g4 = float4(dc.g, g_ratio * scale.g);
	b4 = float4(dc.b, b_ratio * scale.b);
}

// Reconstruct L1 *irradiance* (cosine-weighted incoming light, i.e. the
// Lambertian surface response) from SH projected with the raw basis.
// Coefficients are the Ramamoorthi irradiance-kernel weights:
//   π × 0.282095        ≈ 0.88623   (L0)
//   (2π/3) × 0.488603   ≈ 1.02333   (L1)
float _gi_eval_sh(float4 sh_v, float3 n) {
	return dot(sh_v, float4(0.88623, 1.02333 * n.y, 1.02333 * n.z, 1.02333 * n.x));
}

// === ZH3 hallucinate (Roughton et al., I3D 2024) ===
// 1 = Augment linear SH evaluation with a "hallucinated" quadratic zonal term.
//     No storage cost: the Y_2^0 coefficient along the luminance-weighted L1
//     direction is *inferred* from each channel's |L1|/L0 ratio via a curve
//     fit (paper Sec. 3.4.3, k = sh[0]·(0.08·r + 0.6·r²)). The implicit
//     assumption — that real lighting roughly matches an ambient+directional
//     or sphere-light shape — pulls reconstruction noticeably closer to full
//     SH3 quality. Costs ~12 ALU per probe sample.
// 0 = Plain L1 SH only — original behavior.
#define GI_HALLUCINATE_ZH3 1

// Evaluate L1 SH irradiance for all three channels at once. With
// GI_HALLUCINATE_ZH3 on, also adds the hallucinated ZH3 contribution.
//
// Layout matches gi_voxel_to_sh.hlsl's projection (positive Y_1^m basis):
//   sh.x = L0     sh.y = L1_y     sh.z = L1_z     sh.w = L1_x
// so the "direction toward light" implied by L1 is (sh.w, sh.y, sh.z) with
// no sign flips. The shared zonal axis uses the luminance-weighted L1 vector
// across R/G/B — paper's recommended trick to suppress per-channel ZH3 axes
// drifting apart and producing color fringes.
float3 _gi_eval_sh_rgb(float4 r4, float4 g4, float4 b4, float3 n) {
	float3 base = float3(_gi_eval_sh(r4, n),
	                     _gi_eval_sh(g4, n),
	                     _gi_eval_sh(b4, n));
#if GI_HALLUCINATE_ZH3
	const float3 LUM = float3(0.2126, 0.7152, 0.0722);
	float3 L0   = float3(r4.x, g4.x, b4.x);
	float3 L1_x = float3(r4.w, g4.w, b4.w);
	float3 L1_y = float3(r4.y, g4.y, b4.y);
	float3 L1_z = float3(r4.z, g4.z, b4.z);

	float3 axis    = float3(dot(L1_x, LUM), dot(L1_y, LUM), dot(L1_z, LUM));
	float  axis_l2 = dot(axis, axis);
	if (axis_l2 < 1e-12) return base;          // no L1 signal — ZH3 contributes nothing
	axis *= rsqrt(axis_l2);

	// Per-channel: |projection of channel's L1 onto shared axis| / DC.
	float3 ratio;
	ratio.r = abs(dot(float3(L1_x.r, L1_y.r, L1_z.r), axis));
	ratio.g = abs(dot(float3(L1_x.g, L1_y.g, L1_z.g), axis));
	ratio.b = abs(dot(float3(L1_x.b, L1_y.b, L1_z.b), axis));
	ratio /= max(L0, 1e-6);

	float3 zh3 = L0 * (0.08 * ratio + 0.6 * ratio * ratio);

	// L2 zonal irradiance kernel × basis constant:
	//   A_2 · Y_2^0(fZ) = (π/4) · √(5/(16π)) · (3·fZ² − 1)
	//                   ≈ 0.2477 · (3·fZ² − 1)
	// Matches the un-normalised cosine-kernel convention used by _gi_eval_sh.
	float fZ = dot(axis, n);
	base += zh3 * (0.2477 * (3.0 * fZ * fZ - 1.0));
#endif
	return base;
}

// === Lookup mode ===
// 1 = Tetrahedral 4-tap. Decomposes the surrounding probe cube into 5 Kuhn
//     tetrahedra, picks whichever one contains the sample point, and weights
//     its 4 vertex probes by tetrahedral barycentrics. Halves the probe loads
//     and SH decodes vs. the 8-tap. Interpolation is C0 across tet faces —
//     the C1 discontinuities aren't visible at probe density for diffuse SH.
// 0 = Trilinear 8-tap — full cube; original behavior.
#define GI_TETRAHEDRAL 0

// Per-probe accumulate shared by the directional samplers below. Wrap-shading
// + trap-factor weighting is identical between callers; the only difference
// is which corner they target and the geometric weight `w_tri` they pass in.
void _gi_tap(int3 cell, float w_tri,
             float3 sample_pos, float3 normal, float probe_size,
             inout float3 acc, inout float w_sum)
{
	// Toroidal slot — bit-mask form. PROBE_RES is power-of-two so this is
	// equivalent to `((cell % PROBE_RES) + PROBE_RES) % PROBE_RES` (the
	// two's-complement low bits *are* the positive remainder for any sign
	// of `cell`), at ~13 fewer instructions per axis.
	int3 slot = cell & PROBE_RES_MASK;

	// Wrap-shading directional weight. rsqrt form is safe even when the
	// sample lands exactly on a probe center (length=0).
	float3 probe_w = (float3(cell) + 0.5) * probe_size;
	float3 d       = probe_w - sample_pos;
	float3 to_p    = d * rsqrt(max(dot(d, d), 1e-6));
	float  w_dir   = (dot(to_p, normal) + 1.0) * 0.5;
	w_dir          = w_dir * w_dir + 0.02;

	// Decode the probe + trap factor in one shot (trap is packed into
	// gi_sh_dc.a). Trap zeroes this probe's contribution when it's buried
	// in geometry — (1 - trap) is a smooth multiplier; probes near a
	// surface (low trap) keep ~full weight, probes deep inside (trap → 1)
	// drop out.
	float4 r4, g4, b4_;
	float  trap;
	_gi_decode_probe(slot, r4, g4, b4_, trap);
	float  w  = w_tri * w_dir * (1.0 - trap);
	float3 ir = _gi_eval_sh_rgb(r4, g4, b4_, normal);

	acc   += ir * w;
	w_sum += w;
}

// Directional 8-tap probe sampler. Per-probe barycentric trilinear weights,
// scaled by a "wrap shading" weight that favours probes on the front side of
// the surface, plus a (1 − trap) factor that suppresses probes flagged as
// buried in geometry. Each probe is loaded → ZH3-evaluated → weighted →
// accumulated.
//
// World cell → slot via the same toroidal modulo as the SH compute.
float3 _gi_lookup_directional(float3 sample_pos, float3 normal, float probe_size)
{
	// Probes live at (cell + 0.5) * probe_size in world units, so the lower
	// corner of the 8-tap cube is at world cell `base`.
	float3 p    = sample_pos / probe_size - 0.5;
	int3   base = int3(floor(p));
	float3 f    = p - float3(base);

	float3 acc   = 0;
	float  w_sum = 0;

	[unroll]
	for (int i = 0; i < 8; i++) {
		int3   corner = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
		// Trilinear barycentric weight (lerp picks 1-f or f per axis).
		float3 cw     = lerp(1.0 - f, f, float3(corner));
		float  w_tri  = cw.x * cw.y * cw.z;
		_gi_tap(base + corner, w_tri, sample_pos, normal, probe_size, acc, w_sum);
	}

	return max(0, acc / max(w_sum, 1e-6));
}

// Directional 4-tap probe sampler. Same per-probe work as the 8-tap, but the
// surrounding cube is split into 6 tetrahedra (Freudenthal triangulation —
// one tet per permutation of (x,y,z), all sharing the cube's main diagonal
// 000↔111). The sample point sits in exactly one of those tets; only that
// tet's 4 vertex probes get loaded.
//
// We use Freudenthal rather than the simpler Kuhn 5-tet split because Kuhn
// triangulates each cube face along a diagonal that doesn't agree with the
// neighbour cube on the other side — adjacent cubes end up activating
// disjoint probe sets at the boundary, visible as axis-aligned square seams.
// Freudenthal's face split always runs along the 000↔111-equivalent diagonal,
// so neighbours match and interpolation is C0 across cubes as well as within.
float3 _gi_lookup_directional_tet(float3 sample_pos, float3 normal, float probe_size)
{
	float3 p    = sample_pos / probe_size - 0.5;
	int3   base = int3(floor(p));
	float3 f    = p - float3(base);

	// Sort (fx, fy, fz) descending. The ordering picks one of 6 tets;
	// every tet shares vertices 000 and 111 — the two middle vertices are
	// found by progressively setting the bits of the sorted axes (largest
	// component's axis first). Weights are differences of consecutive
	// sorted components, which is a Schläfli-orthoscheme barycentric: all
	// non-negative iff the ordering is correct, and summing to 1 by
	// telescoping.
	int3 c0 = int3(0,0,0);
	int3 c3 = int3(1,1,1);
	int3 c1, c2;
	float4 w;  // (w_000, w_c1, w_c2, w_111)
	if (f.x >= f.y) {
		if (f.y >= f.z) {           // x ≥ y ≥ z
			c1 = int3(1,0,0); c2 = int3(1,1,0);
			w  = float4(1 - f.x, f.x - f.y, f.y - f.z, f.z);
		} else if (f.x >= f.z) {    // x ≥ z ≥ y
			c1 = int3(1,0,0); c2 = int3(1,0,1);
			w  = float4(1 - f.x, f.x - f.z, f.z - f.y, f.y);
		} else {                    // z ≥ x ≥ y
			c1 = int3(0,0,1); c2 = int3(1,0,1);
			w  = float4(1 - f.z, f.z - f.x, f.x - f.y, f.y);
		}
	} else {
		if (f.x >= f.z) {           // y ≥ x ≥ z
			c1 = int3(0,1,0); c2 = int3(1,1,0);
			w  = float4(1 - f.y, f.y - f.x, f.x - f.z, f.z);
		} else if (f.y >= f.z) {    // y ≥ z ≥ x
			c1 = int3(0,1,0); c2 = int3(0,1,1);
			w  = float4(1 - f.y, f.y - f.z, f.z - f.x, f.x);
		} else {                    // z ≥ y ≥ x
			c1 = int3(0,0,1); c2 = int3(0,1,1);
			w  = float4(1 - f.z, f.z - f.y, f.y - f.x, f.x);
		}
	}

	float3 acc   = 0;
	float  w_sum = 0;
	_gi_tap(base + c0, w.x, sample_pos, normal, probe_size, acc, w_sum);
	_gi_tap(base + c1, w.y, sample_pos, normal, probe_size, acc, w_sum);
	_gi_tap(base + c2, w.z, sample_pos, normal, probe_size, acc, w_sum);
	_gi_tap(base + c3, w.w, sample_pos, normal, probe_size, acc, w_sum);

	return max(0, acc / max(w_sum, 1e-6));
}

// Sample the SH probe grid at world position, evaluate against the surface
// normal, and return the resulting irradiance scaled by gi_intensity.
//
// With toroidal addressing under Wrap, anything sampled outside the active
// window pulls in content from a *different* world cell that happens to share
// the same modular slot — visible as a player-relative colored splotch ~half
// the volume away. We avoid that here by fading to sk_lighting (the sky SH
// from Renderer.SkyLight) over the half-probe strip nearest each edge, and
// using pure sky beyond the volume. Inside the safe core, GI is full; at the
// very edge it's pure sky; smoothly blended between.
float3 GISampleIndirect(float3 world_pos, float3 normal) {
	// Volume can be non-cubic (64×32×64 probes → 96×48×96 m at 1.5 m spacing),
	// so volume_extent is per-axis. Probe spacing stays uniform because the
	// probe count and volume size scale together — use any axis to derive it.
	float3 volume_extent = 1.0 / gi_volume_inv;
	float  probe_size    = volume_extent.x / (float)PROBE_RES_X;
	float  half_probe    = probe_size * 0.5;
	// Receiver normal bias. Originally pushed the lookup point off the surface
	// by half a probe to distance-suppress buried probes — trap detection now
	// handles that case precisely, so we no longer need a large bias. What
	// remains is "self-intersection insurance": a small offset keeps the
	// lookup point reliably on the front side of the surface despite tiny
	// imprecisions in interpolated vertex normals, so the directional 8-tap
	// doesn't flip-flop its front/back classification for fragments straddling
	// the exact-zero plane.
	float3 sample_pos    = world_pos + normal * (half_probe * 0.25);
	float3 to_min        = sample_pos - gi_volume_min;
	float3 sky           = sk_lighting(normal);

	// Beyond the volume entirely: pure sky ambient.
	if (any(to_min < 0) || any(to_min > volume_extent)) return sky;

	// Fully mask the bad-blend zone with sky, then fade GI in over the next
	// half-probe. The bad-blend zone (within half-probe of each edge) is
	// where toroidal addressing's slot adjacency disagrees with world-cell
	// adjacency — sampling there blends in content from the opposite side
	// of the volume. So:
	//   to_min ∈ [0,        half_probe]   → fade=0 (pure sky)
	//   to_min ∈ [half_probe, probe_size] → fade ramps 0→1 (sky → GI)
	//   to_min ∈ [probe_size, …]          → fade=1 (pure GI)
	// Same logic mirrored at the upper edge.
	float3 fade_lo = saturate((to_min            - half_probe) / half_probe);
	float3 fade_hi = saturate((volume_extent - to_min - half_probe) / half_probe);
	float  fade    = min(min(min(fade_lo.x, fade_lo.y), fade_lo.z),
	                     min(min(fade_hi.x, fade_hi.y), fade_hi.z));
	if (fade <= 0) return sky;

#if GI_TETRAHEDRAL
	float3 gi = _gi_lookup_directional_tet(sample_pos, normal, probe_size);
#else
	float3 gi = _gi_lookup_directional(sample_pos, normal, probe_size);
#endif
	gi *= gi_intensity;
	return lerp(sky, gi, fade);
}

#include <stereokit.hlsli>
#include "common.hlsli"
#include "lighting.hlsli"
#include "gi.hlsli"

//--name = app/gi_voxelize
// Voxelize variant for the GI pipeline. Renders the scene from 3 axis-aligned
// ortho cameras (X, Y, Z) in a single multi-view pass; the pixel shader uses
// ddx/ddy to recover the geometric face normal, discards fragments whose
// dominant axis doesn't match the current view, and writes the lit color of
// the fragment into a 3D RWTexture UAV.
//
// Adapted from the StereoKit GI demo. Project-specific changes:
//   - Captures **dynamic-light contribution** (lighting.hlsli) plus the **sun
//     directional + shadow** (from common.hlsli). Interior cells have a zero
//     shadow_light_color so the sun term contributes nothing there; exterior
//     gets sun bounce. Cell ambient is intentionally left out so the SH
//     represents *bounced light only* and the main pass adds it on top of the
//     direct ambient without double-counting.
//   - **Multi-bounce feedback**: samples last frame's GI (via gi.hlsli's
//     GISampleIndirect) per vertex and folds it into the lit color written
//     into the voxel grid. Each frame the voxels capture one additional
//     bounce, so over a few frames the system converges to a fully multi-
//     bounce GI solution. The feedback gain is BOUNCE_STRENGTH below.
//   - GIBuffer + SH textures come from gi.hlsli (shared with the consumer
//     and SH compute).

// Multi-bounce feedback gain. 1.0 = use last frame's GI verbatim as input
// for this frame's voxel lit color. Drop below 1.0 if albedo-1.0 surfaces
// drive runaway feedback; values around 0.5–0.8 are a safer default if the
// scene has saturated bright surfaces.
#define BOUNCE_STRENGTH 1.0

// Boost factor for self-illumination written into the voxel grid as a bounce
// source. 1.0 = use the material's emissive verbatim. Raise above 1.0 to make
// glowing surfaces (lantern panes, candles, magic effects) cast more indirect
// light without affecting their own surface brightness in the lit pass.
#define EMISSIVE_STRENGTH 20.0

//--color:color = 1,1,1,1
//--tex_trans   = 0,0,1,1
//--emissive    = 0,0,0,0
//--diffuse     = white
float4       color;
float4       tex_trans;
float4       emissive;   // self-illumination, modulated by albedo at write time so the voxel grid captures glowing surfaces as bounce sources
float4       _pad1;
Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0);

[[vk::image_format("rgba8")]]
RWTexture3D<unorm float4> voxel_tex : register(u0);

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos          : SV_Position;
	float3 world_pos    : TEXCOORD0;
	float3 normal       : NORMAL0;
	float2 uv           : TEXCOORD1;
	float4 color        : COLOR0;
	float3 shadow_uv    : TEXCOORD2;
	float  shadow_ndotl : TEXCOORD3;
	// GI bounce from last frame, sampled per-vertex (SH is low frequency,
	// safe to linearly interpolate). Dyn-light direct is per-pixel in the
	// PS — point lights are localized enough that a per-vertex sample misses
	// any light whose radius doesn't reach a vertex on a coarse triangle.
	float3 gi_bounce    : TEXCOORD5;
	nointerpolation uint view_axis : TEXCOORD4;
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;
	float4 world = mul(input.pos, sk_inst[ids.inst].world);
	o.pos        = mul(world,     sk_viewproj[ids.view]);
	o.world_pos  = world.xyz;
	o.normal     = normalize(mul(input.norm, (float3x3)sk_inst[ids.inst].world));
	o.uv         = input.uv * tex_trans.zw + tex_trans.xy;
	o.color      = color * input.col * sk_inst[ids.inst].color;
	o.shadow_uv  = ShadowVS(world.xyz, o.normal, o.shadow_ndotl);
	// Fold emissive into the per-vertex precomputed term so the voxel write
	// in the PS picks it up alongside GI bounce. Matches mw_lit's `indirect`
	// pattern, just under the voxelize's `gi_bounce` name.
	o.gi_bounce  = GISampleIndirect(world.xyz, o.normal) * BOUNCE_STRENGTH
	             + emissive.rgb * EMISSIVE_STRENGTH;
	o.view_axis  = ids.view; // 0=X, 1=Y, 2=Z
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	// Geometric face normal via screen-space derivatives, then dominant
	// axis selection — only the matching view writes this triangle.
	float3 face_n = normalize(cross(ddx(input.world_pos), ddy(input.world_pos)));
	float3 abs_n  = abs(face_n);
	uint   dominant = (abs_n.x >= abs_n.y && abs_n.x >= abs_n.z) ? 0
	                : (abs_n.y >= abs_n.z) ? 1 : 2;
	if (dominant != input.view_axis) discard;

	// Toroidal voxel write. World cell = floor(world_pos / voxel_size); the
	// destination texel is that cell mod voxel_res, per-axis. Fragments whose
	// world cell isn't in the current active window are discarded so we don't
	// stamp data into slots that belong to other side of the wrap. Volume can
	// be non-cubic (256×128×256 voxels) but spacing is uniform, so voxel_size
	// is still a single scalar derived from any axis.
	float voxel_size = (1.0 / gi_volume_inv.x) / (float)VOXEL_RES_X;
	int3  origin_vox = int3(floor(gi_volume_min / voxel_size));
	int3  world_vox  = int3(floor(input.world_pos / voxel_size));
	int3  win_local  = world_vox - origin_vox;
	if (any(win_local < 0) || any(win_local >= VOXEL_RES)) discard;
	// Bit-mask form of `((world_vox % VOXEL_RES) + VOXEL_RES) % VOXEL_RES` —
	// each axis res is a power of two so the two's-complement low bits *are*
	// the positive remainder for any sign of world_vox.
	int3  tx         = world_vox & VOXEL_RES_MASK;

	float4 albedo  = diffuse.Sample(diffuse_s, input.uv) * input.color;
	// Cheap single-tap shadow — voxel detail beyond ~1 m is wasted, no point
	// paying for 4-tap PCF + edge fade here.
	float  sun_f   = ShadowPS1Tap(input.shadow_uv, input.shadow_ndotl);
	float3 sun_lit = sun_f * shadow_light_color;
	// Direct point-light contribution comes through input.gi_bounce now (the
	// probe SH carries dyn-light + visibility from NEE in gi_voxel_to_sh).
	// Sampling AccumulateLights here would double-count, so we omit it.

	float3 lit = albedo.rgb * (sun_lit + input.gi_bounce);

	// Store lit / voxel_range so unorm rgba8 holds values up to voxel_range
	// (instead of clipping at 1.0). SH compute multiplies back on read. The
	// comparison still operates on the unscaled magnitudes (v_new built
	// from unclamped scaled value) so brightest-wins ordering is preserved.
	float3 v_stored = lit / gi_voxel_range_pad.x;

	// Always write surfaces (alpha = 1 = "this cell contains geometry") so
	// unlit/back-facing fragments still appear as occluders for GI rays.
	// When multiple triangles target the same voxel, prefer the brightest
	// sample as a tiebreaker.
	float4 v_new  = float4(v_stored, 1.0);
	float4 v_prev = voxel_tex[tx];
	if (v_prev.a < 0.5 || dot(v_new.rgb, v_new.rgb) > dot(v_prev.rgb, v_prev.rgb))
		voxel_tex[tx] = v_new;

	return float4(0, 0, 0, 0);
}

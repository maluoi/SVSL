cbuffer FogBuffer : register(b3) {
	float4 fog_color;   // legacy: horizon-sampled fog color. Still consumed by sky.hlsl's sky-dome horizon haze; mw_lit / mw_water use FogScatter (below ShadowBuffer) for view-directional scatter.
	float4 fog_data;    // x = extinction coefficient (per meter), y = unused, z = sky horizon band, w = sky reference distance
};

// Pure Beer-Lambert extinction: 1 - exp(-density * dist). No start offset, no
// end clamp — those framings tied "how dense is the fog" to "where does the
// lerp saturate", which forced an unphysically thick atmosphere whenever the
// scene wanted distant geometry to fade. Real atmospheric extinction is just
// σ·d, with σ varying by air quality:
//
//   σ ≈ 0.00002 /m   clear air        (50 km characteristic length)
//   σ ≈ 0.0005  /m   light haze       (2 km)
//   σ ≈ 0.002   /m   moderate haze    (500 m)
//   σ ≈ 0.01    /m   meteorological fog (100 m)
//   σ ≈ 0.1     /m   dense fog        (10 m)
//
// Tune via Fog.SetDensity / Fog.DefaultDensity in the C# side. σ = 0 disables
// fog entirely (used for interior cells), reaching exp(0) = 1, so 1 - 1 = 0.
float FogExtinction(float3 view_pt) {
	float density = fog_data.x;
	return 1.0 - exp(-density * length(view_pt));
}

// Backwards-compat alias. The name was misleading — it returns the fog
// FACTOR, not a color. New shaders should use FogExtinction; old callers
// (sky.hlsl) still work without churn.
float FogColor(float3 view_pt) { return FogExtinction(view_pt); }

///////////////////////////////////////////
// Shadow mapping
///////////////////////////////////////////

cbuffer ShadowBuffer : register(b13) {
	float4x4 shadow_transform;
	float3   shadow_light_dir;
	float    shadow_bias;
	float3   shadow_light_color;
	float    shadow_pixel_size;
};
Texture2D              shadow_map   : register(t13);
SamplerComparisonState shadow_map_s : register(s13);

///////////////////////////////////////////
// Fog scatter color (depends on ShadowBuffer)
///////////////////////////////////////////

// Directional in-scatter color for fog at `view_dir_world` (normalized, world
// space, pointing AWAY from the camera — toward the fragment). Two terms:
//
//   - Sky: sk_lighting(view_dir) — the L2 sky-SH evaluated in the view
//     direction. This is what the viewer would see through arbitrarily-thick
//     fog at infinity, matching the sky color behind it. Closes the artifact
//     where a single horizon-sampled fog tint would read orange-blue while
//     looking straight up at a noon zenith.
//
//   - Sun: Henyey-Greenstein forward scatter. Adds a bright halo around the
//     sun direction, which is what atmospheric mist actually does — particles
//     scatter incident light preferentially forward. g=0.7 sits in the
//     mist/fog anisotropy band; HG_STRENGTH tames the forward peak so it
//     reads as glow rather than blowout. shadow_light_dir is "to the sun" so
//     dot(view_dir, sun_dir) is +1 looking AT the sun and -1 looking away.
//
// Composed into the surface color the same way the old single-tint fog was:
//   col = lerp(col, FogScatter(view_dir), FogExtinction(view_pt))
// which is mathematically `col * transmittance + scatter * (1 - transmittance)`
// — Beer-Lambert composition with the surface color absorbing through the
// medium and the fog's own scatter color filling in the rest.
float3 FogScatter(float3 view_dir_world) {
	float3 sky = sk_lighting(view_dir_world);

	const float g           = 0.7;
	const float HG_STRENGTH = 0.25;
	float cos_theta = dot(view_dir_world, shadow_light_dir);
	float denom = max(1.0 + g*g - 2.0*g*cos_theta, 1e-3);
	float phase = (1.0 - g*g) / pow(denom, 1.5);
	float3 sun_glow = shadow_light_color * (phase * HG_STRENGTH);

	return sky + sun_glow;
}

// 4-tap PCF. Each tap is a hardware 2×2 PCF with footprint ~1 texel wide;
// a 1.5-texel sample spacing keeps adjacent footprints disjoint regardless
// of where the lookup UV lands relative to texel boundaries. The previous
// 0.5-texel spacing meant the 4 footprints all overlapped on the central
// texel — counting it 4× — collapsing the filter to an effective 3×3
// binomial. With h = 1.5 the 4 footprints cover up to 16 unique texels
// (with a small unsampled cross-gap in the middle, accepted as a trade
// since the alternative is double-counting).
float ShadowFactorPCF2(float3 uv) {
	float h = shadow_pixel_size * 0.5;
	float s0 = shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2(-h, -h), uv.z);
	float s1 = shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2( h, -h), uv.z);
	float s2 = shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2(-h,  h), uv.z);
	float s3 = shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2( h,  h), uv.z);
	return (s0 + s1 + s2 + s3) * 0.25;
}

// 9-tap 3×3 PCF with Gaussian weighting (1-2-1 / 2-4-2 / 1-2-1, normalized by
// 1/16). The center-heavy falloff softens the square box-filter look of a flat
// average into a rounder penumbra.
float ShadowFactorPCF3(float3 uv) {
	float h = shadow_pixel_size;
	// Gaussian weights: corners 1, edges 2, center 4 — sum 16.
	const float wc = 4.0 / 16.0;
	const float we = 2.0 / 16.0;
	const float wk = 1.0 / 16.0;
	float sum = 0;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2(-h, -h), uv.z) * wk;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2( 0, -h), uv.z) * we;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2( h, -h), uv.z) * wk;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2(-h,  0), uv.z) * we;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2( 0,  0), uv.z) * wc;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2( h,  0), uv.z) * we;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2(-h,  h), uv.z) * wk;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2( 0,  h), uv.z) * we;
	sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + float2( h,  h), uv.z) * wk;
	return sum;
}

// Wider 8-tap Poisson-disk PCF. Each tap is a hardware 2×2 PCF; the 8 tap
// centers are spread across a unit disk via a precomputed Poisson pattern,
// scaled by SHADOW_SOFT_RADIUS shadow texels. The irregular layout dithers
// the shadow-edge stair-stepping into noise the eye reads as smooth penumbra
// rather than discrete pixel jumps.
#define SHADOW_SOFT_RADIUS 3.0
float ShadowFactorPCFSoft(float3 uv) {
	const float2 poisson_disk[8] = {
		float2(-0.94201624, -0.39906216),
		float2( 0.94558609, -0.76890725),
		float2(-0.09418410, -0.92938870),
		float2( 0.34495938,  0.29387760),
		float2(-0.91588581,  0.45771432),
		float2(-0.81544232, -0.87912464),
		float2(-0.38277543,  0.27676845),
		float2( 0.97484398,  0.75648379),
	};
	float r   = shadow_pixel_size * SHADOW_SOFT_RADIUS;
	float sum = 0;
	[unroll] for (int i = 0; i < 8; i++)
		sum += shadow_map.SampleCmpLevelZero(shadow_map_s, uv.xy + poisson_disk[i] * r, uv.z);
	return sum * 0.125;
}

// Compute shadow UV and N dot L in the vertex shader.
// Returns shadow map UV; ndotl is written via out parameter.
float3 ShadowVS(float3 world_pos, float3 normal, out float ndotl) {
	ndotl = dot(normal, shadow_light_dir);
	float slope = saturate(min(1, sqrt(1 - ndotl * ndotl) / max(ndotl, 0.001)));
	float3 bias = normal * (shadow_bias * slope);
	float4 sp   = mul(float4(world_pos + bias, 1), shadow_transform);
	float3 uv   = float3(sp.xy, sp.z / sp.w);
	uv.xy = uv.xy * float2(0.5, -0.5) + 0.5;
	return uv;
}

// Evaluate shadow in the pixel shader. Returns directional light contribution.
float ShadowPS(float3 shadow_uv, float ndotl) {
	if (ndotl <= 0) return 0;
	// Fade to fully lit at shadow map edges
	float2 edge = saturate(min(shadow_uv.xy, 1 - shadow_uv.xy) * 10);
	float  fade = edge.x * edge.y;
	float  shadow = lerp(1, ShadowFactorPCF3(shadow_uv), fade);
	return min(ndotl, shadow);
}

// Cheap single-tap shadow lookup for passes where shadow-edge detail is
// wasted (voxelize at 0.5 m voxel size, low-LOD passes, etc). Skips the
// 4-tap PCF blur and the smooth-edge-fade — outside the shadow map just
// returns full ndotl. SampleCmpLevelZero still benefits from the sampler's
// hardware 2x2 PCF, so this isn't a true "hard" shadow, just much cheaper.
float ShadowPS1Tap(float3 shadow_uv, float ndotl) {
	if (ndotl <= 0) return 0;
	if (any(shadow_uv.xy < 0) || any(shadow_uv.xy > 1)) return ndotl;
	float shadow = shadow_map.SampleCmpLevelZero(shadow_map_s, shadow_uv.xy, shadow_uv.z);
	return min(ndotl, shadow);
}

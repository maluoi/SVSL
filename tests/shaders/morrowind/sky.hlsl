#include <stereokit.hlsli>
#include "common.hlsli"

///////////////////////////////////////////

//--sun_dir      = {0.5, 0.7, 0.5, 0}
//--ground_color = {0.02, 0.02, 0.03, 1}
//--rayleigh     = {5.8e-6, 13.5e-6, 33.1e-6, 7994}
//--mie          = {21e-6, 1200, 0.76, 0}
//--sun_info     = {20, 8, 0, 0}
float4 sun_dir;
float4 ground_color;
float4 rayleigh;  // xyz = beta_r, w = Hr
float4 mie;       // x = beta_m, y = Hm, z = g
float4 sun_info;  // x = I_sun, y = exposure

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos   : SV_POSITION;
	float3 color : TEXCOORD0;
	float  fog   : TEXCOORD1;
};

///////////////////////////////////////////

#define PI 3.141592653589793

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;

	float3 dir = normalize(input.pos.xyz);

	// View rotation only (no translation) — locks dome around head
	float4x4 rot_view = sk_view[ids.view];
	rot_view._m30 = 0;
	rot_view._m31 = 0;
	rot_view._m32 = 0;
	o.pos   = mul(mul(float4(dir, 1), rot_view), sk_proj[ids.view]);
	o.pos.z = o.pos.w; // just inside far plane to avoid z-fight with clear

	// Atmospheric scattering (single-scatter analytical approximation)
	float3 sun = normalize(sun_dir.xyz);

	// Always compute scattering from a direction at or above the horizon
	float3 sky_dir = dir;
	sky_dir.y      = max(sky_dir.y, 0.02);
	sky_dir        = normalize(sky_dir);
	float  mu      = dot(sky_dir, sun);

	// Scattering coefficients from uniforms
	float3 beta_r = rayleigh.xyz;
	float  beta_m = mie.x;
	float  Hr     = rayleigh.w;
	float  Hm     = mie.y;
	float  g      = mie.z;
	float  I_sun  = sun_info.x;

	// Rayleigh phase function
	float phase_r = (3.0 / (16.0 * PI)) * (1.0 + mu * mu);

	// Mie phase function (Henyey-Greenstein)
	float g2      = g * g;
	float phase_m = (3.0 / (8.0 * PI))
		* ((1.0 - g2) * (1.0 + mu * mu))
		/ ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));

	// Optical depth: scale_height / cos(zenith) per component
	float  cos_view = max(sky_dir.y, 0.01);
	float  cos_sun  = max(sun.y, 0.01);
	float3 tau      = beta_r * Hr + beta_m * Hm;

	float3 ext_view = exp(-tau / cos_view);
	float3 ext_sun  = exp(-tau / cos_sun);

	// In-scattered light along the view ray
	float3 scatter = I_sun * ext_sun
		* (beta_r * phase_r + beta_m * phase_m)
		/ (beta_r + beta_m)
		* (1.0 - ext_view);

	// Tone map (exposure)
	float3 sky_color = 1.0 - exp(-scatter * sun_info.y);

	// Below the horizon: blend from horizon color down to dark ground
	float below = smoothstep(0.0, 0.75, -dir.y); // eases in at horizon, full ground by ~30° below
	o.color = lerp(sky_color, ground_color.rgb, below);

	// Fog at the horizon: sky at far distance, fading out as we look upward
	float fog_str = FogColor(float3(0, 0, fog_data.w));
	o.fog = fog_str * (1.0 - smoothstep(0.0, fog_data.z, dir.y));
	return o;
}

///////////////////////////////////////////

float4 ps(psIn input) : SV_TARGET {
	float3 color = lerp(input.color, fog_color.rgb, input.fog);
	return float4(color, 1);
}

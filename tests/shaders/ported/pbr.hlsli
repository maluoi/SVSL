#ifndef _SK_RENDERER_PBR_HLSLI
#define _SK_RENDERER_PBR_HLSLI

#include "common.hlsli"

///////////////////////////////////////////
// PBR Helper Functions
///////////////////////////////////////////

// Calculate mip level based on screen-space derivatives
float pbr_mip_level(float ndotv) {
	float2 dx    = ddx(ndotv * cubemap_info.x);
	float2 dy    = ddy(ndotv * cubemap_info.y);
	float  delta = max(dot(dx, dx), dot(dy, dy));
	return 0.5 * log2(delta);
}

///////////////////////////////////////////

// Fresnel-Schlick approximation with roughness
float3 pbr_fresnel_schlick_roughness(float ndotv, float3 F0, float roughness) {
	return F0 + (max(1 - roughness, F0) - F0) * pow(1.0 - ndotv, 5.0);
}

///////////////////////////////////////////

// BRDF approximation for split-sum IBL
// See: https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
float2 pbr_brdf_appx(float roughness, float ndotv) {
	const float4 c0   = { -1, -0.0275, -0.572,  0.022 };
	const float4 c1   = {  1,  0.0425,  1.04,  -0.04  };
	float4       r    = roughness * c0 + c1;
	float        a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
	float2       AB   = float2(-1.04, 1.04) * a004 + r.zw;
	return AB;
}

///////////////////////////////////////////

// Main PBR shading function
// Returns final lit color with alpha
float4 pbr_shade(
	float4 albedo,
	float3 irradiance,
	float  ao,
	float  metallic,
	float  roughness,
	float3 view_dir,
	float3 surface_normal,
	TextureCube env_map,
	SamplerState env_sampler
) {
	float3 view       = normalize(view_dir);
	float3 reflection = reflect(-view, surface_normal);
	float  ndotv      = max(0, dot(surface_normal, view));

	// Reduce specular aliasing by capping roughness at glancing angles
	// See Advanced VR Rendering p43 by Alex Vlachos:
	// https://gdcvault.com/play/1021772/Advanced-VR
	float3 norm_ddx        = ddx(surface_normal.xyz);
	float3 norm_ddy        = ddy(surface_normal.xyz);
	float  geometric_rough = pow(saturate(max(dot(norm_ddx.xyz, norm_ddx.xyz), dot(norm_ddy.xyz, norm_ddy.xyz))), 0.45);
	roughness = max(roughness, geometric_rough);

	// Calculate Fresnel reflectance at normal incidence
	float3 F0 = lerp(0.04, albedo.rgb, metallic);
	float3 F  = pbr_fresnel_schlick_roughness(ndotv, F0, roughness);
	float3 kS = F;

	// Sample environment map at appropriate mip level based on roughness
	float mip = (1 - pow(1 - roughness, 2)) * cubemap_info.z;
	mip = max(mip, pbr_mip_level(ndotv));
	float3 prefilteredColor = env_map.SampleLevel(env_sampler, reflection, mip).rgb;
	float2 envBRDF          = pbr_brdf_appx(roughness, ndotv);
	float3 specular         = prefilteredColor * (F * envBRDF.x + envBRDF.y);

	// Energy conservation - what's not reflected is refracted (diffuse)
	float3 kD = 1 - kS;
	kD *= 1.0 - metallic;  // Metals have no diffuse lighting

	// Combine diffuse and specular with ambient occlusion
	float3 diffuse = albedo.rgb * irradiance * ao;
	float3 color   = (kD * diffuse + specular * ao);

	return float4(color, albedo.a);
}

///////////////////////////////////////////

#endif

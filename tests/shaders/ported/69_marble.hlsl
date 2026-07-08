//--name = marble
//--tex = white

#include "common.hlsli"

// Glass marble - transparent sphere showing environment through refraction

float4 color           = float4(0.98, 1.0, 0.98, 1);  // Slight green tint (glass)
float4 tex_trans       = float4(0, 0, 1, 1);
float  ior             = 1.3;                        // Lower IOR for subtler refraction
float  chromatic       = 0.015;                       // Subtle chromatic aberration
float  internal_glow   = 0.02;                        // Minimal internal caustic

struct Inst {
	float4x4 world;
};
StructuredBuffer<Inst> inst : register(t2, space0);

struct vsIn {
	float3 pos   : SV_POSITION;
	float3 norm  : NORMAL;
	float2 uv    : TEXCOORD0;
	float4 color : COLOR0;
};
struct psIn {
	float4 pos       : SV_POSITION;
	float3 normal    : NORMAL;
	float2 uv        : TEXCOORD0;
	float4 color     : COLOR0;
	float3 world_pos : TEXCOORD1;
	uint   layer     : SV_RenderTargetArrayIndex;
};

Texture2D    tex             : register(t3);
SamplerState tex_sampler     : register(s3);
TextureCube  environment_map : register(t4);
SamplerState env_sampler     : register(s4);

psIn vs(vsIn input, uint id : SV_InstanceID) {
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	float4 world_pos = mul(float4(input.pos, 1), inst[inst_idx].world);

	psIn output;
	output.pos       = mul(world_pos, viewproj[view_idx]);
	output.normal    = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;
	output.world_pos = world_pos.xyz;
	output.layer     = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	float3 normal = normalize(input.normal);
	float3 view_dir = normalize(cam_pos[input.layer].xyz - input.world_pos);

	float ndotv = saturate(dot(view_dir, normal));

	// Schlick's Fresnel with very low base reflectance for crystal clarity
	float f0 = 0.02;
	float fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, 5.0);

	// Clamp fresnel low to minimize reflections - crystal balls are see-through
	fresnel = min(fresnel, 0.15);

	// Reflection (very subtle, soft, mostly at edges)
	float3 reflect_dir = reflect(-view_dir, normal);
	float3 reflect_color = environment_map.SampleLevel(env_sampler, reflect_dir, 2.0).rgb;

	// Crystal ball effect - see through to what's behind, with lens distortion
	// The sphere acts like a lens, inverting and distorting the background
	float3 through_dir = -view_dir;  // Direction to what's behind the sphere

	// Add lens-like distortion: rays bend toward center based on where they hit
	float distortion = (1.0 - ndotv) * (ior - 1.0);
	float3 lens_dir = normalize(lerp(through_dir, -normal, distortion));

	// Chromatic aberration - different wavelengths bend slightly differently
	float3 lens_r = normalize(lerp(through_dir, -normal, distortion * (1.0 - chromatic)));
	float3 lens_g = lens_dir;
	float3 lens_b = normalize(lerp(through_dir, -normal, distortion * (1.0 + chromatic)));

	// Sample environment through the crystal ball
	float r = environment_map.SampleLevel(env_sampler, lens_r, 0.0).r;
	float g = environment_map.SampleLevel(env_sampler, lens_g, 0.0).g;
	float b = environment_map.SampleLevel(env_sampler, lens_b, 0.0).b;
	float3 refract_color = float3(r, g, b);

	// Apply glass tint
	refract_color *= input.color.rgb;

	// Fake internal caustics - brighten where rays converge (center of sphere)
	float caustic = pow(ndotv, 2.0) * internal_glow;
	refract_color += caustic;

	// Blend: mostly refraction, slight reflection at edges
	float3 final_color = lerp(refract_color, reflect_color, fresnel);

	return float4(final_color, 1.0);
}

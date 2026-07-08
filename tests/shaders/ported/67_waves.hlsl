//--name = waves
//--tex = white

#include "common.hlsli"

// Animated wave displacement with multiple octaves

float4 color        = float4(0.2, 0.4, 0.8, 1);
float4 foam_color   = float4(1.0, 1.0, 1.0, 1);
float4 tex_trans    = float4(0, 0, 1, 1);
float  wave_height  = 0.2;
float  wave_speed   = 1.5;
float  wave_scale   = 3.0;

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
	float  wave      : TEXCOORD2; // wave height for foam
	uint   layer     : SV_RenderTargetArrayIndex;
};

Texture2D    tex         : register(t3);
SamplerState tex_sampler : register(s3);

// Wave function
float wave(float2 pos, float t) {
	float w = 0.0;
	// Multiple wave octaves
	w += sin(pos.x * wave_scale + t * wave_speed) * 0.5;
	w += sin(pos.y * wave_scale * 1.3 + t * wave_speed * 0.7) * 0.3;
	w += sin((pos.x + pos.y) * wave_scale * 0.8 + t * wave_speed * 1.2) * 0.2;
	return w * wave_height;
}

// Wave gradient for normal calculation
float3 wave_normal(float2 pos, float t) {
	float eps = 0.01;
	float h0 = wave(pos, t);
	float hx = wave(pos + float2(eps, 0), t);
	float hy = wave(pos + float2(0, eps), t);

	float3 tx = normalize(float3(eps, hx - h0, 0));
	float3 ty = normalize(float3(0, hy - h0, eps));

	return normalize(cross(ty, tx));
}

psIn vs(vsIn input, uint id : SV_InstanceID) {
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	// Apply wave displacement in local space
	float3 local_pos = input.pos;
	float w = wave(local_pos.xz, time);
	local_pos.y += w;

	float4 world_pos = mul(float4(local_pos, 1), inst[inst_idx].world);

	// Calculate wave normal
	float3 wave_n = wave_normal(input.pos.xz, time);
	float3 world_normal = normalize(mul(float4(wave_n, 0), inst[inst_idx].world).xyz);

	psIn output;
	output.pos       = mul(world_pos, viewproj[view_idx]);
	output.normal    = world_normal;
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color * color;
	output.world_pos = world_pos.xyz;
	output.wave      = w;
	output.layer     = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	const float3 light_dir = normalize(float3(1, 3, 2));
	const float3 view_dir = normalize(cam_pos[input.layer].xyz - input.world_pos);

	float3 normal = normalize(input.normal);

	// Diffuse
	float ndotl = saturate(dot(normal, light_dir));

	// Specular (Blinn-Phong)
	float3 half_dir = normalize(light_dir + view_dir);
	float spec = pow(saturate(dot(normal, half_dir)), 64.0);

	// Fresnel for water
	float fresnel = pow(1.0 - saturate(dot(view_dir, normal)), 3.0);

	// Sample texture
	float4 albedo = tex.Sample(tex_sampler, input.uv) * input.color;

	// Add foam on wave peaks
	float foam = smoothstep(wave_height * 0.6, wave_height, input.wave);
	albedo = lerp(albedo, foam_color, foam * 0.5);

	// Final color
	float3 final_color = albedo.rgb * (0.3 + ndotl * 0.5);
	final_color += spec * 0.5;
	final_color = lerp(final_color, float3(0.5, 0.7, 1.0), fresnel * 0.3);

	return float4(final_color, albedo.a);
}

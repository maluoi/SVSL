//--name = orbital_particles

#include "common.hlsli"

float3 color_slow;
float  max_speed;
float3 color_fast;
float  _pad2;

struct Particle {
	float3 position;
	float3 velocity;
};
StructuredBuffer<Particle> particles : register(t3, space0);

struct vsIn {
	float3 pos  : SV_POSITION;
	float3 norm : NORMAL;
	float2 uv   : TEXCOORD0;
	float4 color: COLOR0;
};
struct psIn {
	float4 pos   : SV_POSITION;
	float3 color : COLOR0;
};

psIn vs(vsIn input, uint id : SV_InstanceID) {
	const float3 light_dir = normalize(float3(1, 4, 2));

	// Multi-view instancing: extract particle index and view index
	uint particle_idx = id / view_count;
	uint view_idx = id % view_count;

	psIn output;
	// Apply only translation from particle buffer - no rotation or scale matrix needed
	float3 world_pos = input.pos.xyz * 0.02 + particles[particle_idx].position;
	output.pos = mul(float4(world_pos, 1), viewproj[view_idx]);

	// Simple lighting with vertex normals
	float3 normal = normalize(input.norm);
	float ndotl = saturate(dot(normal, light_dir));

	// Color based on particle speed
	float speed   = length(particles[particle_idx].velocity);
	float speed_t = saturate(speed / max_speed) * 2;
	float3 base_color = lerp(color_slow, color_fast, speed_t);
	output.color = ndotl * base_color;

	return output;
}
float4 ps(psIn input) : SV_TARGET {
	return float4(input.color, 1);
}

// Particle orbital physics compute shader

struct Particle {
	float3 position;
	float3 velocity;
};

float  time;
float  delta_time;
float  damping;
float  max_speed;
float  strength;
uint   particle_count;

StructuredBuffer  <Particle> input  : register(t1);
RWStructuredBuffer<Particle> output : register(u2);

[numthreads(256, 1, 1)]
void cs(uint3 dispatchThreadID : SV_DispatchThreadID) {
	uint id = dispatchThreadID.x;
	if (id >= particle_count) return;

	// Define moving attractors (same as CPU version)
	float3 attractors[3];
	attractors[0] = float3(cos(time * 0.5) * 2.0, sin(time * 0.7) * 1.5, sin(time * 0.5) * 2.0);
	attractors[1] = float3(sin(time * 0.6) * 2.5, cos(time * 0.4) * 2.0, cos(time * 0.6) * 1.5);
	attractors[2] = float3(cos(time * 0.4) * 1.5, sin(time * 0.8) * 2.5, sin(time * 0.3) * 2.0);

	Particle p = input[id];
	float3 force = float3(0, 0, 0);

	// Find nearest 2 attractors
	float dist1 = 1e10, dist2 = 1e10;
	int idx1 = 0, idx2 = 1;

	for (int a = 0; a < 3; a++) {
		float3 diff = attractors[a] - p.position;
		float dist = length(diff);

		if (dist < dist1) {
			dist2 = dist1;
			idx2  = idx1;
			dist1 = dist;
			idx1  = a;
		} else if (dist < dist2) {
			dist2 = dist;
			idx2  = a;
		}
	}

	// Apply gravitational force from nearest 2 attractors
	for (int a = 0; a < 2; a++) {
		int idx = (a == 0) ? idx1 : idx2;
		float3 diff = attractors[idx] - p.position;
		float dist_sq = dot(diff, diff) + 0.1;
		float dist = sqrt(dist_sq);

		float3 direction = diff / dist;
		float force_mag = strength / dist_sq;
		force += direction * force_mag;
	}

	// Update velocity and position
	p.velocity += force * delta_time;
	p.velocity *= damping;

	// Clamp velocity
	float speed = length(p.velocity);
	if (speed > max_speed) {
		p.velocity *= max_speed / speed;
	}

	p.position += p.velocity * delta_time;

	output[id] = p;
}

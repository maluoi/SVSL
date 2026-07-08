//--name = granite
//--tex = white

#include "common.hlsli"

// Granite shader - speckled stone with veins and crystalline structure

float4 color1        = float4(0.85, 0.82, 0.78, 1);  // Light grey base
float4 color2        = float4(0.3, 0.28, 0.25, 1);   // Dark speckles
float4 color3        = float4(0.6, 0.55, 0.5, 1);    // Mid-tone veins
float4 tex_trans     = float4(0, 0, 1, 1);
float  noise_scale   = 4.0;
float  speckle_scale = 20.0;
float  vein_strength = 0.4;
float  specular_pow  = 32.0;

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

Texture2D    tex         : register(t3);
SamplerState tex_sampler : register(s3);

// Hash functions for noise
float hash3(float3 p) {
	p = frac(p * 0.1031);
	p += dot(p, p.yzx + 33.33);
	return frac((p.x + p.y) * p.z);
}

float hash2(float2 p) {
	float3 p3 = frac(float3(p.x, p.y, p.x) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return frac((p3.x + p3.y) * p3.z);
}

// 3D value noise
float noise3(float3 p) {
	float3 i = floor(p);
	float3 f = frac(p);
	f = f * f * (3.0 - 2.0 * f);

	float a = hash3(i);
	float b = hash3(i + float3(1, 0, 0));
	float c = hash3(i + float3(0, 1, 0));
	float d = hash3(i + float3(1, 1, 0));
	float e = hash3(i + float3(0, 0, 1));
	float g = hash3(i + float3(1, 0, 1));
	float h = hash3(i + float3(0, 1, 1));
	float j = hash3(i + float3(1, 1, 1));

	float x1 = lerp(a, b, f.x);
	float x2 = lerp(c, d, f.x);
	float y1 = lerp(x1, x2, f.y);

	float x3 = lerp(e, g, f.x);
	float x4 = lerp(h, j, f.x);
	float y2 = lerp(x3, x4, f.y);

	return lerp(y1, y2, f.z);
}

// FBM for larger features
float fbm(float3 p) {
	float f = 0.0;
	f += 0.5000 * noise3(p); p *= 2.01;
	f += 0.2500 * noise3(p); p *= 2.02;
	f += 0.1250 * noise3(p); p *= 2.03;
	f += 0.0625 * noise3(p);
	return f;
}

// Voronoi for crystalline speckles (simplified 2D for performance)
float voronoi(float3 p) {
	float2 ip = floor(p.xy);
	float2 fp = frac(p.xy);

	float min_dist = 1.0;
	int x, y;
	for (x = -1; x <= 1; x++) {
		for (y = -1; y <= 1; y++) {
			float2 offset = float2(x, y);
			float2 cell = ip + offset;
			float2 pt = offset + hash2(cell) - fp;
			float dist = dot(pt, pt);
			min_dist = min(min_dist, dist);
		}
	}
	return sqrt(min_dist);
}

psIn vs(vsIn input, uint id : SV_InstanceID) {
	uint inst_idx = id / view_count;
	uint view_idx = id % view_count;

	float4 world_pos = mul(float4(input.pos, 1), inst[inst_idx].world);

	psIn output;
	output.pos       = mul(world_pos, viewproj[view_idx]);
	output.normal    = normalize(mul(float4(input.norm, 0), inst[inst_idx].world).xyz);
	output.uv        = (input.uv * tex_trans.zw) + tex_trans.xy;
	output.color     = input.color;
	output.world_pos = world_pos.xyz;
	output.layer     = view_idx;
	return output;
}

float4 ps(psIn input) : SV_TARGET {
	const float3 light_dir = normalize(float3(1, 3, 2));

	float3 normal = normalize(input.normal);
	float3 view_dir = normalize(cam_pos[input.layer].xyz - input.world_pos);

	// Multi-scale granite pattern
	float3 p = input.world_pos;

	// Large-scale veining using FBM
	float vein = fbm(p * noise_scale);
	vein = sin(p.x * 1.5 + p.z * 0.8 + vein * 6.0) * 0.5 + 0.5;
	vein = smoothstep(0.3, 0.7, vein);

	// Crystalline speckles using Voronoi
	float speckle = voronoi(p * speckle_scale);
	speckle = smoothstep(0.0, 0.3, speckle);

	// Additional fine noise for surface variation
	float fine = noise3(p * speckle_scale * 2.0);

	// Blend colors based on patterns
	float3 granite_color = color1.rgb;
	granite_color = lerp(granite_color, color3.rgb, vein * vein_strength);
	granite_color = lerp(color2.rgb, granite_color, speckle);
	granite_color = lerp(granite_color, granite_color * (0.9 + fine * 0.2), 0.5);

	// Sample base texture and modulate
	float4 tex_color = tex.Sample(tex_sampler, input.uv);
	granite_color *= tex_color.rgb * input.color.rgb;

	// Lighting
	float ndotl = saturate(dot(normal, light_dir));
	float ambient = 0.25;
	float diffuse = ndotl * 0.6;

	// Specular - granite has subtle sheen from mica/quartz
	float3 half_dir = normalize(light_dir + view_dir);
	float spec = pow(saturate(dot(normal, half_dir)), specular_pow);
	spec *= (1.0 - speckle) * 0.3; // Speckles are more reflective

	float3 final_color = granite_color * (ambient + diffuse) + spec;

	return float4(final_color, 1.0);
}

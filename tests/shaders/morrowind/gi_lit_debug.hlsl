#include "gi.hlsli"

//--name = app/gi_lit_debug
// Debug viz: surface rendered using *only* the GI bounce as its lit term.
// Same vertex pipeline as mw_lit (white default diffuse, tint, vertex color)
// minus sun/shadow/dyn-light/emissive — so what you see on the sphere is
// exactly what GISampleIndirect returns at each fragment's world position
// + normal, including the fade-to-sky at the volume edges.
//
// Cbuffer sizes in this shader: $Global (16 B), GIBuffer (80 B), and the
// implicit SK stereokit_buffer at b1. All distinct — $Global is safe.

//--color = 1,1,1,1
float4 color;

//--diffuse = white
Texture2D    diffuse   : register(t0);
SamplerState diffuse_s : register(s0);

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos       : SV_Position;
	float2 uv        : TEXCOORD0;
	float4 color     : COLOR0;
	float3 world_pos : TEXCOORD1;
	float3 normal_w  : TEXCOORD2;
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;
	float4 world = mul(input.pos, sk_inst[ids.inst].world);
	o.pos        = mul(world, sk_viewproj[ids.view]);
	o.uv         = input.uv;
	o.color      = color * input.col * sk_inst[ids.inst].color;
	o.world_pos  = world.xyz;
	o.normal_w   = normalize(mul(input.norm, (float3x3)sk_inst[ids.inst].world));
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	float4 col = diffuse.Sample(diffuse_s, input.uv);
	float3 gi  = GISampleIndirect(input.world_pos, normalize(input.normal_w));
	col.rgb    = col.rgb * input.color.rgb * gi;
	col.a      = col.a * input.color.a;
	return col;
}

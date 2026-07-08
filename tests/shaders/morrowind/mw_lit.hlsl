#include "stereokit.hlsli"
#include "common.hlsli"
#include "lighting.hlsli"
#include "gi.hlsli"

// === GI sampling rate ===
// 1 = Per-pixel GI. VS passes world_pos + normal to the PS, which calls
//     GISampleIndirect per fragment. Most expensive but most accurate —
//     captures within-triangle GI variation that per-vertex misses on
//     large polys (terrain, walls).
// 0 = Per-vertex GI. VS calls GISampleIndirect once per vertex and the
//     result is interpolated. Cheap; GI is low-frequency enough that the
//     loss is usually invisible on tessellated geometry.
#define MW_LIT_GI_PER_PIXEL 0

// Note: $Global cbuffer is sized to 64 bytes — distinct from FogBuffer (32 B),
// LightingBuffer (48 B), GIBuffer (80 B), and ShadowBuffer (96 B) that this
// shader also pulls in. skshaderc's spirv-opt performance pass collapses
// $Global if its size matches any other cbuffer in the shader; we hit that
// previously with a 32-byte $Global matching FogBuffer, and `color` silently
// vanished. Add new members carefully and keep the total padded to 64.
//--color = 1,1,1,1
float4 color;
//--emissive = 0,0,0,0
float4 emissive;
float4 _pad0;
float4 _pad1;

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
	float4 pos          : SV_Position;
	float2 uv           : TEXCOORD0;
	float3 shadow_uv    : TEXCOORD1;
	float  shadow_ndotl : TEXCOORD2;
	float4 color        : COLOR0;
	// .rgb = directional in-scatter color (Sky-SH + Henyey-Greenstein sun
	//        glow at this vertex's view direction, see FogScatter in
	//        common.hlsli). Per-vertex is fine — the underlying SH is
	//        low-frequency and the HG phase varies smoothly over screen-space.
	// .a   = extinction factor (1 - transmittance) from FogExtinction.
	// Packed as float4 so the related scatter+extinction terms share one
	// interpolator slot, applied together as lerp(col, fog.rgb, fog.a).
	float4 fog          : TEXCOORD3;
#if MW_LIT_GI_PER_PIXEL
	// Per-pixel GI path: pass through the data GISampleIndirect needs and
	// let the PS evaluate it per fragment. Normal arrives un-normalised
	// from interpolation; the PS renormalises before sampling.
	float3 world_pos    : TEXCOORD4;
	float3 normal       : TEXCOORD5;
#else
	// Per-vertex path: all lighting computable without per-pixel data —
	// dyn-light bounce, GI indirect, and self-illumination. The only
	// per-pixel term left is the sun + shadow contribution, added in the
	// PS combine.
	float3 indirect     : TEXCOORD4;
#endif
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;

	float4 world = mul(input.pos, sk_inst[ids.inst].world);
	o.pos        = mul(world,     sk_viewproj[ids.view]);

	float3 normal = normalize(mul(input.norm, (float3x3)sk_inst[ids.inst].world));

	o.uv         = input.uv;
	o.color      = color * input.col * sk_inst[ids.inst].color;
	float3 vdir  = normalize(world.xyz - sk_camera_pos[ids.view].xyz);
	o.fog        = float4(FogScatter(vdir),
	                       FogExtinction(mul(world, sk_view[ids.view]).xyz));
	o.shadow_uv  = ShadowVS(world.xyz, normal, o.shadow_ndotl);
	// Direct point-light contribution is baked into the probe SH via next-
	// event sampling in gi_voxel_to_sh.hlsl, so we don't sample AccumulateLights
	// here — that would double-count. GISampleIndirect carries bounce +
	// direct-with-visibility for every dyn-light.
#if MW_LIT_GI_PER_PIXEL
	o.world_pos  = world.xyz;
	o.normal     = normal;
#else
	o.indirect   = GISampleIndirect(world.xyz, normal) + emissive.rgb;
#endif
	return o;
}

float4 ps(psIn input) : SV_TARGET {
	float4 col   = diffuse.Sample(diffuse_s, input.uv);
	float  light = ShadowPS(input.shadow_uv, input.shadow_ndotl);
#if MW_LIT_GI_PER_PIXEL
	float3 indirect = GISampleIndirect(input.world_pos, normalize(input.normal)) + emissive.rgb;
#else
	float3 indirect = input.indirect;
#endif
	col.rgb      = col.rgb * input.color.rgb * (light * shadow_light_color + indirect);
	//col.rgb      =                              light * shadow_light_color + indirect;
	// Beer-Lambert composition: col * transmittance + scatter * (1 - transmittance).
	// `lerp(col, scatter, fog.a)` with fog.a = 1 - transmittance is the same form.
	col.rgb      = lerp(col.rgb, input.fog.rgb, input.fog.a);
	// Preserve diffuse + tint alpha so transparent materials actually blend.
	// Fog only blends RGB; alpha is the texture's alpha modulated by the tint.
	col.a        = col.a * input.color.a;
	return col;
}

// Water surface shader. Modeled after mw_lit.hlsl so it shares fog + shadow
// integration via common.hlsli. Owns its own UV transform (tex_trans) so we
// can tile the diffuse across a cell-sized plane without rebuilding the mesh,
// and is the planned home for expansions: scrolling UVs for animation, sky
// reflection (Sky.ComputeColor), fresnel, depth-fade, etc.

#include "stereokit.hlsli"
#include "common.hlsli"

// Note on cbuffer sizing: skshaderc's spirv-opt performance pass collapses
// $Global if it matches any other cbuffer's size in this shader. FogBuffer is
// 32 bytes and ShadowBuffer is 96 bytes (see common.hlsli), so $Global must
// be a different size. We use 48 bytes here (color + tex_trans + frame_data).
//--color = 1,1,1,1
float4 color;
//--tex_trans = 0,0,1,1
// XY = UV offset, ZW = UV scale. UVs are derived from world XZ (not the
// mesh's per-vertex UV) so the texture pattern is continuous across cell
// boundaries — without this the dual-tap blend would visibly seam at every
// cell edge. ZW is therefore "tiles per world meter"; e.g. 0.27 ≈ one tile
// per 3.66 m, matching the old mesh-space rate of 32 tiles per 117 m cell.
float4 tex_trans;
//--frame_data = 0,0,0,0
// .x = blend factor in [0,1] for cross-fading from diffuse → diffuse_next.
// .y..w reserved (padding; do not remove without re-checking cbuffer sizing).
float4 frame_data;

//--diffuse = white
Texture2D    diffuse        : register(t0);
SamplerState diffuse_s      : register(s0);
// Next-frame slot for flipbook cross-fading. C# binds this to frame n+1 while
// `diffuse` carries frame n; frame_data.x slides 0→1 over the frame's display
// duration so the swap from one frame to the next is continuous, not stepped.
//--diffuse_next = white
Texture2D    diffuse_next   : register(t1);
SamplerState diffuse_next_s : register(s1);

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
	float3 ambient      : TEXCOORD3;
	// .rgb = directional in-scatter color (Sky-SH + Henyey-Greenstein sun
	//        glow at this vertex's view direction; see FogScatter in
	//        common.hlsli).
	// .a   = extinction factor (1 - transmittance) from FogExtinction.
	// Packed float4 so the related scatter+extinction share one interpolator,
	// applied together as lerp(col, fog.rgb, fog.a).
	float4 fog          : TEXCOORD4;
	// Camera-relative view direction (frag → camera). Computed in VS so we can
	// grab the correct per-eye sk_camera_pos[ids.view]; PS just normalizes.
	float3 view_dir     : TEXCOORD5;
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;

	float4 world = mul(input.pos, sk_inst[ids.inst].world);
	o.pos        = mul(world,     sk_viewproj[ids.view]);

	float3 normal = normalize(mul(input.norm, (float3x3)sk_inst[ids.inst].world));

	o.uv         = world.xz * tex_trans.zw + tex_trans.xy;
	o.color      = color * input.col * sk_inst[ids.inst].color;
	o.ambient    = sk_lighting(normal);
	o.shadow_uv  = ShadowVS(world.xyz, normal, o.shadow_ndotl);
	float3 cam_to_frag = world.xyz - sk_camera_pos[ids.view].xyz;
	o.view_dir   = -cam_to_frag;                       // existing convention: frag → camera
	o.fog        = float4(FogScatter(normalize(cam_to_frag)),
	                       FogExtinction(mul(world, sk_view[ids.view]).xyz));
	return o;
}

// Cross-fade between current and next flipbook frames at a given UV, using
// explicit derivatives so the random per-corner UV offsets in SampleNoTile
// don't fool the GPU into picking the wrong mip level.
float4 SampleFlipbookGrad(float2 uv, float2 ddx_uv, float2 ddy_uv) {
	float4 a = diffuse     .SampleGrad(diffuse_s,      uv, ddx_uv, ddy_uv);
	float4 b = diffuse_next.SampleGrad(diffuse_next_s, uv, ddx_uv, ddy_uv);
	return lerp(a, b, frame_data.x);
}

// Cheap 2D hash, paraphrased from Inigo Quilez's article on texture
// repetition: https://iquilezles.org/articles/texturerepetition/
// The constants are chosen so the output is well-distributed for the
// integer "supertile" coordinates we feed it (output of floor(uv)).
float hash11(float2 p) {
	p = 50.0 * frac(p * 0.3183099 + float2(0.71, 0.113));
	return frac(p.x * p.y * (p.x + p.y));
}
float2 hash22(float2 p) {
	return float2(hash11(p), hash11(p + float2(17.1, 31.7)));
}

// IQ Technique 3: per-pixel, take the integer part of UV ("supertile cell"),
// hash each of the 4 corners of that cell into a 2D random UV-space offset,
// sample the (flipbook-cross-faded) texture at each corner's offset UV, then
// smoothstep-blend the four samples by the fractional UV. Within a supertile
// the four samples blend continuously; across supertiles, since the corners
// are shared with neighbors, the blend stays continuous as well — no seams,
// and the underlying texture's tile pattern is destroyed because each
// supertile sees a different shifted copy of the source.
float4 SampleNoTile(float2 uv) {
	float2 iuv = floor(uv);
	float2 fuv = frac (uv);

	// Compute screen-space derivatives once on the natural UV so we get a
	// sane mip level regardless of how chaotic the per-corner offsets look.
	float2 ddx_uv = ddx(uv);
	float2 ddy_uv = ddy(uv);

	float2 offa = hash22(iuv + float2(0, 0));
	float2 offb = hash22(iuv + float2(1, 0));
	float2 offc = hash22(iuv + float2(0, 1));
	float2 offd = hash22(iuv + float2(1, 1));

	float4 a = SampleFlipbookGrad(uv + offa, ddx_uv, ddy_uv);
	float4 b = SampleFlipbookGrad(uv + offb, ddx_uv, ddy_uv);
	float4 c = SampleFlipbookGrad(uv + offc, ddx_uv, ddy_uv);
	float4 d = SampleFlipbookGrad(uv + offd, ddx_uv, ddy_uv);

	float2 w = smoothstep(0.0, 1.0, fuv);
	return lerp(
		lerp(a, b, w.x),
		lerp(c, d, w.x),
		w.y);
}

float4 ps(psIn input) : SV_TARGET {
	float4 col = SampleNoTile(input.uv);

	// Treat the surface as geometrically flat — just the +Y plane normal. The
	// brightness-gradient bump approach (ddx/ddy(luminance) → tangent normal)
	// produced too much specular aliasing in the distance to be salvageable
	// even with a falloff, so we go without. SH reflection still gives the
	// surface a view-angle-dependent sky tint via fresnel, just without
	// per-wave variation.
	float3 nWorld = float3(0, 1, 0);

	// Sample SH at the reflection direction. Spherical harmonics is
	// fundamentally low-frequency / "diffuse"-style, but evaluating it at the
	// *reflection* of the view about our flat normal still gives a usable
	// "sky color" contribution: a darker tone when looking straight down
	// (sampling the SH at +Y), brighter at oblique angles (sampling toward
	// the horizon).
	float3 viewDir = normalize(input.view_dir);
	float3 reflDir = reflect(-viewDir, nWorld);
	float3 specEnv = sk_lighting(reflDir);

	float  light   = ShadowPS(input.shadow_uv, input.shadow_ndotl);

	// Specular shadow: if the reflection is aimed at the sun but the surface
	// is shadowed (a tree or cliff between us and the sun), the sun's light
	// isn't actually arriving here, so its reflection shouldn't either. Recover
	// the raw shadow factor by undoing ShadowPS's `min(ndotl, shadow_pcf)`,
	// then concentrate the dip to the sharp "sun glint" spot via pow(·, 16) —
	// reflections elsewhere on the SH still pass through full strength.
	float sun_visibility    = saturate(light / max(input.shadow_ndotl, 1e-3));
	float refl_toward_sun   = pow(saturate(dot(reflDir, shadow_light_dir)), 16.0);
	float sun_mask          = lerp(1.0, sun_visibility, refl_toward_sun);
	specEnv                *= sun_mask;

	// Schlick fresnel — water reflects ~2% straight on, almost 100% at
	// grazing angles. This naturally hides specular when the player looks
	// straight down and brings it out at oblique angles.
	float NdotV   = saturate(dot(nWorld, viewDir));
	float fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);

	float3 diffuse = col.rgb * input.color.rgb * (input.ambient + light * shadow_light_color);

	col.rgb = diffuse + specEnv * fresnel;
	// Beer-Lambert composition (same form as the old single-tint lerp; the
	// content of fog.rgb is now view-directional sky + sun glow).
	col.rgb = lerp(col.rgb, input.fog.rgb, input.fog.a);
	// Preserve diffuse + tint alpha so transparent materials actually blend.
	// Fog only blends RGB; alpha is the texture's alpha modulated by the tint.
	col.a   = col.a * input.color.a;
	return col;
}

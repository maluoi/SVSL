#include <stereokit.hlsli>

//--name = app/gi_probe_debug
// Debug viz for the SH probe grid. One sphere per probe; samples sh_r/g/b at
// the sphere's center and evaluates L1 SH at the fragment normal so each probe
// surface shows how incoming radiance varies by direction.

cbuffer GIBuffer : register(b4) {
	float3 gi_volume_min;
	float  gi_voxel_res;
	float3 gi_volume_inv;
	float  gi_intensity;
	float4 _gi_pad0;
	// .x = voxel storage range scale (this shader doesn't sample voxels, but
	// the field has to be present for the cbuffer layout to match GIBufferData).
	float4 voxel_range_pad;
	float4 _gi_pad2;
};

// ZH3-compressed live SH (see gi.hlsli for the encode/decode). Local copies
// of the bindings + decode helper because this shader doesn't pull in
// gi.hlsli (and including it would drag in the full GISampleIndirect path
// the debug viz is deliberately bypassing). The linear-sampled decode below
// blends ratios in unorm space first then scales by interpolated DC — an
// approximation, but acceptable for low-frequency debug viz. Trap factor
// lives in sh_dc.a (Stage 2 — the dedicated R8 sh_state texture is gone).
Texture3D<float4> sh_dc    : register(t6);
Texture3D<float4> sh_r_l1  : register(t7);
Texture3D<float4> sh_g_l1  : register(t8);
Texture3D<float4> sh_b_l1  : register(t10);
SamplerState     sh_s     : register(s6);

static const float GI_L1_NORM = 1.7320508; // √3, matches gi_voxel_to_sh

void _decode_probe_linear(float3 uvw, out float4 r4, out float4 g4, out float4 b4) {
	float3 dc      = sh_dc  .SampleLevel(sh_s, uvw, 0).rgb;
	float3 r_ratio = sh_r_l1.SampleLevel(sh_s, uvw, 0).rgb * 2.0 - 1.0;
	float3 g_ratio = sh_g_l1.SampleLevel(sh_s, uvw, 0).rgb * 2.0 - 1.0;
	float3 b_ratio = sh_b_l1.SampleLevel(sh_s, uvw, 0).rgb * 2.0 - 1.0;
	float3 scale   = GI_L1_NORM * dc;
	r4 = float4(dc.r, r_ratio * scale.r);
	g4 = float4(dc.g, g_ratio * scale.g);
	b4 = float4(dc.b, b_ratio * scale.b);
}

struct vsIn {
	float4 pos  : SV_Position;
	float3 norm : NORMAL0;
	float2 uv   : TEXCOORD0;
	float4 col  : COLOR0;
};
struct psIn {
	float4 pos    : SV_POSITION;
	float3 normal : NORMAL0;
	nointerpolation float3 sphere_center : TEXCOORD0;
};

psIn vs(vsIn input, sk_ids_t ids) {
	psIn o;
	float4 world    = mul(input.pos, sk_inst[ids.inst].world);
	o.pos           = mul(world, sk_viewproj[ids.view]);
	o.normal        = normalize(mul(input.norm, (float3x3)sk_inst[ids.inst].world));
	o.sphere_center = mul(float4(0, 0, 0, 1), sk_inst[ids.inst].world).xyz;
	return o;
}

float _eval_sh(float4 sh, float3 n) {
	return sh.x * 0.28209 + sh.y * 0.48860 * n.y + sh.z * 0.48860 * n.z + sh.w * 0.48860 * n.x;
}

float4 ps(psIn input) : SV_TARGET {
	// Per-axis everywhere — the SH probe texture is ProbeRes_* per axis
	// (64×32×64) so a world position has a different UVW per axis. Dividing
	// by a single axis's extent under-samples the others (Y looked stretched
	// 2× under the old / volume_extent.x form).
	float3 volume_extent = 1.0 / gi_volume_inv;
	float3 to_min        = input.sphere_center - gi_volume_min;
	if (any(to_min < 0) || any(to_min > volume_extent)) discard;
	float3 uvw = input.sphere_center / volume_extent;

	float3 n = normalize(input.normal);
	float4 r, g, b;
	_decode_probe_linear(uvw, r, g, b);

	float3 col = max(0, float3(_eval_sh(r, n), _eval_sh(g, n), _eval_sh(b, n)));

	// Trap-factor tint: fully-trapped probes go pure red, half-trapped show
	// as bruised-purple. Probes the consumer 8-tap downweights to zero are
	// exactly the ones rendered red here. Trap is packed into sh_dc.a.
	float trap = sh_dc.SampleLevel(sh_s, uvw, 0).a;
	col = lerp(col, float3(1, 0, 0), trap);

	// TEMP debug: the demo adds += 0.05 here so empty probes stay visible.
	// Skipping that floor so we can see what the SH actually carries — empty
	// probes will render fully black instead of middle-gray.
	return float4(col, 1);
}

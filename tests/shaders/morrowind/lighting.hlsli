// Drop-in dynamic point-light system. Include this header and call
// AccumulateLights(world.xyz, normal) from a vertex (or pixel) shader to get
// the additive contribution of up to 4 dynamic lights that influence the
// vertex's cell.
//
// Setup is project-side (see SKMorrowind.Lighting): the CPU owns a light
// pool and a toroidal 2D grid where each cell stores the 4 slot indices of
// the lights with the strongest influence at the cell center.
//
// Bindings (StereoKit reserves b1/t11/t12 in stereokit.hlsli):
//   cbuffer b12   — LightingBuffer (48 bytes; see size note below)
//   t14/s14       — light_grid_tex  (Rgba128: 4 light slot indices per cell)
//   t15/s15       — light_table_tex (Rgba128: 2 texels per slot:
//                                    [pos.xyz, radius] then [color.rgb, _])
//
// Cbuffer size note: 48 bytes is chosen to be distinct from FogBuffer (32 B)
// and ShadowBuffer (96 B) — skshaderc's spirv-opt collapses $Global when its
// size matches another cbuffer in the same shader. See CLAUDE.md.

cbuffer LightingBuffer : register(b12) {
	// .xy = grid origin XZ (snapped, world m); .z = 1/cellSize_m;
	// .w = light table width in texels (= maxLights * 2).
	float4 light_origin_inv;
	// .xy = cellsX, cellsZ; .zw = 1/cellsX, 1/cellsZ.
	float4 light_grid;
	// reserved
	float4 light_misc;
};

Texture2D    light_grid_tex  : register(t14);
SamplerState light_grid_s    : register(s14);
Texture2D    light_table_tex : register(t15);
SamplerState light_table_s   : register(s15);

float3 _LightSample(int slot, float3 world_pos, float3 normal) {
	// Slot 0 is the "empty" sentinel. Skip cheaply.
	if (slot == 0) return 0;
	float inv_w = 1.0 / light_origin_inv.w;
	float u0    = (slot * 2 + 0.5) * inv_w;
	float u1    = (slot * 2 + 1.5) * inv_w;
	float4 pr   = light_table_tex.SampleLevel(light_table_s, float2(u0, 0.5), 0);
	float4 cp   = light_table_tex.SampleLevel(light_table_s, float2(u1, 0.5), 0);
	float3 toL  = pr.xyz - world_pos;
	float  d    = length(toL);
	float  r    = max(pr.w, 1e-4);
	// Vanilla MW attenuation: att = r / (3 * d), pure hyperbolic with no
	// hard cutoff — at d=r att is 1/3, and contribution spikes toward
	// infinity at the source. We cap the peak at LIGHT_MAX_GAIN to keep
	// the bright pool finite, but otherwise preserve the vanilla curve
	// (no smooth ramp to 0 at d=r). The grid binner's per-cell d ≤ r
	// inclusion test still bounds the per-cell influence radius — past
	// that distance the light simply isn't sampled in the grid lookup.
	const float LIGHT_MAX_GAIN = 30.0;
	float  dr  = d / r;
	float  att = min(LIGHT_MAX_GAIN, 1.0 / (3.0 * max(dr, 1e-4)));
	float  ndotl  = max(0, dot(normal, toL / max(d, 1e-4)));
	return cp.rgb * (ndotl * att);
}

float3 AccumulateLights(float3 world_pos, float3 normal) {
	// World cell index is anchored at the world origin; with Wrap addressing,
	// cells outside the current window alias back into the texture. The CPU
	// keeps each texel aligned to whichever absolute world cell it currently
	// represents.
	float2 world_cell = floor(world_pos.xz * light_origin_inv.z);
	float2 uv         = (world_cell + 0.5) * light_grid.zw;
	float4 idx        = light_grid_tex.SampleLevel(light_grid_s, uv, 0);
	return _LightSample((int)idx.x, world_pos, normal)
	     + _LightSample((int)idx.y, world_pos, normal)
	     + _LightSample((int)idx.z, world_pos, normal)
	     + _LightSample((int)idx.w, world_pos, normal);
}

//--name = app/gi_sdf_seed
// JFA seed pass — each occupied cell (voxel_tex.a > 0) becomes a "seed" whose
// payload is its *slot-local* position within the active window, in
// [0, VOXEL_RES). Subsequent JFA passes propagate seeds outward; the
// finalizer reads the seed's slot-local back and computes Euclidean distance
// to the current cell's slot-local.
//
// Slot-local instead of raw-slot avoids the toroidal-seam confusion in the
// previous encoding: slot 63 and slot 64 are adjacent in slot space but
// represent world cells 128 voxels apart whenever origin_vox & 127 falls in
// between them. With slot-local positions the JFA neighbour read can also be
// *bounded* (gi_sdf_jfa.hlsl skips neighbours whose slot-local lies outside
// [0, VOXEL_RES)), so seeds never propagate across the world-discontinuity
// seam — the resulting SDF reports true world distance everywhere inside
// the active window.
//
// Seed encoding (Rgba8 unorm):
//   rgb = slot_local / 255  — recovers integer position via round.
//   a   = 1.0 for occupied, 0.0 otherwise (valid flag).
// VOXEL_RES is non-cubic 256×128×256; 8-bit slot-local positions cover
// [0, 255] per axis, exact for all three.

cbuffer GIBuffer : register(b4) {
	float3 gi_volume_min;
	float  gi_voxel_res;
	float3 gi_volume_inv;
	float  gi_intensity;
	float4 gi_prev_origin_probe_res;
	float4 gi_voxel_range_pad;
	float4 _gi_pad2;
};

Texture3D<float4>                                voxel_tex : register(t0);
[[vk::image_format("rgba8")]] RWTexture3D<unorm float4> jfa_out : register(u0);

// Per-axis voxel resolution. Must match gi.hlsli + gi_voxel_to_sh.hlsl.
#define VOXEL_RES_X     256
#define VOXEL_RES_Y     128
#define VOXEL_RES_Z     256
static const int3 VOXEL_RES      = int3(VOXEL_RES_X,     VOXEL_RES_Y,     VOXEL_RES_Z);
static const int3 VOXEL_RES_MASK = int3(VOXEL_RES_X - 1, VOXEL_RES_Y - 1, VOXEL_RES_Z - 1);

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	float4 v = voxel_tex.Load(int4(id, 0));
	if (v.a <= 0) {
		jfa_out[id] = float4(0, 0, 0, 0);
		return;
	}
	// Slot-local position of this cell within the active window:
	//   local = (id - (origin_vox & MASK)) mod VOXEL_RES, in [0, VOXEL_RES).
	// origin_vox is the world voxel coord of the active window's low corner;
	// (origin_vox & MASK) is the *slot* that holds that low-corner world cell
	// (the slot mapping is toroidal: world_vox & MASK). Subtracting and re-
	// wrapping gives this cell's offset from the low corner in slot-local
	// space, which is the same as its world-voxel offset from origin_vox
	// (in [0, VOXEL_RES)) for any cell inside the active window.
	// Volume can be non-cubic but voxel spacing is uniform, so derive
	// voxel_size from any axis (X here as the canonical choice).
	float voxel_size = (1.0 / gi_volume_inv.x) / (float)VOXEL_RES_X;
	int3  origin_off = int3(floor(gi_volume_min / voxel_size)) & VOXEL_RES_MASK;
	int3  local      = (int3(id) - origin_off + VOXEL_RES) & VOXEL_RES_MASK;
	jfa_out[id] = float4(float3(local) / 255.0, 1.0);
}

//--name = app/gi_voxel_clear
// Wipe the voxel volume to (0,0,0,0) before each voxelize pass, so stale
// lit values from the previous frame don't survive via the voxelize PS's
// "brightest sample wins" merge.

[[vk::image_format("rgba8")]]
RWTexture3D<unorm float4> voxel_tex : register(u0);

// Per-axis volume resolution. Hardcoded to match gi.hlsli — runtime uniforms
// would prevent the SSA optimizer from folding the bounds check into a
// straight-line branch and the dispatch shape (in GI.cs) is sized to match
// these exactly, so an out-of-bounds early-out is only a defensive cap.
#define VOXEL_RES_X 256u
#define VOXEL_RES_Y 128u
#define VOXEL_RES_Z 256u
static const uint3 VOXEL_RES = uint3(VOXEL_RES_X, VOXEL_RES_Y, VOXEL_RES_Z);

[numthreads(4, 4, 4)]
void cs(uint3 id : SV_DispatchThreadID) {
	if (any(id >= VOXEL_RES)) return;
	voxel_tex[id] = float4(0, 0, 0, 0);
}

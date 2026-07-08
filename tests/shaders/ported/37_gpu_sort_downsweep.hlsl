/******************************************************************************
 * GPU Sort Downsweep Kernel
 * Ranks keys using wave-level multi-split and scatters to sorted positions
 *
 * Dispatch: thread_blocks workgroups
 ******************************************************************************/

// Configuration
#define KEY_UINT 1
#define PAYLOAD_UINT 1
#define SHOULD_ASCEND 1
#define SORT_PAIRS 1

#include "gpu_sort_common.hlsli"

[numthreads(D_DIM, 1, 1)]
void cs(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Downsweep_Main(gtid.x, gid.x);
}

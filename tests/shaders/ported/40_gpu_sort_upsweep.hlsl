/******************************************************************************
 * GPU Sort Upsweep Kernel
 * Builds per-partition histograms and reduces to global histogram
 *
 * Dispatch: thread_blocks workgroups
 ******************************************************************************/

// Configuration
#define KEY_UINT 1
#define PAYLOAD_UINT 1
#define SHOULD_ASCEND 1
#define SORT_PAIRS 1

#include "gpu_sort_common.hlsli"

[numthreads(US_DIM, 1, 1)]
void cs(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Upsweep_Main(gtid.x, gid.x);
}

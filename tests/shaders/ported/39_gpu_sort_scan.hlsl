/******************************************************************************
 * GPU Sort Scan Kernel
 * Exclusive prefix sum over partition histograms
 *
 * Dispatch: RADIX (256) workgroups - one per digit
 ******************************************************************************/

// Configuration
#define KEY_UINT 1
#define PAYLOAD_UINT 1
#define SHOULD_ASCEND 1
#define SORT_PAIRS 1

#include "gpu_sort_common.hlsli"

[numthreads(SCAN_DIM, 1, 1)]
void cs(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    Scan_Main(gtid.x, gid.x);
}

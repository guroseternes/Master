#ifndef UTIL_H_
#define UTIL_H_

#include "GpuPtr.h"

void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y);

void setCoarsePermIntegrationArgs(CoarsePermIntegrationKernelArgs* args, GpuPtrRaw K, GpuPtrRaw k_data, GpuPtrRaw k_heights, GpuPtrRaw p_cap_ref_table, GpuPtrRaw s_b_ref_table, unsigned int nx, unsigned int ny, unsigned int border);

#endif /* UTIL_H_ */

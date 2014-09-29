#ifndef UTIL_H_
#define UTIL_H_

#include "GpuPtr.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "KernelArgStructs.h"

void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y);

void setCoarsePermIntegrationKernelArgs(CoarsePermIntegrationKernelArgs* args, GpuRawPtr K, GpuRawPtr H,
										cudaPitchedPtr k, GpuRawPtr nI,
										float dz, unsigned int nx, unsigned int ny, unsigned int nz,
										unsigned int border);

// set arguments for the coarse permeability integration
void setCoarseMobIntegrationArgs(CoarseMobIntegrationKernelArgs* args,
								  GpuRawPtr K, GpuRawPtr h, GpuRawPtr k_data, GpuRawPtr k_heights,
								  GpuRawPtr p_cap_ref_table, GpuRawPtr s_b_ref_table,
								  float p_ci, float g, float rho_delta,
								  float dz, unsigned int nx, unsigned int ny, unsigned int border);

#endif /* UTIL_H_ */

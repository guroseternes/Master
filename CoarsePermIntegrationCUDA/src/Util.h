#ifndef UTIL_H_
#define UTIL_H_

#include "GpuPtr.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "KernelArgStructs.h"


void setCoarsePermIntegrationKernelArgs(CoarsePermIntegrationKernelArgs* args, GpuRawPtr K, GpuRawPtr H,
										cudaPitchedPtr k, GpuRawPtr nI,
										float dz, unsigned int nx, unsigned int ny, unsigned int nz,
										unsigned int border);

// set arguments for the coarse permeability integration
void setCoarseMobIntegrationArgs(CoarseMobIntegrationKernelArgs* args,
								  GpuRawPtr Lambda, GpuRawPtr H, GpuRawPtr h, cudaPitchedPtr k,
								  GpuRawPtr K, GpuRawPtr nI,
								  GpuRawPtr p_cap_ref_table,
								  GpuRawPtr s_c_ref_table,
								  float p_ci, float g, float delta_rho,
								  float dz, unsigned int nx, unsigned int ny, unsigned int border);
void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y);
#endif /* UTIL_H_ */

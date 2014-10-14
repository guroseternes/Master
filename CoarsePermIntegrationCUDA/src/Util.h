#ifndef UTIL_H_
#define UTIL_H_

#include "GpuPtr.h"
#include "CpuPtr.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "KernelArgStructs.h"
#include <vector>


void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y);
void createGridMask(CpuPtr_2D H, dim3 grid, dim3 block, int nx, int ny,
				    std::vector<int> &activeBlockIndexes, int& nActiveBlocks);


void setCommonArgs(CommonArgs* args, float delta_rho, float g,
				   GpuRawPtr H, GpuRawPtr p_cap_ref_table, GpuRawPtr s_c_ref_table,
			       unsigned int nx, unsigned int ny, unsigned int border);

void setCoarsePermIntegrationKernelArgs(CoarsePermIntegrationKernelArgs* args, GpuRawPtr K,
										cudaPitchedPtr k, GpuRawPtr nI,
										float dz);

// set arguments for the coarse permeability integration
void setCoarseMobIntegrationKernelArgs(CoarseMobIntegrationKernelArgs* args,
								  GpuRawPtr Lambda, GpuRawPtr h, cudaPitchedPtr k,
								  GpuRawPtr K, GpuRawPtr nI,
								  GpuRawPtrInt a_b_i,
								  float p_ci,
								  float dz);

void setFluxKernelArgs(FluxKernelArgs* args,
					   GpuRawPtr Lambda_c, GpuRawPtr Lambda_b,
					   GpuRawPtr U_x, GpuRawPtr U_y,
					   GpuRawPtr h, GpuRawPtr z);

#endif /* UTIL_H_ */

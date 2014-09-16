#include "kernels.h"

__constant__ CoarsePermIntegrationKernelArgs coarse_perm_int_ctx;

callCoarsePermIntegrationKernel(dim3 grid, dim3 block, CoarsePermIntegrationKernelArgs args){
	cudaMemcpyToSymbolAsync(coarse_perm_int_ctx, args, sizeof(RKKernelArgs), 0, cudaMemcpyHostToDevice);
	CoarsePermIntegrationKernel<<<grid, block>>>;
}

CoarsePermIntegrationKernel<<<grid, block>>>;

#include "Kernels.h"
#define N 100;

__constant__ CoarsePermIntegrationKernelArgs cpi_ctx;

void initAllocate(CoarsePermIntegrationKernelArgs* args){
	cudaHostAlloc(&args, sizeof(CoarsePermIntegrationKernelArgs), cudaHostAllocWriteCombined);
}

// Function to create an array of the permeability on the subintervals
__device__ void kDistribution(float dz, float h, float* k_heights, float* k_data, float* k_values){
	float z = 0;
	int j = 0;
	float curr_height = 0;
	for (int i = 0; i < 10; i++){
		curr_height = k_heights[i];
		while (z < curr_height && z <= h){
			z += dz;
			k_values[j] = k_data[i];
			j++;
		}
	}
}

__device__ float* global_index(float* base, unsigned int pitch, int x, int y, int border = 0) {
        return (float*) ((char*) base+(y+border)*pitch) + (x+border);
}

__device__ float trapezoidal(float dz, int n, float* function_values){
	float sum = 0;
	sum += 0.5*(function_values[0] + function_values[n]);
	for (int i = 1; i < n; i++){
		sum += function_values[i];
	}
	return sum*dz;
}


__global__ void CoarsePermIntegrationKernel(){

	int xid = blockIdx.x*blockDim.x + threadIdx.x;
    int yid = blockIdx.y*blockDim.y + threadIdx.y;

    if ( xid < 10 && yid < 10 ){
		// Get local height
		float h = global_index(cpi_ctx.height_distribution.ptr,
							   cpi_ctx.height_distribution.pitch, xid, yid)[0];
		float k_values[101];
		for (int i = 0; i<101; i++){
			k_values[i] = 0;
		}

		kDistribution(cpi_ctx.dz, h, cpi_ctx.k_heights.ptr, cpi_ctx.k_data.ptr, k_values);

		float K = trapezoidal(cpi_ctx.dz, 100, k_values);

		global_index(cpi_ctx.K.ptr, cpi_ctx.K.pitch, xid, yid)[0] = K;
    }
}


void callCoarsePermIntegrationKernel(dim3 grid, dim3 block, CoarsePermIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(cpi_ctx, args, sizeof(CoarsePermIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	CoarsePermIntegrationKernel<<<grid, block>>>();
}

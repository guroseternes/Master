#include "Kernels.h"
#define N 100;

__constant__ CoarsePermIntegrationKernelArgs cpi_ctx;

void initAllocate(CoarsePermIntegrationKernelArgs* args){
	cudaHostAlloc(&args, sizeof(CoarsePermIntegrationKernelArgs), cudaHostAllocWriteCombined);
}

// Function to create an array of the permeability on the subintervals
__device__ int kDistribution(float dz, float h, float* k_heights, float* k_data, float* k_values){
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
	return j;
}

__device__ void multiply(int n, float* x_values, float* y_values, float* product){
	for (int i = 0; i < n; i++){
		product[i] = x_values[i]*y_values[i];
	}
}

// Function to compute the capillary pressure in the subintervals
__device__ void computeCapillaryPressure(float p_ci, float g, float delta_rho, float h, float dz, int n, float* p_cap_values){
	for (int i = 0; i < n+1 ; i++){
		p_cap_values[i] = p_ci + g*(-delta_rho)*(dz*i-h);
	}
}

__device__ void inverseCapillaryPressure(int n, float* p_cap_values, float* p_cap_ref_table, float* s_b_ref_table){
	// pCap-saturation reference table
	for (int i = 0; i < n+1; i++){
		float curr_p_cap = p_cap_values[i];
		int j = 0;
		while (curr_p_cap > p_cap_ref_table[j]){
			j++;
		}
		p_cap_values[i] = s_b_ref_table[j];
	}
}

__device__ void computeMobility(int n, float* s_b_values){
	float lambda_end_point = 1;
	for (int i = 0; i < n+1; i++){
		s_b_values[i] = pow(s_b_values[i], 3)*lambda_end_point;
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
		float p_cap_values[101];

		for (int i = 0; i<101; i++){
			k_values[i] = 0;
			p_cap_values[i] = 0;
		}

		int n = kDistribution(cpi_ctx.dz, h, cpi_ctx.k_heights.ptr, cpi_ctx.k_data.ptr, k_values);
		float K = trapezoidal(cpi_ctx.dz, n-1, k_values);

		computeCapillaryPressure(cpi_ctx.p_ci, cpi_ctx.g, cpi_ctx.delta_rho,
								 h, cpi_ctx.dz, n, p_cap_values);


		inverseCapillaryPressure(n, p_cap_values, cpi_ctx.p_cap_ref_table.ptr, cpi_ctx.s_b_ref_table.ptr);

		computeMobility(n, p_cap_values);

		multiply(n, k_values, p_cap_values, k_values);

		float L = trapezoidal(cpi_ctx.dz, n-1, k_values)/K;

		global_index(cpi_ctx.K.ptr, cpi_ctx.K.pitch, xid, yid)[0] = L;
    }
}


void callCoarsePermIntegrationKernel(dim3 grid, dim3 block, CoarsePermIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(cpi_ctx, args, sizeof(CoarsePermIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	CoarsePermIntegrationKernel<<<grid, block>>>();
}

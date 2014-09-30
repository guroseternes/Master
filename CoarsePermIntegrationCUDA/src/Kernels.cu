#include "Kernels.h"

__constant__ CoarsePermIntegrationKernelArgs cpi_ctx;
__constant__ CoarseMobIntegrationKernelArgs cmi_ctx;

void initAllocate(CoarsePermIntegrationKernelArgs* args1, CoarseMobIntegrationKernelArgs* args2){
	cudaHostAlloc(&args1, sizeof(CoarsePermIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args2, sizeof(CoarseMobIntegrationKernelArgs), cudaHostAllocWriteCombined);
}

__device__ float lookupSaturation(float curr_p_cap, float* p_cap_ref_table, float* s_c_ref_table){
	int j = 0;
	while (curr_p_cap < p_cap_ref_table[j] && j < 99) {
		j++;
	}
	return s_c_ref_table[j];
}

__device__ float computeMobility(float s_c){
	float lambda_end_point = 1;
		return (pow(1-s_c, 3)*lambda_end_point);
}
// Function to compute the capillary pressure in the subintervals
__device__ float computeLambda(float p_ci, float g, float delta_rho, float h, float dz, int n,
							   float* k_values, float* p_cap_ref_table, float* s_c_ref_table){
	float current_p_cap = 0;
	float current_satu_c = 0;
	float current_mob = 0;
	float sum = 0;
	for (int i = 0; i < n ; i++){
		current_p_cap = p_ci + g*(-delta_rho)*(dz*i-h);
		current_satu_c = lookupSaturation(current_p_cap, p_cap_ref_table, s_c_ref_table);
		current_mob = computeMobility(current_satu_c);
		sum += current_mob*k_values[i];
	}
	return sum*dz;
}


__device__ float* global_index(float* base, unsigned int pitch, int x, int y, int border = 0) {
        return (float*) ((char*) base+(y+border)*pitch) + (x+border);
}

__device__ float* global_index(cudaPitchedPtr ptr, int x, int y, int z, int border = 0) {
        return (float*) ((char*) (ptr.ptr+(x+border)*(ptr.ysize)*ptr.pitch)) + (y+border)*(ptr.pitch/sizeof(float)) + z;
}


__device__ float trapezoidal(float H, float dz, int n, float* function_values){
	float sum = 0;
	float extra = 0;
	sum += 0.5*(function_values[0] + function_values[n]);
	if ( n > 1){
		for (int i = 1; i < (n-1); i++){
			sum += function_values[i];
		}
		//extra = (dz*n-H)*function_values[n-1];
	}
	return(sum*dz + extra);
}
__global__ void CoarsePermIntegrationKernel(){

	int xid = blockIdx.x*blockDim.x + threadIdx.x;
    int yid = blockIdx.y*blockDim.y + threadIdx.y;

    if ( xid < cpi_ctx.nx && yid < cpi_ctx.ny ){
		float H = global_index(cpi_ctx.height_distribution.ptr,
							   cpi_ctx.height_distribution.pitch, xid, yid)[0];
		float* k_values = global_index(cpi_ctx.perm_distribution, xid, yid, 0);
		int n = global_index(cpi_ctx.nIntervals_dist.ptr,
							 cpi_ctx.nIntervals_dist.pitch, xid, yid)[0];
		float K = trapezoidal(H, cpi_ctx.dz, n, k_values);
		global_index(cpi_ctx.K.ptr, cpi_ctx.K.pitch, xid, yid)[0] = K;
    }
}

void callCoarsePermIntegrationKernel(dim3 grid, dim3 block, CoarsePermIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(cpi_ctx, args, sizeof(CoarsePermIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	CoarsePermIntegrationKernel<<<grid, block>>>();
}


__global__ void CoarseMobIntegrationKernel(){

	int xid = blockIdx.x*blockDim.x + threadIdx.x;
    int yid = blockIdx.y*blockDim.y + threadIdx.y;

    if ( xid < cmi_ctx.nx && yid < cmi_ctx.ny ){
		// Get full local height
		float H = global_index(cmi_ctx.H_distribution.ptr,
							   cmi_ctx.H_distribution.pitch, xid, yid)[0];
		float h = global_index(cmi_ctx.h_distribution.ptr,
						       cmi_ctx.h_distribution.pitch, xid, yid)[0];
		int nIntervalsForh = ceil(h/cmi_ctx.dz);

		float K = global_index(cmi_ctx.K.ptr, cmi_ctx.K.pitch, xid, yid)[0];

		float* k_values = global_index(cpi_ctx.perm_distribution, xid, yid, 0);

		float L = computeLambda(cmi_ctx.p_ci, cmi_ctx.g, cmi_ctx.delta_rho, h, cmi_ctx.dz, nIntervalsForh,
							    k_values, cmi_ctx.p_cap_ref_table.ptr, cmi_ctx.s_c_ref_table.ptr);
		if (K != 0){
			global_index(cmi_ctx.Lambda.ptr, cmi_ctx.Lambda.pitch, xid, yid)[0] = L/K;
		}
	}
}


void callCoarseMobIntegrationKernel(dim3 grid, dim3 block, CoarseMobIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(cmi_ctx, args, sizeof(CoarseMobIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	CoarseMobIntegrationKernel<<<grid, block>>>();
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



__device__ float trapezoidal(float dz, int n, cudaPitchedPtr function_pointer, int xid, int yid){
	float sum = 0;

	sum += 0.5*(global_index(function_pointer, xid, yid, 0)[0] + global_index(function_pointer, xid, yid, 0)[n]);
	if ( n > 1){
		for (int i = 1; i < (n-1); i++){
			sum += global_index(function_pointer, xid, yid, i)[0];
		}
	}
	return sum*dz;
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


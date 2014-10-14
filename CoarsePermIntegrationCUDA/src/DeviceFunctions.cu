#include "DeviceFunctions.h"
#include "Kernels.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "KernelArgStructs.h"
#include "vector_types.h"
#include "Functions.h"
#include "Util.h"
#include <iostream>
#include <stdio.h>


/*
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
*/

__device__ void multiply(int n, float* x_values, float* y_values, float* product){
	for (int i = 0; i < n; i++){
		product[i] = x_values[i]*y_values[i];
	}
}

/*
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
*/
// Function to create an array of the permeability on the subintervals
/*
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
*/

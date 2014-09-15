#include "functions.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>

void computeRandomHeights(float min, float max, cpu_ptr_2D domain){
	/* initialize random seed: */
	srand (time(NULL));
	for (int j = 0; j < domain.get_ny() ; j++){
		for (int i = 0; i < domain.get_nx(); i++){
			domain(i,j) = rand() % (max-min) + min;
		}
	}
}

// Reference table for conversion between saturation and capillary pressure
void createReferenceTable(float g, float h, float delta_rho, float c_cap, float resolution, float* p_cap_ref_table, float* s_b_ref_table){
	int n = 1/resolution;
	for (int i = 0; i < n+1; i++){
		// Insert equation for analytic capillary pressure curve
		p_cap_ref_table[n-i] = delta_rho*g*h*c_cap*pow(resolution*i,-0.5);
		s_b_ref_table[n-i] = resolution*i;
	}
}

// Function to create an array of the permeability on the subintervals
void kDistribution(float dz, float h, float* k_heights, float* k_data, float* k_values){
	float z = 0;
	int curr_height_index = 0;
	int k_table_index = 0;
	while (z <= h){
		while (z < k_heights[curr_height_index]){
			k_values[k_table_index] = k_heights[curr_height_index];
			z += dz;
			k_table_index++;
		}
		curr_height_index++;
	}
}

// Function to compute the capillary pressure in the subintervals
void computeCapillaryPressure(float p_ci, float g, float delta_rho, float h, float dz, int n, float* p_cap_values){

	for (int i = 0; i < n+1 ; i++){
		p_cap_values[i] = p_ci + g*(-delta_rho)*(dz*i-h);
	}
}

void inverseCapillaryPressure(int n, float* p_cap_values, float* s_b_values){
	// pCap-saturation reference table
	for (int i = 0; i < n+1; i++){
		float curr_p_cap = p_cap_values[i];
		int j = 0;
		while (curr_p_cap > p_cap_ref_table[j]){
			j++;
		}
		s_b_values[i] = s_b_ref_table[j];
	}
}

void computeMobility(int n, float* s_b_values, float lambda_end_point, float* lambda_values){
	for (int i = 0; i < n+1; i++){
		lambda_values[i] = pow(s_b_values[i], 3)*lambda_end_point;
	}
}

void multiply(int n, float* x_values, float* y_values, float* product){
	for (int i = 0; i < n; i++){
		product[i] = x_values[i]*y_values[i];
	}
}

float trapezoidal(float dz, int n, float* function_values){
	float sum = 0;
	sum += 0.5*(function_values[0] + function_values[n]);
	for (int i = 1; i < n; i++){
		sum += function_values[i];
	}
	return sum*dz;
}

void printArray(int n, float* array){
	for (int i = 0; i < n; i++){
		printf("%.3f ", array[i]);
				if (i % 10 == 0){
					printf("\n");
				}
	}
}

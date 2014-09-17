#include "Functions.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>


void computeRandomHeights(float min, float max, CpuPtr_2D &domain){
	/* initialize random seed: */
	srand (time(NULL));
	for (int j = 0; j < domain.get_ny(); j++){
		for (int i = 0; i < domain.get_nx(); i++){
			domain(i,j) = 50 + j - i;
		}
	}
}

// Reference table for conversion between saturation and capillary pressure
void createReferenceTable(float g, float h, float delta_rho, float c_cap, float resolution, float* p_cap_ref_table, float* s_b_ref_table){
	int n = 1/resolution;
	for (int i = 0; i < n+1; i++){
		// Insert equation for analytic capillary pressure curve
		p_cap_ref_table[n-i] = delta_rho*g*h*c_cap*pow(resolution*i,-0.5f);
		s_b_ref_table[n-i] = resolution*i;
	}
}


void multiply(int n, float* x_values, float* y_values, float* product){
	for (int i = 0; i < n; i++){
		product[i] = x_values[i]*y_values[i];
	}
}


void printArray(int n, float* array){
	for (int i = 0; i < n; i++){
		printf("%.3f ", array[i]);
				if (i % 10 == 0){
					printf("\n");
				}
	}
}

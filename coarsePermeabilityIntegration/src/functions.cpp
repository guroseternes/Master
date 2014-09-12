#include "functions.h"

// Reference table for conversion between saturation and capillary pressure
void createReferenceTable(float g, float h, float delta_rho, float resolution, float* p_cap_ref_table, float* s_b_ref_table){
	int n = 1/resolution;
	for (i = 0; i < n+1; i++){
		// Insert equation for analytic capillary pressure curve
		p_cap_ref_table[n+1-i] = delta_rho*g*h*c_cap*pow(resolution*i,-0.5);
		s_b_ref_table[n+1-i] = resolution*i;
	}
}


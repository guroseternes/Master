#include "InitialConditions.h"

InitialConditions::InitialConditions(int nx, int ny, float dz){
	this->nx = nx;
	this->ny = ny;
	this->dz = dz;
	this->integral_res = 0.01f;
	// Density difference between brine and CO2
	this->delta_rho = 686.54-975.86;
	// Gravitational acceleration
	this->g =  9.8066;
	// Non-dimensional constant that scales the strength of the capillary forces
	this->c_cap = 1.0/90.0;
	// Pressure at capillary interface, which is known
	this-> p_ci = 0;
	// Mu-values
	this->mu_c = 0.056641/1000;
	this->mu_b = 0.30860/1000;
	//Residual saturation values
	this->s_b_res = 0.1;
	this->s_c_res = 0.2;
	this->lambda_end_point_c = 1/pow(1-s_b_res,3);
	this->lambda_end_point_b = 1/pow(1-s_c_res,3);

	// Table of capillary pressure values for our subintervals along the z-axis ranging from 0 to h
	this->resolution = 0.01;
	this->size_tables = 1.0/resolution + 1;

	// Permeability data (In real simulations this will be a table based on rock data, here we use a random distribution )
	float data[10] =  {0.9352, 1.0444, 0.9947, 0.9305, 0.9682, 1.0215, 0.9383, 1.0477, 0.9486, 1.0835};
	this->k_data = new float[10];
	for (int i = 0; i < 10; i++){
		this->k_data[i] = data[i];
	}
	float heights[10] =  {10, 20, 150, 220, 240, 255, 301, 323, 380, 400};
	this->k_heights = new float[10];
	for (int i = 0; i < 10; i++){
		this->k_heights[i] = heights[i];
	}
}

/*void InitialConditions::computeRandomHeights(){
	 initialize random seed:
	srand (time(NULL));
	height_distribution = CpuPtr_2D(nx, ny, 0, true);
	for (int j = 0; j < nx; j++){
		for (int i = 0; i < ny; i++){
			height_distribution(i,j) = 50 + j - i;
		}
	}
}*/

void InitialConditions::createScalingParameterTable(CpuPtr_2D H){
	scaling_parameter = CpuPtr_2D(nx,ny,0,true);
	for (int j = 0; j < ny; j++){
		for (int i = 0; i < nx; i++){
			scaling_parameter(i,j) = 0.4*(-g)*delta_rho*H(i,j);
		}
	}
}

void InitialConditions::createInitialCoarseSatu(CpuPtr_2D H, CpuPtr_2D h){
	initial_coarse_satu_c = CpuPtr_2D(nx, ny, 0, true);
	float res = integral_res/100;
	for (int j = 0; j < ny; j++){
			for (int i = 0; i < nx; i++){
				initial_coarse_satu_c(i,j) = computeCoarseSaturation(p_ci, g, delta_rho, s_b_res, h(i,j), res, ceil(h(i,j)/res),
																	 scaling_parameter(i,j), H(i,j));

				//h(i,j) = computeCoarseSaturation(p_ci, g, delta_rho, s_b_res, h(i,j), integral_res, 1,
					//												 scaling_parameter(i,j), H(i,j));
			}
		}
}

// Reference table for conversion between saturation and capillary pressure
void InitialConditions::createReferenceTable(){
	int n = 1/resolution;
	p_cap_ref_table = new float[n];
	s_c_ref_table = new float[n];
	for (int i = 0; i < n; i++){
		// Insert equation for analytic capillary pressure curve
		p_cap_ref_table[n-i-1] = delta_rho*g*max_height*c_cap*pow(1-resolution*i,-0.5f);
		s_c_ref_table[n-i-1] = 1;//resolution*i;
	}
}

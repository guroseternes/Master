#include "InitialConditions.h"

InitialConditions::InitialConditions(int nx, int ny, float max_height){
	this->nx = nx;
	this->ny = ny;
	this->max_height = max_height;
	this->dz = 1;
	// Density difference between brine and CO2
	this->delta_rho = 500;
	// Gravitational acceleration
	this->g = 9.87;
	// Non-dimensional constant that scales the strength of the capillary forces
	this->c_cap = 1.0/6.0;
	// Pressure at capillary interface, which is known
	this-> p_ci = 1;
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

// Reference table for conversion between saturation and capillary pressure
void InitialConditions::createReferenceTable(){
	int n = 1/resolution;
	p_cap_ref_table = new float[n+1];
	s_b_ref_table = new float[n+1];
	for (int i = 0; i < n+1; i++){
		// Insert equation for analytic capillary pressure curve
		p_cap_ref_table[n-i] = delta_rho*g*H*c_cap*pow(resolution*i,-0.5f);
		s_b_ref_table[n-i] = resolution*i;
	}
}

#include "InitialConditions.h"

InitialConditions::InitialConditions(int nx, int ny, float dz){
	this->nx = nx;
	this->ny = ny;
	this->dz = dz;
	this->border = 1;

	this->cfl_scale = 0.5;
	this->dt_test = 4.3711 * pow((float)10, 7);
	this->global_time_data[0] = 0;
	this->global_time_data[1] = 0;
	//tf
	this->global_time_data[2] = 63072000;

	this->integral_res = 1.0f;
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
	this->lambda_end_point_c = 0.2142;
	float temp = (1-s_c_res-s_b_res)/(1-s_b_res);
	this->lambda_end_point_b = 0.85;//1/pow(temp,3);

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
void InitialConditions::computeAllGridBlocks(){
	computeGridBlock(grid, block, nx, ny, BLOCKDIM_X_FLUX, BLOCKDIM_Y_FLUX);
	computeGridBlock(grid_flux, block_flux, nx + 2*border, ny + 2*border, BLOCKDIM_X_FLUX,
			BLOCKDIM_Y_FLUX, TILEDIM_X, TILEDIM_Y);
}

void InitialConditions::createDtVec(){
	dt_vector = new float[nElements];
	for (int k = 0; k < nElements; k++){
			dt_vector[k] = FLT_MAX;
	}
}

void InitialConditions::createnIntervalsTable(CpuPtr_2D H){
	nIntervals = CpuPtr_2D(nx,ny,0,true);
	for (int j = 0; j < ny; j++){
		for (int i = 0; i < nx; i++){
			nIntervals(i, j) = ceil(H(i, j) / dz);
		}
	}
}

void InitialConditions::createScalingParameterTable(CpuPtr_2D H){
	scaling_parameter = CpuPtr_2D(nx,ny,0,true);
	for (int j = 0; j < ny; j++){
		for (int i = 0; i < nx; i++){
			scaling_parameter(i,j) = 0.1*(-g)*delta_rho*H(i,j);
		}
	}
}

void InitialConditions::createInitialCoarseSatu(CpuPtr_2D H, CpuPtr_2D h){
	initial_coarse_satu_c = CpuPtr_2D(nx, ny, 0, true);
	float res = integral_res/100;
	for (int j = 0; j < ny; j++){
			for (int i = 0; i < nx; i++){
				//initial_coarse_satu_c(i,j) = computeCoarseSaturation(p_ci, g, delta_rho, s_b_res, h(i,j), res, ceil(h(i,j)/res),
																	// scaling_parameter(i,j), H(i,j));
				initial_coarse_satu_c(i,j) = h(i,j)*(1-s_b_res);
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

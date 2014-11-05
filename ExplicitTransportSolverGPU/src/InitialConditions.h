#ifndef INITIALCONDITIONS_H_
#define INITIALCONDITIONS_H_

#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <vector>
#include <math.h>
#include "CpuPtr.h"
#include "Util.h"
#include "Kernels.h"

class InitialConditions{
public:
	InitialConditions(int nx, int ny, float max_height);
	void createScalingParameterTable(CpuPtr_2D H);
	void createInitialCoarseSatu(CpuPtr_2D H, CpuPtr_2D h);
	void createReferenceTable();
	void createnIntervalsTable(CpuPtr_2D H);
	void computeAllGridBlocks();
	void createDtVec();
	//void computeRandomHeights();

	unsigned int nx;
	unsigned int ny;
	unsigned int border;
	float dz;
	float integral_res;
	float max_height;

	float cfl_scale;
	float dt_test;

	float delta_rho;
	float g;
	float mu_c;
	float mu_b;
	float s_b_res;
	float s_c_res;

	float c_cap;
	float p_ci;
	float lambda_end_point_c;
	float lambda_end_point_b;

	// Permeability data
	float* k_data;
	float* k_heights;

	// Saturation - Permeability tables
	float resolution;
	float* p_cap_ref_table;
	float* s_c_ref_table;
	float size_tables;

	CpuPtr_2D nIntervals;
	CpuPtr_2D initial_coarse_satu_c;
	CpuPtr_2D scaling_parameter;

	//Block and grid
	dim3 grid, block, block_flux, grid_flux;
	int nElements;
	float* dt_vector;


};

#endif /* INITIALCONDITIONS_H_ */

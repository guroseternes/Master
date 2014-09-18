#ifndef INITIALCONDITIONS_H_
#define INITIALCONDITIONS_H_

#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <vector>
#include <math.h>
#include "CpuPtr.h"

struct CoarsePermIC {

};

class InitialConditions{
public:
	InitialConditions(int nx, int ny, float H);
	void createReferenceTable();
	void computeRandomHeights();

	unsigned int nx;
	unsigned int ny;
	float dz;
	//max_height
	float H;

	float delta_rho;
	float g;
	float c_cap;
	float p_ci;

	// Permeability data
	float* k_data;
	float* k_heights;

	// Saturation - Permeability tables
	float resolution;
	float* p_cap_ref_table;
	float* s_b_ref_table;
	float size_tables;

	CpuPtr_2D height_distribution;

private:
	CoarsePermIC coarse_perm_IC;
};

#endif /* INITIALCONDITIONS_H_ */

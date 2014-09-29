#ifndef KARGS_H_
#define KARGS_H_

#include "GpuPtr.h"

struct CoarsePermIntegrationKernelArgs {
	GpuRawPtr K;
	GpuRawPtr height_distribution;
	GpuRawPtr nIntervals_dist;
	cudaPitchedPtr perm_distribution;
	float dz;
	unsigned int nx, ny, nz;
	unsigned int border;
};

struct CoarseMobIntegrationKernelArgs {
	GpuRawPtr Lambda;
	GpuRawPtr height_distribution;
	GpuRawPtr k_data;
	GpuRawPtr k_heights;
	GpuRawPtr p_cap_ref_table;
	GpuRawPtr s_b_ref_table;
	float dz;
	float g, p_ci, delta_rho;
	unsigned int nx, ny;
	unsigned int border;
};

#endif /* KARGS_H_ */

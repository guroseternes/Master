#ifndef KARGS_H_
#define KARGS_H_

#include "GpuPtr.h"

struct CoarsePermIntegrationKernelArgs {
	GpuRawPtr K;
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

#ifndef KARGS_H_
#define KARGS_H_

#include "GpuPtr.h"

struct CommonArgs {
	GpuRawPtr H;
	GpuRawPtr p_cap_ref_table;
	GpuRawPtr s_c_ref_table;
	float g, delta_rho, mu_c, mu_b;
	unsigned int nx, ny, nz;
	unsigned int border;
};

struct CoarsePermIntegrationKernelArgs {
	GpuRawPtr K;
	GpuRawPtr nIntervals_dist;
	cudaPitchedPtr perm_distribution;
	float dz;
};

struct CoarseMobIntegrationKernelArgs {
	GpuRawPtr Lambda;
	GpuRawPtr h;
	GpuRawPtr K;
	GpuRawPtr nIntervals;
	cudaPitchedPtr perm_distribution;
	GpuRawPtrInt active_block_indexes;
	float dz;
	float p_ci;
};

struct FluxKernelArgs  {
	GpuRawPtr Lambda_c;
	GpuRawPtr Lambda_b;
	GpuRawPtr U_x;
	GpuRawPtr U_y;
	GpuRawPtr h;
	GpuRawPtr z;
};

struct TimeIntegrationKernelArgs {
	GpuRawPtr R;
	GpuRawPtr S_c;
	float p_ci;
	float dz;
	float* dt;
};

#endif /* KARGS_H_ */

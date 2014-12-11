#ifndef KARGS_H_
#define KARGS_H_

#include "GpuPtr.h"

struct CommonArgs {
	GpuRawPtr H;
	GpuRawPtr active_east;
	GpuRawPtr active_north;
	GpuRawPtr pv;
	float p_ci;
	float g, delta_rho, mu_c, mu_b;
	float s_c_res, s_b_res;
	float lambda_end_point_c, lambda_end_point_b;
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
	GpuRawPtr Lambda_c;
	GpuRawPtr Lambda_b;
	GpuRawPtr dLambda_c;
	GpuRawPtr dLambda_b;
	GpuRawPtr h;
	GpuRawPtr K;
	GpuRawPtr nIntervals;
	cudaPitchedPtr perm_distribution;
	GpuRawPtr scaling_parameter_C;
	GpuRawPtrInt active_block_indexes;
	float dz;
	float p_ci;
};

struct FluxKernelArgs  {
	GpuRawPtr R;
	GpuRawPtr Lambda_c;
	GpuRawPtr Lambda_b;
	GpuRawPtr dLambda_c;
	GpuRawPtr dLambda_b;
	GpuRawPtr U_x;
	GpuRawPtr U_y;
	GpuRawPtr h;
	GpuRawPtr z;
	GpuRawPtr normal_z;
	GpuRawPtr K_face_east;
	GpuRawPtr K_face_north;
	GpuRawPtr g_vec_east;
	GpuRawPtr g_vec_north;
	GpuRawPtr source;
	GpuRawPtr test_output;
	GpuRawPtrInt active_block_indexes;

	float* dt_vector;
};

struct TimeIntegrationKernelArgs {
	GpuRawPtr pv;
	GpuRawPtr h;
	GpuRawPtr R;
	GpuRawPtr S_c;
	GpuRawPtr vol_old;
	GpuRawPtr vol_new;
	GpuRawPtr scaling_parameter_C;
	float dz;
	float* global_dt;
};

struct TimestepReductionKernelArgs {
	int nThreads;
	int nElements;
	float cfl_scale;
	float* global_dt;
	float* dt_vec;
};

#endif /* KARGS_H_ */

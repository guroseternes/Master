#include "Util.h"

void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y){
        block.x = block_x;
        block.y = block_y;
        grid.x = (NX + block_x - 1)/block_x;
        grid.y = (NY + block_y - 1)/block_y;
}


void setCoarsePermIntegrationKernelArgs(CoarsePermIntegrationKernelArgs* args, GpuRawPtr K, GpuRawPtr H,
										cudaPitchedPtr k, GpuRawPtr nI,
										float dz, unsigned int nx, unsigned int ny, unsigned int nz,
										unsigned int border){
	args->K = K;
	args->height_distribution = H;
	args->perm_distribution = k;
	args->nIntervals_dist = nI;

	args->dz = dz;
	args->nx = nx;
	args->ny = ny;
	args->nz = nz;
	args->border = border;
}

void setCoarseMobIntegrationArgs(CoarseMobIntegrationKernelArgs* args,
								  GpuRawPtr Lambda, GpuRawPtr H, GpuRawPtr h, cudaPitchedPtr k,
								  GpuRawPtr K, GpuRawPtr nI,
								  GpuRawPtr p_cap_ref_table,
								  GpuRawPtr s_c_ref_table,
								  float p_ci, float g, float delta_rho,
								  float dz, unsigned int nx, unsigned int ny, unsigned int border){
	args->Lambda = Lambda;
	args->H_distribution = H;
	args->h_distribution = h;
	args->perm_distribution = k;
	args->K = K;
	args->nIntervals = nI;

	args->p_cap_ref_table = p_cap_ref_table;
	args->s_c_ref_table = s_c_ref_table;

	args->p_ci = p_ci;
	args->g = g;
	args->delta_rho = delta_rho;

	args->dz = dz;
	args->nx = nx;
	args->ny = ny;
	args->border = border;
}

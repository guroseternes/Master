#include "Util.h"

void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y){
        block.x = block_x;
        block.y = block_y;
        grid.x = (NX + block_x - 1)/block_x;
        grid.y = (NY + block_y - 1)/block_y;
}

void setCoarsePermIntegrationArgs(CoarsePermIntegrationKernelArgs* args,
								  GpuRawPtr K, GpuRawPtr h, GpuRawPtr k_data,
								  GpuRawPtr k_heights, GpuRawPtr p_cap_ref_table,
								  GpuRawPtr s_b_ref_table,
								  float p_ci, float g, float delta_rho,
								  float dz, unsigned int nx, unsigned int ny, unsigned int border){
	args->K = K;
	args->height_distribution = h;
	args->k_data = k_data;
	args->k_heights = k_heights;
	args->p_cap_ref_table = p_cap_ref_table;
	args->s_b_ref_table = s_b_ref_table;

	args->p_ci = p_ci;
	args->g = g;
	args->delta_rho = delta_rho;

	args->dz = dz;
	args->nx = nx;
	args->ny = ny;
	args->border = border;
}

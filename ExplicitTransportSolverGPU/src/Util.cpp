#include "Util.h"

void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y){
        block.x = block_x;
        block.y = block_y;
        grid.x = (NX + block_x - 1)/block_x;
        grid.y = (NY + block_y - 1)/block_y;
}

void createGridMask(CpuPtr_2D H, dim3 grid, dim3 block, int nx, int ny, std::vector<int> &activeBlockIndexes, int& nActiveBlocks){
	int xid, yid;
	int size = grid.x*grid.y;
	// Array to keep track of which block that are active
	int activeBlocks[size];
	for (int k = 0; k < size; k++){
		activeBlocks[k] = -1;
	}
	for (int j = 0; j < grid.y; j++){
		for (int i = 0; i < grid.x; i++){
			for (int y = 0; y < block.y; y++){
				for (int x = 0; x < block.x; x++){
					xid = i*block.x + x;
					yid = j*block.y + y;
					if (xid < nx && yid < ny){
						if (H(xid, yid) > 0){
								activeBlocks[grid.x*j + i] = 0;
						} else {
							H(xid, yid) = 0; //Change this later
						}
					}
				}
			}
		}
	}

	for (int k = 0; k < size; k++){
		if (activeBlocks[k] == 0){
			activeBlockIndexes.push_back(k);
			nActiveBlocks++;
		}
	}
}

void setCommonArgs(CommonArgs* args, float delta_rho, float g, float mu_c, float mu_b,
				   GpuRawPtr H, GpuRawPtr p_cap_ref_table, GpuRawPtr s_c_ref_table,
			       unsigned int nx, unsigned int ny, unsigned int border){
	args->H = H;
	args->p_cap_ref_table = p_cap_ref_table;
	args->s_c_ref_table = s_c_ref_table;

	args->delta_rho = delta_rho;
	args->g = g;
	args->mu_c = mu_c;
	args->mu_b = mu_b;

	args->nx = nx;
	args->ny = ny;
	args->border = border;
}

void setCoarsePermIntegrationKernelArgs(CoarsePermIntegrationKernelArgs* args,
									    GpuRawPtr K, cudaPitchedPtr k, GpuRawPtr nI, float dz){
	args->K = K;
	args->perm_distribution = k;
	args->nIntervals_dist = nI;

	args->dz = dz;
}

void setCoarseMobIntegrationKernelArgs(CoarseMobIntegrationKernelArgs* args,
								  GpuRawPtr Lambda, GpuRawPtr h, cudaPitchedPtr k,
								  GpuRawPtr K, GpuRawPtr nI,
								  GpuRawPtrInt a_b_i,
								  float p_ci,float dz){
	args->Lambda = Lambda;
	args->h = h;
	args->perm_distribution = k;
	args->K = K;
	args->nIntervals = nI;

	args->active_block_indexes = a_b_i;

	args->p_ci = p_ci;
	args->dz = dz;
}

void setFluxKernelArgs(FluxKernelArgs* args,
					   GpuRawPtr Lambda_c, GpuRawPtr Lambda_b,
					   GpuRawPtr U_x, GpuRawPtr U_y,
					   GpuRawPtr h, GpuRawPtr z){
	args->Lambda_c = Lambda_c;
	args->Lambda_b = Lambda_b;
	args->U_x = U_x;
	args->U_y = U_y;
	args->h = h;
	args->z = z;
}

void setTimeIntegrationKernelArgs(TimeIntegrationKernelArgs* args,
								  GpuRawPtr F, GpuRawPtr S_c){
	args->R = F;
	args->S_c = S_c;
}

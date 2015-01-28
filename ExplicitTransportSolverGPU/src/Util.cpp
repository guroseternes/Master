#include "Util.h"
#include "Kernels.h"
#include "Config.h"


void computeGridBlockBisection(dim3& grid, dim3& block, int NX, int n_cells_per_block){
        int threads_per_cell = 64;
		int num_threads = NX*threads_per_cell;
        block.x = n_cells_per_block*threads_per_cell;
        grid.x = (NX + block.x - 1)/block.x;
}

void computeGridBlock(dim3& grid, dim3& block, int NX, int block_x){
        block.x = block_x;
        grid.x = (NX + block_x - 1)/block_x;
}

void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y){
        block.x = block_x;
        block.y = block_y;
        grid.x = (NX + block_x - 1)/block_x;
        grid.y = (NY + block_y - 1)/block_y;
}

void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y, int tile_x, int tile_y){
        block.x = block_x;
        block.y = block_y;
        grid.x = (NX + tile_x - 1)/tile_x;
        grid.y = (NY + tile_y - 1)/tile_y;
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

void createGridMaskFlux(CpuPtr_2D H, dim3 grid, dim3 block, int nx, int ny, std::vector<int> &activeBlockIndexes, int& nActiveBlocks){
	int xid, yid;
	int size = grid.x*grid.y;
	int border = 1;

	// Array to keep track of which block that are active
	int activeBlocks[size];
	for (int k = 0; k < size; k++){
		activeBlocks[k] = -1;
	}
	for (int j = 0; j < grid.y; j++){
		for (int i = 0; i < grid.x; i++){
			for (int y = 0; y < block.y; y++){
				for (int x = 0; x < block.x; x++){
					int xid = i*TILEDIM_X + x - border;
				    int yid = j*TILEDIM_Y + y-border;
				    xid = fmaxf(xid, 0);
				    yid = fmaxf(yid, 0);
					if (xid < nx && yid < ny){
						if (H(xid, yid) > 0){
								activeBlocks[grid.x*j + i] = 0;
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

void setCommonArgs(CommonArgs* args, float p_ci, float delta_rho, float g, float mu_c, float mu_b,
				   float s_c_res, float s_b_res, float l_e_p_c, float l_e_p_b,
				   GpuRawPtr active_east, GpuRawPtr active_north,
				   GpuRawPtr H, GpuRawPtr pv,
			       unsigned int nx, unsigned int ny, unsigned int border){
	args->active_east = active_east;
	args->active_north = active_north;
	args->H = H;
	args->pv = pv;

	args->delta_rho = delta_rho;
	args->g = g;
	args->mu_c = mu_c;
	args->mu_b = mu_b;

	args->s_b_res = s_b_res;
	args->s_c_res = s_c_res;

	args->p_ci = p_ci;

	args->lambda_end_point_c = l_e_p_c;
	args->lambda_end_point_b = l_e_p_b;

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
								  GpuRawPtr Lambda_c, GpuRawPtr Lambda_b,
								  GpuRawPtr dLambda_c, GpuRawPtr dLambda_b,
								  GpuRawPtr h, cudaPitchedPtr k,
								  GpuRawPtr K, GpuRawPtr nI, GpuRawPtr scaling_para_C,
								  GpuRawPtrInt a_b_i,
								  float p_ci,float dz, Perm perm_type){
	args->Lambda_c = Lambda_c;
	args->Lambda_b = Lambda_b;
	args->dLambda_c = dLambda_c;
	args->dLambda_b = dLambda_b;
	args->h = h;
	args->perm_distribution = k;
	args->K = K;
	args->nIntervals = nI;
	args->scaling_parameter_C = scaling_para_C;

	args->active_block_indexes = a_b_i;

	args->p_ci = p_ci;
	args->dz = dz;

	args->perm_type = perm_type;
}

void setFluxKernelArgs(FluxKernelArgs* args,
					   GpuRawPtr Lambda_c, GpuRawPtr Lambda_b,
					   GpuRawPtr dLambda_c, GpuRawPtr dLambda_b,
					   GpuRawPtr U_x, GpuRawPtr U_y, GpuRawPtr source,
					   GpuRawPtr h, GpuRawPtr z,
					   GpuRawPtr normal_z,
					   GpuRawPtr K_face_east, GpuRawPtr K_face_north,
					   GpuRawPtr g_vec_east, GpuRawPtr g_vec_north,
					   GpuRawPtr R, float* dt_vector, GpuRawPtrInt a_b_i,
					   GpuRawPtr test_output){
	args->Lambda_c = Lambda_c;
	args->Lambda_b = Lambda_b;
	args->dLambda_c = dLambda_c;
	args->dLambda_b = dLambda_b;
	args->test_output = test_output;
	args->U_x = U_x;
	args->U_y = U_y;
	args->source = source;
	args->h = h;
	args->z = z;

	args->normal_z = normal_z;
	args->K_face_east = K_face_east;
	args->K_face_north = K_face_north;
	args->g_vec_east = g_vec_east;
	args->g_vec_north = g_vec_north;
	args->R = R;

	args->active_block_indexes = a_b_i;
	args->dt_vector = dt_vector;
}

void setTimeIntegrationKernelArgs(TimeIntegrationKernelArgs* args, float* global_dt, float dz,
								  GpuRawPtr pv, GpuRawPtr h, GpuRawPtr F,
								  GpuRawPtr S_c, GpuRawPtr scaling_para_C,
								  GpuRawPtr vol_old, GpuRawPtr vol_new, unsigned int* d_isValid, int* d_in){
	args->global_dt = global_dt;
	args->vol_old = vol_old;
	args->vol_new = vol_new;
	args->dz = dz;
	args->pv = pv;
	args->h = h;
	args->R = F;
	args->S_c = S_c;
	args->scaling_parameter_C = scaling_para_C;

	args->d_in = d_in;
	args->d_isValid = d_isValid;
}

void setTimestepReductionKernelArgs(TimestepReductionKernelArgs* args, int nThreads, int nElements,
									float* global_dt, float cfl_scale, float* dt_vec){
	args->nThreads = nThreads;
	args->nElements = nElements;
	args->global_dt = global_dt;
	args->cfl_scale = cfl_scale;
	args->dt_vec = dt_vec;
}

void setSolveForhProblemCellsKernelArgs(SolveForhProblemCellsKernelArgs* args, GpuRawPtr h,
									    GpuRawPtr S_c, GpuRawPtr scaling_parameter_C,
									    int* d_out, float dz, size_t* d_numValid){

	args->dz = dz;
	args->h = h;
	args->S_c = S_c;
	args->scaling_parameter_C = scaling_parameter_C;
	args->d_out = d_out;
	args->d_numValid = d_numValid;
}

float computeBrineSaturation(float p_cap, float C, float s_b_res){
	return fmaxf(C*C/((C+p_cap)*(C+p_cap)), s_b_res);
}

float computeCoarseSaturationSharpInterface(float h, float H){
	return h/H;
}


// Function to compute the capillary pressure in the subintervals
float computeCoarseSaturation(float p_ci, float g, float delta_rho, float s_b_res, float h, float dz, int n,
					    float scaling_parameter_C, float H){
	float current_p_cap = p_ci + g*(delta_rho)*(dz*0-h);
    //printf("n: %i h: %.15f", n, h);
	float current_satu_c = 0;
	float prev_satu_c = 1-computeBrineSaturation(current_p_cap, scaling_parameter_C, s_b_res);
	bool corr_h = false;
	float sum_c = 0;
	if (n>0){
		for (int i = 1; i < n; i++){
			current_p_cap = p_ci + g*(delta_rho)*(dz*i-h);
			current_satu_c = 1-computeBrineSaturation(current_p_cap, scaling_parameter_C, s_b_res);
			sum_c += dz*0.5*(current_satu_c+prev_satu_c);
			prev_satu_c = current_satu_c;
		}
			current_p_cap = p_ci + g*(delta_rho)*(h-h);
			current_satu_c = 1-computeBrineSaturation(current_p_cap, scaling_parameter_C, s_b_res);

			sum_c += 0.5*(prev_satu_c+current_satu_c)*(h-dz*(n-1));
	}
	if (h > 1.126 && h < 1.128){
		float C = scaling_parameter_C;
		printf("sum_c%.4f \n", sum_c);
	}
	return sum_c;
}


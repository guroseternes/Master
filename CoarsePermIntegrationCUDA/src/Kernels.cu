#include "Kernels.h"

__constant__ FluxKernelArgs fk_ctx;
__constant__ CommonArgs common_ctx;
__constant__ TimeIntegrationKernelArgs tik_ctx;
__constant__ CoarsePermIntegrationKernelArgs cpi_ctx;
__constant__ CoarseMobIntegrationKernelArgs cmi_ctx;

void initAllocate(CommonArgs* args1, CoarsePermIntegrationKernelArgs* args2,
				  CoarseMobIntegrationKernelArgs* args3, FluxKernelArgs* args4,
				  TimeIntegrationKernelArgs* args5){
	cudaHostAlloc(&args1, sizeof(CommonArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args2, sizeof(CoarsePermIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args3, sizeof(CoarseMobIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args4, sizeof(FluxKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args5, sizeof(TimeIntegrationKernelArgs), cudaHostAllocWriteCombined);
}

void setupGPU(CommonArgs* args){
	cudaMemcpyToSymbolAsync(common_ctx, args, sizeof(CommonArgs), 0, cudaMemcpyHostToDevice);
}

inline __device__ float* global_index(float* base, unsigned int pitch, int x, int y, int border) {
        return (float*) ((char*) base+(y+border)*pitch) + (x+border);
}

inline __device__ float* global_index(cudaPitchedPtr ptr, int x, int y, int z, int border) {
        return (float*) ((char*) (ptr.ptr+(x+border)*(ptr.ysize)*ptr.pitch)) + (y+border)*(ptr.pitch/sizeof(float)) + z;
}

__device__ float lookupSaturation(float curr_p_cap, float* p_cap_ref_table, float* s_c_ref_table){
	int j = 0;
	while (curr_p_cap < p_cap_ref_table[j] && j < 99) {
		j++;
	}
	return s_c_ref_table[j];
}

__device__ float computeMobility(float s_c){
	float lambda_end_point = 1;
		return (pow(s_c, 3)*lambda_end_point);
}

__device__ float trapezoidal(float H, float dz, int n, float* function_values){
	float sum = 0;
	float extra = 0;
	for (int i = 0; i < (n-1); i++){
		sum += function_values[i] + function_values[i+1];
	}
	if (n >= 1){
		extra = (H-dz*(n-1))*(function_values[n-1] +function_values[n]);
	}

	return 0.5*(sum*dz + extra);
}

inline __device__ void computeFluxEast(float (&U)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   float (&lambda_c)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   float (&lambda_b)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   float (&h)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   float (&z)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   unsigned int i, unsigned int j){
	float face_mob_c, face_mob_b, tot_mob, F_c;
	float h_diff = h[i][j]-h[i+1][j];
	float z_diff = z[i][j]-z[i+1][j];
	float b = z_diff + h_diff;
	float g_flux = common_ctx.g*b*common_ctx.delta_rho;

	// Determine the upwind cell evaluation for the two phases
	if (g_flux*U[i][j] >= 0) {
		if (U[i] > 0){
			face_mob_c = lambda_c[i][j];
		} else {
			face_mob_c = lambda_c[i+1][j];
		}
		if ((U[i][j] - face_mob_c*g_flux) > 0){
			face_mob_b = lambda_b[i][j];
		} else {
			face_mob_b = lambda_b[i+1][j];
		}
	} else {
		if (U[i] > 0) {
			face_mob_b = lambda_b[i][j];
		} else {
			face_mob_b = lambda_b[i+1][j];
		}
		if (U[i][j] + face_mob_b*g_flux > 0) {
			face_mob_c = lambda_c[i][j];
		} else {
			face_mob_c = lambda_c[i+1][j];
		}
	}

	tot_mob = face_mob_c + face_mob_b;
	F_c = face_mob_c/tot_mob;
	float flux = F_c*(U[i][j]+face_mob_b*g_flux);

}

inline __device__ void computeFluxNorth(float (&U)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   float (&lambda_c)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   float (&lambda_b)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   float (&h)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   float (&z)[BLOCKDIM_X][SM_BLOCKDIM_Y],
									   unsigned int i, unsigned int j){
	float face_mob_c, face_mob_b, tot_mob, F_c;
	float h_diff = h[i][j]-h[i][j+1];
	float z_diff = z[i][j]-z[i][j+1];
	float b = z_diff + h_diff;
	float g_flux = common_ctx.g*b*common_ctx.delta_rho;

	// Determine the upwind cell evaluation for the two phases
	if (g_flux*U[i][j] >= 0) {
		if (U[i] > 0){
			face_mob_c = lambda_c[i][j];
		} else {
			face_mob_c = lambda_c[i][j+1];
		}
		if ((U[i][j] - face_mob_c*g_flux) > 0){
			face_mob_b = lambda_b[i][j];
		} else {
			face_mob_b = lambda_b[i][j+1];
		}
	} else {
		if (U[i] > 0) {
			face_mob_b = lambda_b[i][j];
		} else {
			face_mob_b = lambda_b[i][j+1];
		}
		if (U[i][j] + face_mob_b*g_flux > 0) {
			face_mob_c = lambda_c[i][j];
		} else {
			face_mob_c = lambda_c[i][j+1];
		}
	}

	tot_mob = face_mob_c + face_mob_b;
	F_c = face_mob_c/tot_mob;
	float flux = F_c*(U[i][j]+face_mob_b*g_flux);

}

__global__ void FluxKernel(){

	int border = common_ctx.border;
	//float r = FLT_MAX;

	// Global id
	int xid = blockIdx.x*TILEDIM_X + threadIdx.x-border;
    int yid = blockIdx.y*TILEDIM_Y + threadIdx.y-border;

    xid = fminf(xid, common_ctx.nx);
    yid = fminf(yid, common_ctx.ny);

    // Local id
    int i = threadIdx.x;
    int j = threadIdx.y;

    __shared__ float U_local_x[BLOCKDIM_X][SM_BLOCKDIM_Y];
    __shared__ float U_local_y[BLOCKDIM_X][SM_BLOCKDIM_Y];
    __shared__ float lambda_c_local[BLOCKDIM_X][SM_BLOCKDIM_Y];
    __shared__ float lambda_b_local[BLOCKDIM_X][SM_BLOCKDIM_Y];
    __shared__ float h_local[BLOCKDIM_X][SM_BLOCKDIM_Y];
    __shared__ float z_local[BLOCKDIM_X][SM_BLOCKDIM_Y];

    U_local_x[i][j] = global_index(fk_ctx.U_x.ptr, fk_ctx.U_x.pitch, xid, yid, border)[0];
    U_local_y[i][j] = global_index(fk_ctx.U_y.ptr, fk_ctx.U_y.pitch, xid, yid, border)[0];
    lambda_c_local[i][j] = global_index(fk_ctx.Lambda_c.ptr, fk_ctx.Lambda_c.pitch, xid, yid, border)[0];
    lambda_b_local[i][j] = global_index(fk_ctx.Lambda_b.ptr, fk_ctx.Lambda_b.pitch, xid, yid, border)[0];
    h_local[i][j] = global_index(fk_ctx.h.ptr, fk_ctx.z.pitch, xid, yid, border)[0];
    z_local[i][j] = global_index(fk_ctx.z.ptr, fk_ctx.z.pitch, xid, yid, border)[0];

    __syncthreads();

    if (i < TILEDIM_X && j < TILEDIM_Y) {
    	computeFluxEast(U_local_x, lambda_c_local, lambda_b_local, h_local, z_local, i, j);
    	computeFluxNorth(U_local_y, lambda_c_local, lambda_b_local, h_local, z_local, i, j);
    }
}

void callFluxKernel(dim3 grid, dim3 block, FluxKernelArgs* args){
	cudaMemcpyToSymbolAsync(fk_ctx, args, sizeof(FluxKernelArgs), 0, cudaMemcpyHostToDevice);
	FluxKernel<<<grid, block>>>();
}

inline __device__ float solveForh(float S_c_new, float H, float p_ci, float dz, float delta_rho, float g,
								  float* p_cap_ref_table, float* s_c_ref_table){
	float h = 0;
	float sum = 0;
	float curr_s_c = 0;
	float z = 0;
	float curr_p_cap = p_ci + g*delta_rho*(z-h);
	float prev_s_c = lookupSaturation(curr_p_cap, p_cap_ref_table, s_c_ref_table);
	while (sum < S_c_new){
		z += dz;
		curr_p_cap = p_ci + g*delta_rho*(z-h);
		curr_s_c = lookupSaturation(curr_p_cap, p_cap_ref_table, s_c_ref_table);
		sum += 0.5*(1/H)*dz*(curr_s_c + prev_s_c);
		prev_s_c = curr_s_c;
	}
	return z;
}

__global__ void TimeIntegrationKernel(){

	// Global id
	int border = common_ctx.border;
	int dt = tik_ctx.dt[0];
	int xid = blockIdx.x*blockDim.x + threadIdx.x-border;
    int yid = blockIdx.y*blockDim.y + threadIdx.y-border;

    float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, xid, yid, border)[0];
    float S_c_old, S_c_new;
    S_c_old = global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, border)[0];
    float r = global_index(tik_ctx.R.ptr, tik_ctx.R.pitch, xid, yid, border)[0];
    S_c_new = S_c_old + dt*r;
    global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, border)[0] = S_c_new;
    float h = solveForh(S_c_new, H, tik_ctx.p_ci, tik_ctx.dz, common_ctx.delta_rho, common_ctx.g,
    			        common_ctx.p_cap_ref_table.ptr, common_ctx.s_c_ref_table.ptr);

}

void callTimeIntegration(dim3 grid, dim3 block, TimeIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(tik_ctx, args, sizeof(TimeIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	TimeIntegrationKernel<<<grid, block>>>();
}

// Function to compute the capillary pressure in the subintervals
__device__ float computeLambda(float p_ci, float g, float delta_rho, float h, float dz, int n,
							   float* k_values, float* p_cap_ref_table, float* s_c_ref_table){
	float curr_mobk = 0;
	float current_p_cap = p_ci + g*(-delta_rho)*(dz*0-h);
	float current_satu_c = lookupSaturation(current_p_cap, p_cap_ref_table, s_c_ref_table);;
	float current_mob = computeMobility(current_satu_c);
	float prev_mobk = current_mob*k_values[0];
	float sum = 0;
	for (int i = 1; i < n; i++){
		current_p_cap = p_ci + g*(-delta_rho)*(dz*i-h);
		current_satu_c = lookupSaturation(current_p_cap, p_cap_ref_table, s_c_ref_table);
		current_mob = computeMobility(current_satu_c);
		curr_mobk = current_mob*k_values[i];
		sum += dz*0.5*(curr_mobk+prev_mobk);
		prev_mobk = curr_mobk;
	}
		current_p_cap = p_ci + g*(-delta_rho)*(h-h);
		current_satu_c = lookupSaturation(current_p_cap, p_cap_ref_table, s_c_ref_table);
		current_mob = computeMobility(current_satu_c);
		curr_mobk = current_mob*k_values[n];
		sum += 0.5*(prev_mobk+curr_mobk)*(h-dz*(n-1));
		return sum;
}

__global__ void CoarsePermIntegrationKernel(){

	int xid = blockIdx.x*blockDim.x + threadIdx.x;
    int yid = blockIdx.y*blockDim.y + threadIdx.y;

    if ( xid < common_ctx.nx && yid < common_ctx.ny ){
		float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, xid, yid,0)[0];
		float* k_values = global_index(cpi_ctx.perm_distribution, xid, yid, 0, 0);
		int n = global_index(cpi_ctx.nIntervals_dist.ptr,
							 cpi_ctx.nIntervals_dist.pitch, xid, yid,0)[0];
		float K = trapezoidal(H, cpi_ctx.dz, n, k_values);
		global_index(cpi_ctx.K.ptr, cpi_ctx.K.pitch, xid, yid,0)[0] = K;
    }
}

void callCoarsePermIntegrationKernel(dim3 grid, dim3 block, CoarsePermIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(cpi_ctx, args, sizeof(CoarsePermIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	CoarsePermIntegrationKernel<<<grid, block>>>();
}

__global__ void CoarseMobIntegrationKernel(int gridDimX){


	int xid = (cmi_ctx.active_block_indexes.ptr[blockIdx.x] % gridDimX)*blockDim.x + threadIdx.x;
    int yid = (cmi_ctx.active_block_indexes.ptr[blockIdx.x]/gridDimX)*blockDim.y + threadIdx.y;

    if ( xid < common_ctx.nx && yid < common_ctx.ny ){
		// Get full local height
		float H = global_index(common_ctx.H.ptr,
							   common_ctx.H.pitch, xid, yid, 0)[0];
		float h = global_index(cmi_ctx.h.ptr,
						       cmi_ctx.h.pitch, xid, yid, 0)[0];
		int nIntervalsForh = ceil(h/cmi_ctx.dz);

		float K = global_index(cmi_ctx.K.ptr, cmi_ctx.K.pitch, xid, yid, 0)[0];

		float* k_values = global_index(cpi_ctx.perm_distribution, xid, yid, 0, 0);

		float L = computeLambda(cmi_ctx.p_ci, common_ctx.g, common_ctx.delta_rho, h, cmi_ctx.dz, nIntervalsForh,
							    k_values, common_ctx.p_cap_ref_table.ptr, common_ctx.s_c_ref_table.ptr);
		if (K != 0){
			global_index(cmi_ctx.Lambda.ptr, cmi_ctx.Lambda.pitch, xid, yid, 0)[0] = L/K;
		}
	}
}

void callCoarseMobIntegrationKernel(dim3 grid, dim3 block, int gridDimX, CoarseMobIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(cmi_ctx, args, sizeof(CoarseMobIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	printf("gridDimX %i",gridDimX);
	CoarseMobIntegrationKernel<<<grid, block>>>(gridDimX);
}






#include "Kernels.h"

__constant__ FluxKernelArgs fk_ctx;
__constant__ CommonArgs common_ctx;
__constant__ TimeIntegrationKernelArgs tik_ctx;
__constant__ CoarsePermIntegrationKernelArgs cpi_ctx;
__constant__ CoarseMobIntegrationKernelArgs cmi_ctx;
__constant__ TimestepReductionKernelArgs trk_ctx;
__constant__ SolveForhProblemCellsKernelArgs spc_ctx;


void initAllocate(CommonArgs* args1, CoarsePermIntegrationKernelArgs* args2,
				  CoarseMobIntegrationKernelArgs* args3, FluxKernelArgs* args4,
				  TimeIntegrationKernelArgs* args5, TimestepReductionKernelArgs* args6,
				  SolveForhProblemCellsKernelArgs* args7){
	cudaHostAlloc(&args1, sizeof(CommonArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args2, sizeof(CoarsePermIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args3, sizeof(CoarseMobIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args4, sizeof(FluxKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args5, sizeof(TimeIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args6, sizeof(TimeIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args7, sizeof(SolveForhProblemCellsKernelArgs), cudaHostAllocWriteCombined);
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

inline __device__ int sign(float val) {
    return (0 < val) - (val < 0);
}

//NORMAL
__device__ float computeBrineSaturation(float p_cap, float C){
	return fmaxf(C*C/((C+p_cap)*(C+p_cap)), common_ctx.s_b_res);
}

//LEVRETT
__device__ float computeBrineSaturationLevrettJ(float p_cap, float C){
	if (p_cap >= 0)
		return fmaxf(1/((1+C*p_cap)*(1+C*p_cap)), common_ctx.s_b_res);
	else
		return 1;
}

__device__ float computeRelPermBrine(float s_e, float lambda_end_point_b){
		return (pow(s_e, 3)*lambda_end_point_b);
}
__device__ float computeRelPermCarbon(float s_e, float lambda_end_point_c){
		return (pow(1-s_e, 3)*lambda_end_point_c);
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

__device__ float simpson(float length, float dz, int n, float* function_values){
	if (n%2 != 0)
		n = n+1;
	float sum;
	dz = length/max(n,1);
	float function_value_0 = function_values[0];
	float function_value_1, function_value_2;
	for (int i = 1; i<=n; i+=2){
		function_value_1 = function_values[i];
		function_value_2 = function_values[i+1];
		sum += function_value_0 + 4*function_value_1 + function_value_0;
		function_value_0 = function_value_2;
	}
	return sum*(dz/(float)3.0);
}

inline __device__ float computeFluxEast(float (&U)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&lambda_c)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&lambda_b)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&dlambda_c)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&dlambda_b)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&h)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&z)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float(&normal_z)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float K_face, float g_vec, float (&pv)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   unsigned int i, unsigned int j){
	float face_mob_c, face_mob_b, dface_mob_c, dface_mob_b, tot_mob, F_c, z_diff;
	float U_b, U_c;
	U_c = 0;
	U_b = 0;
	float h_diff, g_flux, b;
	float delta_rho = common_ctx.delta_rho;
	float ff = 0;
	z_diff= z[i][j]-z[i+1][j];
	h_diff = h[i][j]*normal_z[i][j]-h[i+1][j]*normal_z[i+1][j];
	b = z_diff + h_diff;
	g_flux = -g_vec*b*delta_rho*K_face;
	bool aa = !(U[i][j]<0) & !(g_flux<0);
	bool bb = !(U[i][j]>0) & !(g_flux>0);
	face_mob_c = lambda_c[i][j];
	face_mob_b = lambda_b[i][j];

	if (aa | bb ) {
		if (!bb){
			face_mob_c = lambda_c[i][j];
			dface_mob_c = dlambda_c[i][j];
		} else {
			face_mob_c = lambda_c[i+1][j];
			dface_mob_c = dlambda_c[i+1][j];
		}
		if ((U[i][j] - face_mob_c*g_flux) >= 0){
			face_mob_b = lambda_b[i][j];
			dface_mob_b = dlambda_b[i][j];
		} else {
			face_mob_b = lambda_b[i+1][j];
			dface_mob_b = dlambda_b[i+1][j];
		}
	} else {
		bb = !(U[i][j]>0) & !(g_flux<0);
		if (!bb) {
			face_mob_b = lambda_b[i][j];
			dface_mob_b = dlambda_b[i][j];
		} else {
			face_mob_b = lambda_b[i+1][j];
			dface_mob_b = dlambda_b[i+1][j];
		}
		if ((U[i][j] + face_mob_b*g_flux) >= 0) {
			face_mob_c = lambda_c[i][j];
			dface_mob_c = dlambda_c[i][j];
		} else {
			face_mob_c = lambda_c[i+1][j];
			dface_mob_c = dlambda_c[i+1][j];
		}
	}

	tot_mob = face_mob_c + face_mob_b;
	F_c = 0;
	if (tot_mob != 0) {
		F_c = face_mob_c/tot_mob;
		U_c = F_c*(U[i][j]+face_mob_b*g_flux);
		U_b = (1-F_c)*(U[i][j]-face_mob_c*g_flux);
		ff = face_mob_b*dface_mob_c*abs(U_c)/(tot_mob*face_mob_c)+
			 face_mob_c*dface_mob_b*abs(U_b)/(tot_mob*face_mob_b)-
		     (g_vec*delta_rho*K_face*face_mob_b*F_c);
	}
	float dt_temp = FLT_MAX;
	if (pv[i][j] != 0)
		dt_temp = pv[i][j]/ff;
	if (pv[i+1][j] != 0)
		dt_temp = fminf(pv[i+1][j]/ff, dt_temp);
	U[i][j] =  U_c;
	return dt_temp;
}

inline __device__ float computeFluxNorth(float (&U)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&lambda_c)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&lambda_b)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&dlambda_c)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&dlambda_b)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&h)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&z)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&normal_z)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float K_face, float g_vec,  float (&pv)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   unsigned int i, unsigned int j){
	float face_mob_c, face_mob_b, dface_mob_c, dface_mob_b, tot_mob, F_c, z_diff;
	float U_b, U_c;
	U_c = 0;
	U_b = 0;
	float h_diff, b, g_flux;
	float delta_rho = common_ctx.delta_rho;
	float ff = 0;
	h_diff = h[i][j]*normal_z[i][j]-h[i][j+1]*normal_z[i][j+1];
	z_diff= z[i][j]-z[i][j+1];
	b = z_diff + h_diff;
	g_flux = -g_vec*b*delta_rho*K_face;
	bool aa = !(U[i][j]<0) & !(g_flux<0);
	bool bb = !(U[i][j]>0) & !(g_flux>0);
	face_mob_c = lambda_c[i][j];
	face_mob_b = lambda_b[i][j];

	// Determine the upwind cell evaluation for the two phases
	if (aa | bb ) {
		if (!bb){
			face_mob_c = lambda_c[i][j];
			dface_mob_c = dlambda_c[i][j];
			//
		} else {
			face_mob_c = lambda_c[i][j+1];
			dface_mob_c = dlambda_c[i][j+1];
		}
		if ((U[i][j] - face_mob_c*g_flux) >= 0){
			face_mob_b = lambda_b[i][j];
			dface_mob_b = dlambda_b[i][j];
		} else {
			face_mob_b = lambda_b[i][j+1];
			dface_mob_b = dlambda_b[i][j+1];
		}
	} else {
		bb = !(U[i][j]>0) & !(g_flux<0);
		if (!bb) {
			face_mob_b = lambda_b[i][j];
			dface_mob_b = dlambda_b[i][j];
		} else {
			face_mob_b = lambda_b[i][j+1];
			dface_mob_b = dlambda_b[i][j+1];
		}
		if ((U[i][j] + face_mob_b*g_flux) >= 0) {
			face_mob_c = lambda_c[i][j];
			dface_mob_c = dlambda_c[i][j];
		} else {
			face_mob_c = lambda_c[i][j+1];
			dface_mob_c = dlambda_c[i][j+1];
		}
	}
	tot_mob = face_mob_c + face_mob_b;
	if (tot_mob != 0) {
		F_c = face_mob_c/tot_mob;
		U_c = F_c*(U[i][j]+face_mob_b*g_flux);
		U_b = (1-F_c)*(U[i][j]-face_mob_c*g_flux);
		ff = face_mob_b*dface_mob_c*abs(U_c)/(tot_mob*face_mob_c)+
			 face_mob_c*dface_mob_b*abs(U_b)/(tot_mob*face_mob_b)-
		     (g_vec*delta_rho*K_face*face_mob_b*F_c);
	}
	float dt_temp = FLT_MAX;
	if (pv[i][j] != 0)
		dt_temp = pv[i][j]/ff;
	if (pv[i][j+1] != 0)
		dt_temp = fminf(pv[i][j+1]/ff, dt_temp);
	U[i][j] = U_c;
	return dt_temp;
}

__global__ void FluxKernel(int gridDimX){

	int border = common_ctx.border;
	int noborder = 0;

	// Global id with grid mask
	int xid = (fk_ctx.active_block_indexes.ptr[blockIdx.x] % gridDimX)*TILEDIM_X + threadIdx.x-border;
	int yid = (fk_ctx.active_block_indexes.ptr[blockIdx.x]/gridDimX)*TILEDIM_Y + threadIdx.y - border;

	// Global id without gridmask
	//int xid = blockIdx.x*TILEDIM_X + threadIdx.x-border;
    //int yid = blockIdx.y*TILEDIM_Y + threadIdx.y-border;

    xid = fminf(xid, common_ctx.nx);
    yid = fminf(yid, common_ctx.ny);
    int new_xid = fminf(xid, common_ctx.nx-1);
    int new_yid = fminf(yid, common_ctx.ny-1);
    new_xid = fmaxf(new_xid, 0);
    new_yid = fmaxf(new_yid, 0);

    // Local id
    int i = threadIdx.x;
    int j = threadIdx.y;

    int active_north = global_index(common_ctx.active_north.ptr, common_ctx.active_north.pitch,
    		                        new_xid, new_yid, noborder)[0];
    int active_east = global_index(common_ctx.active_east.ptr, common_ctx.active_east.pitch,
    		                        new_xid, new_yid, noborder)[0];

    __shared__ float U_local_x[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float U_local_y[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float lambda_c_local[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float lambda_b_local[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float dlambda_c_local[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float dlambda_b_local[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float h_local[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float z_local[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float normal_z_local[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];
    __shared__ float pv_local[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y];

    //TIMESTEP BLOCK
	const int nthreads = BLOCKDIM_X_FLUX*BLOCKDIM_Y_FLUX;
	__shared__ float timeStep[BLOCKDIM_X_FLUX][BLOCKDIM_Y_FLUX];
	float default_ = FLT_MAX;
	float dt_local = default_;
	timeStep[i][j] = default_;

	// Shared memory blocks required for the upstream mobility flux computation
    U_local_x[i][j] = global_index(fk_ctx.U_x.ptr, fk_ctx.U_x.pitch, xid, yid, border)[0];
    U_local_y[i][j] = global_index(fk_ctx.U_y.ptr, fk_ctx.U_y.pitch, xid, yid, border)[0];
    float lambda_c = global_index(fk_ctx.Lambda_c.ptr, fk_ctx.Lambda_c.pitch, new_xid, new_yid, noborder)[0];
    lambda_c_local[i][j] = lambda_c;
    float lambda_b = global_index(fk_ctx.Lambda_b.ptr, fk_ctx.Lambda_b.pitch, new_xid, new_yid, noborder)[0];
    lambda_b_local[i][j] = lambda_b;
    dlambda_c_local[i][j] = global_index(fk_ctx.dLambda_c.ptr, fk_ctx.dLambda_c.pitch, new_xid, new_yid, noborder)[0];
    dlambda_b_local[i][j] = global_index(fk_ctx.dLambda_b.ptr, fk_ctx.dLambda_b.pitch, new_xid, new_yid, noborder)[0];
    h_local[i][j] = global_index(fk_ctx.h.ptr, fk_ctx.h.pitch, new_xid, new_yid, noborder)[0];
    z_local[i][j] = global_index(fk_ctx.z.ptr, fk_ctx.z.pitch, new_xid, new_yid, noborder)[0];
    normal_z_local[i][j] = global_index(fk_ctx.normal_z.ptr, fk_ctx.normal_z.pitch, new_xid, new_yid, noborder)[0];
    float g_vec_east = global_index(fk_ctx.g_vec_east.ptr, fk_ctx.g_vec_east.pitch, new_xid, new_yid, noborder)[0];
    float g_vec_north = global_index(fk_ctx.g_vec_north.ptr, fk_ctx.g_vec_north.pitch, new_xid, new_yid, noborder)[0];
    float K_face_east  = global_index(fk_ctx.K_face_east.ptr, fk_ctx.K_face_east.pitch, new_xid, new_yid, noborder)[0];
    float K_face_north = global_index(fk_ctx.K_face_north.ptr, fk_ctx.K_face_north.pitch, new_xid, new_yid, noborder)[0];
    pv_local[i][j] = global_index(common_ctx.pv.ptr, common_ctx.pv.pitch, new_xid, new_yid, noborder)[0];
    float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, new_xid, new_yid, noborder)[0];

    __syncthreads();

	if (global_index(common_ctx.H.ptr, common_ctx.H.pitch, new_xid, new_yid, noborder)[0] == 0) {
		dt_local = default_;
		U_local_x[i][j] = 0;
		U_local_y[i][j] = 0;
		return;
	}

    if (i < (TILEDIM_X+1) && j < (TILEDIM_Y+1)) {
    	dt_local = computeFluxEast(U_local_x, lambda_c_local, lambda_b_local,
    			dlambda_c_local, dlambda_b_local, h_local, z_local, normal_z_local, K_face_east, g_vec_east, pv_local, i, j);
    	if (xid == 42 && yid <= 43 ){
    		U_local_x[i][j] = 0;
    		dt_local = default_;
    	}

    	dt_local = fminf(dt_local, computeFluxNorth(U_local_y, lambda_c_local, lambda_b_local,
    			dlambda_c_local, dlambda_b_local, h_local, z_local, normal_z_local, K_face_north ,g_vec_north, pv_local, i, j));

    	// NO-FLOW BOUNDARY
    	if (xid == 0 || yid == 0 || xid == common_ctx.nx || yid == common_ctx.ny) {
    		dt_local = default_;
    		U_local_x[i][j] = 0;
    		U_local_y[i][j] = 0;
    	}

    	if (active_east == -1){
    		U_local_x[i][j] = 0;
    		//dt_local = default_;
    	}
    	if (active_north == -1){
    		U_local_y[i][j] = 0;
    	    //dt_local = default_;
    	}
    }

    // Local time step reduction
    int p = threadIdx.y*blockDim.x+threadIdx.x;

    __syncthreads();

    if (xid > -1 && xid < common_ctx.nx && yid > -1 && yid < common_ctx.ny){
        if (i < TILEDIM_X+1 && i > 0 && j < TILEDIM_Y+1 && j > 0) {
				float r = (U_local_x[i][j] - U_local_x[i-1][j]) + (U_local_y[i][j] - U_local_y[i][j-1]);
				float source = global_index(fk_ctx.source.ptr, fk_ctx.source.pitch , new_xid, new_yid, noborder)[0];
				float F_c = lambda_c/(lambda_c+lambda_b);
				r = r -fmaxf(source, 0) - fminf(source, 0)*F_c;
				global_index(fk_ctx.R.ptr, fk_ctx.R.pitch, new_xid, new_yid, noborder)[0] = r;
				timeStep[i][j] = dt_local;
        }
    }
    __syncthreads();
    		volatile float* B_volatile = timeStep[0];
    		__syncthreads();

    		if (nthreads >= 512) {
    			if (p < 512 && (p+512) < nthreads) timeStep[0][p] = fminf(timeStep[0][p], timeStep[0][p + 512]); //min(1024, nthreads)=>512
    			__syncthreads();
    		}

    		if (nthreads >= 256) {
    			if (p < 256 && (p+256) < nthreads) timeStep[0][p] = fminf(timeStep[0][p], timeStep[0][p + 256]); //min(512, nthreads)=>256
    			__syncthreads();
    		}
    		if (nthreads >= 128) {
    			if (p < 128 && (p+128) < nthreads) timeStep[0][p] = fminf(timeStep[0][p], timeStep[0][p + 128]); //min(256, nthreads)=>128
    			__syncthreads();
    		}
    		if (nthreads >= 64) {
    			if (p < 64 && (p+64) < nthreads) timeStep[0][p] = fminf(timeStep[0][p], timeStep[0][p + 64]); //min(128, nthreads)=>64
    			__syncthreads();
    		}

    		//Will generate out-of-bounds errors for nthreads < 64
    		if (p < 32) {
    			if (nthreads >= 64) B_volatile[p] = fminf(B_volatile[p], B_volatile[p + 32]); //64=>32
    			if (nthreads >= 32) B_volatile[p] = fminf(B_volatile[p], B_volatile[p + 16]); //32=>16
    			if (nthreads >= 16) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  8]); //16=>8
    			if (nthreads >=  8) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  4]); //8=>4
    			if (nthreads >=  4) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  2]); //4=>2
    			if (nthreads >=  2) B_volatile[p] = fminf(B_volatile[p], B_volatile[p +  1]); //2=>1
    		}

    		if (threadIdx.y + threadIdx.x == 0) fk_ctx.dt_vector[blockIdx.x*gridDim.y + blockIdx.y] = B_volatile[0];

}

void callFluxKernel(dim3 grid, dim3 block, int gridDimX, FluxKernelArgs* args){
	cudaMemcpyToSymbolAsync(fk_ctx, args, sizeof(FluxKernelArgs), 0, cudaMemcpyHostToDevice);
	FluxKernel<<<grid, block>>>(gridDimX);
}

inline __device__ float computeSatu(float z, float C){
	float curr_p_cap = common_ctx.p_ci + common_ctx.g*common_ctx.delta_rho*(-z);
	return 1.0-computeBrineSaturation(curr_p_cap, C);
}

// Function to compute the last interval of the "Brute Force" algorithm
inline __device__ void higherResolutionIteration(float& z, float& sum, float C, float& prev_s_c, float& curr_s_c, float& dz, float S_c_new){
	z -= dz;
	sum -= 0.5*dz*(curr_s_c + prev_s_c);
	curr_s_c = prev_s_c;
	dz = dz/10;
	while (sum < S_c_new){
		prev_s_c = curr_s_c;
		z += dz;
		curr_s_c = computeSatu(z, C);
		sum += 0.5*dz*(curr_s_c + prev_s_c);
	}
}

// The shift required for the "Brute Force" algorithm
inline __device__ void higherResolutionIterationWithShift(float H, float& z, float& z_bottom, float& sum, float C, float& prev_s_c, float& curr_s_c,
														   float& prev_s_c_bottom, float& curr_s_c_bottom, float& dz, float S_c_new, bool& firstIter){
	float diff = 1;
	if (firstIter){
		sum -= 0.5*(z-H)*(curr_s_c + prev_s_c);
		z -= (z-H);
	} else{
		if (z_bottom > 0){
			z_bottom -= dz;
			sum += 0.5*dz*(curr_s_c_bottom + prev_s_c_bottom);
			curr_s_c_bottom = prev_s_c_bottom;
		}
		z -= dz;
		sum -= 0.5*dz*(curr_s_c + prev_s_c);
		curr_s_c = prev_s_c;
		dz = dz/10;
	}
	int i = 0;
	while (sum < S_c_new  && diff > 0 ){
		prev_s_c = curr_s_c;
		prev_s_c_bottom = curr_s_c_bottom;
		z += dz;
		curr_s_c = computeSatu(z, C);
		sum += 0.5*dz*(curr_s_c + prev_s_c);
		if (z > H){
			z_bottom += dz;
			curr_s_c_bottom = computeSatu(z_bottom, C);
			sum -= 0.5*dz*(curr_s_c_bottom + prev_s_c_bottom);
			diff = 0.5*dz*(curr_s_c + prev_s_c) - 0.5*dz*(curr_s_c_bottom + prev_s_c_bottom);
		}
		i++;
	}
	firstIter = false;
}

// Old version of F, before adding Simpson's rule, here we apply the trapezoidal rule
inline __device__ float F_wihout_opt(float S_c_new, float H, float h, float p_ci, float dz, float delta_rho, float g, float C){
	float height = fminf(H,h);
	int n = ceil(height/dz);
	float curr_p_cap = p_ci + g*(delta_rho)*(dz*0-h);
	float curr_satu_c = 1-computeBrineSaturation(curr_p_cap, C);
	float prev_satu_c = curr_satu_c;
	float sum_c = 0;
	if (n>0){
		for (int i = 1; i < n; i++){
			curr_p_cap = p_ci + g*(delta_rho)*(dz*i-h);
			curr_satu_c = 1-computeBrineSaturation(curr_p_cap, C);
			sum_c += dz*0.5*(curr_satu_c+prev_satu_c);
			prev_satu_c = curr_satu_c;
		}
			curr_p_cap = p_ci + g*(delta_rho)*(height-h);
			curr_satu_c = 1-computeBrineSaturation(curr_p_cap, C);
			sum_c += 0.5*(prev_satu_c+curr_satu_c)*(height-dz*(n-1));
	}
	return (S_c_new - sum_c);

}

// Add/reduce the top of the integral in Newton's method
inline __device__ float add_to_sum(float sum_c, float old_value, float diff, float dz, float C){
	int old_n = ceil(abs(diff)/dz);
	int n = min(old_n, 1000);
	// Even number of intervals required for  Simpson method
	if (n%2 != 0)
		n = n+1;
	//Interval length
	dz = diff/max(n,1);
	// If n is very large, then there is some error
	if (n > 600)
		printf("ERROR NEWTON: add to sum function %i C %.3f h_old %.3f h_new %.3f \n", old_n, C, old_value, diff);
	// Trapezoidal rule
	float curr_satu_c_1, curr_satu_c_2;
	float curr_satu_c_0 = computeSatu(old_value, C);
	float new_sum_c = 0;
	for (int i=1; i <= n; i+=2){
		curr_satu_c_1 = computeSatu(old_value + dz*i, C);
		curr_satu_c_2 = computeSatu(old_value + dz*(i+1), C);
		new_sum_c += (curr_satu_c_0 + 4*curr_satu_c_1 + curr_satu_c_2);
		curr_satu_c_0 = curr_satu_c_2;
	}
	sum_c += (dz/3)*new_sum_c;
	return sum_c;
}

// Add/reduce the bottom of the integral in Newton's method by Simpson's rule
inline __device__ float add_to_bottom(float sum_c, float old_value, float diff, float dz, float C){
	int old_n = ceil(abs(diff)/dz);
	int n = min(old_n, 600);
	// Even number of intervals required for  Simpson method
	if (n%2 != 0)
		n = n+1;
	// If n is very large, then there is some error
	if (n>600)
		printf("ERROR NEWTON: add to sum function %i C %.3f h_old %.3f h_new %.3f \n", old_n, C, old_value, diff);
	//Interval length
	dz = diff/max(n,1);
	// Trapezoidal rule
	float curr_satu_c_1, curr_satu_c_2;
	float curr_satu_c_0 = computeSatu(old_value, C);
	float new_sum_c = 0;
	for (int i=1; i <= n; i+=2){
		curr_satu_c_1 = computeSatu(old_value + dz*i, C);
		curr_satu_c_2 = computeSatu(old_value + dz*(i+1), C);
		new_sum_c += (curr_satu_c_0 + 4*curr_satu_c_1 + curr_satu_c_2);
		curr_satu_c_0 = curr_satu_c_2;
	}
	sum_c -= (dz/3)*new_sum_c;
	return sum_c;
}

//The functions above written in parallel with a shared memory reduction
inline __device__ float add_to_sum_parallel(float sum_c, float old_value, float diff, float dz, float C,
											int local_tid, float (&shared_mem)[N_CELLS_PER_BLOCK][64],
											int cell_block_id){
	int local_local_tid = local_tid - 32;
	int sign_diff = sign(diff);
	int n = ceil(abs(diff)/dz);
	dz = sign_diff*dz;
	int local_n = (n + 32 - 1)/32;
	float local_start, local_end;
	if (sign_diff > 0){
		local_start = fminf(local_local_tid*local_n*dz, diff);
		local_end = fminf(local_start + (local_n)*dz, diff);
	}
	else {
		local_start = fmaxf(local_local_tid*local_n*dz, diff);
		local_end = fmaxf(local_start + (local_n)*dz, diff);
	}
	float integration_length = local_end-local_start;
	if (local_n%2 != 0)
		local_n = local_n+1;
	dz = integration_length/max(local_n,1);
	float local_sum_c = 0;
	float curr_satu_c_1, curr_satu_c_2;
	float curr_satu_c_0 = computeSatu(old_value+local_start, C);
	for (int i=1; i <= local_n; i+=2){
		curr_satu_c_1 = computeSatu(old_value + local_start + dz*i, C);
		curr_satu_c_2 = computeSatu(old_value + local_start + dz*(i+1), C);
		local_sum_c += (curr_satu_c_0 + float(4)*curr_satu_c_1 + curr_satu_c_2);
		curr_satu_c_0 = curr_satu_c_2;
	}
	shared_mem[cell_block_id][local_tid] = local_sum_c*(dz/float(3));
	if (local_tid < 48){
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 16];
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 8];
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 4];
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 2];
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 1];
	}
	return (shared_mem[cell_block_id][32]);
}
inline __device__ float add_to_bottom_parallel(float sum_c, float old_value, float diff, float dz, float C,
											int local_tid, float (&shared_mem)[N_CELLS_PER_BLOCK][64],
											int cell_block_id){
	int sign_diff = sign(diff);
	int n = ceil(abs(diff)/dz);
	dz = sign_diff*dz;
	int local_n = (n + 32 - 1)/32;
	float local_start, local_end;
	if (sign_diff > 0){
		local_start = fminf(local_tid*local_n*dz, diff);
		local_end = fminf(local_start + (local_n)*dz, diff);
	}
	else {
		local_start = fmaxf(local_tid*local_n*dz, diff);
		local_end = fmaxf(local_start + (local_n)*dz, diff);
	}
	float integration_length = local_end-local_start;

	if (local_n%2 != 0)
		local_n = local_n+1;
	dz = integration_length/max(local_n,1);

	float local_sum_c = 0;
	float curr_satu_c_1, curr_satu_c_2;
	float curr_satu_c_0 = computeSatu(old_value+local_start, C);
	for (int i=1; i <= local_n; i+=2){
		curr_satu_c_1 = computeSatu(old_value + local_start + dz*i, C);
		curr_satu_c_2 = computeSatu(old_value + local_start + dz*(i+1), C);
		local_sum_c += (curr_satu_c_0 + float(4)*curr_satu_c_1 + curr_satu_c_2);
		curr_satu_c_0 = curr_satu_c_2;
	}
	shared_mem[cell_block_id][local_tid] = -(local_sum_c)*(dz/(float)3);
	if (local_tid < 16){
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 16];
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 8];
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 4];
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 2];
		shared_mem[cell_block_id][local_tid] += shared_mem[cell_block_id][local_tid + 1];
	}
	return (shared_mem[cell_block_id][0]);
}

inline __device__ float F(float S_c_new, float H, float h, float dz, float C,
						  float &cut_off_old, float &h_old, float &sum){
	float cut_off = fmaxf(0,h-H);
	float cut_diff = cut_off-cut_off_old;
	float h_diff = h-h_old;
	sum = add_to_bottom(sum, cut_off_old, cut_diff, dz, C);
	sum = add_to_sum(sum, h_old, h_diff, dz, C);
	h_old = h;
	cut_off_old = cut_off;
	return (S_c_new - sum);
}

inline __device__ float F_parallel_new(float S_c_new, float H, float h, float dz, float C,
		  	  	  	  	               float &cut_off_old, float &h_old, float &sum,
		  	  	  	  	               float (&shared_mem)[N_CELLS_PER_BLOCK][64], int local_tid, int cell_block_id){
	float cut_off = fmaxf(0,h-H);
	float cut_diff = cut_off-cut_off_old;
	float h_diff = h-h_old;

	if (local_tid < 32)
		add_to_bottom_parallel(sum, cut_off_old, cut_diff, dz, C, local_tid, shared_mem, cell_block_id);
	else
		add_to_sum_parallel(sum, h_old, h_diff, dz, C, local_tid, shared_mem, cell_block_id);
	//if (local_tid == 0){
	//	printf("S_c_new %.5f h_old %.5f h %.5f sum_c %.5f H %.3f F(h) %.3f \n", S_c_new, h_old, h, sum,H, S_c_new -sum);
	//}
	/*if (local_tid == 0){
		for (int i = 0; i <32; i++)
			printf("%.3f  ", shared_mem[cell_block_id][i]);
	}
	*/
	__syncthreads();
	sum = sum + shared_mem[cell_block_id][0] + shared_mem[cell_block_id][32];
	h_old = h;
	cut_off_old = cut_off;
	return (S_c_new - sum);

}

inline __device__ float solveForhNewton(float S_c_new, float S_c_old, float H, float previous_h, float p_ci, float dz,
		                                float delta_rho, float g, float C, int &iter, int max_iter){
	float h_new = fmaxf(0.01,previous_h);
	h_new = fmaxf(h_new, S_c_new);
	float h_old = 0;
	float cut_off_old = 0;
	float sum_c = 0;
	float e = 1000;
	float init_tol = 0.000001;
	float TOL = init_tol;
	float F_h_old;
	float F_deriv_h_old;
	float eps = TOL/100;
	if (abs(S_c_new) < eps ){
		return 0;
	}
	while (e > TOL && iter < max_iter){
		if (h_new <= H) {
			F_deriv_h_old = -computeSatu(h_new, C);
		} else {
			F_deriv_h_old = -computeSatu(h_new, C) + computeSatu(h_new-H,C);
		}
		if (F_deriv_h_old == 0 || h_new < 0){
			iter = max_iter;
			return 0;
		}
		F_h_old = F(S_c_new, H, h_new, dz, C, cut_off_old, h_old, sum_c);
		h_new = h_old - F_h_old/F_deriv_h_old;
		e = abs(h_new - h_old);
		iter++;
		TOL = fmaxf(h_new/pow(10.0,7), init_tol);
	}
	if (e < TOL)
		iter = min(max_iter-1, iter);
	return h_new;
}

inline __device__ float solveForh(float S_c_new, float H, float p_ci, float dz, float delta_rho, float g, float C){
	float eps = dz/(10*10*10*10*10*10);
	float sum = 0;
	float z = 0;
	float curr_s_c = computeSatu(z, C);
	float prev_s_c = 0;
	float z_bottom = 0;
	float curr_s_c_bottom = computeSatu(z, C);
	float prev_s_c_bottom = 0;
	//int i = 0;
	if (H > eps && eps < S_c_new ){
		while (sum < S_c_new && z < H + dz){
			prev_s_c = curr_s_c;
			z += dz;
			curr_s_c = computeSatu(z, C);
			sum += 0.5*dz*(curr_s_c + prev_s_c);
			//i++;
		}
		int nIter = 0;
		while (z < H && nIter < 5) {
			higherResolutionIteration(z, sum, C, prev_s_c, curr_s_c, dz, S_c_new);
			nIter ++;
		}
		bool firstIter = true;
		while (nIter < 5){
			higherResolutionIterationWithShift(H, z, z_bottom, sum, C, prev_s_c, curr_s_c,
									          prev_s_c_bottom, curr_s_c_bottom, dz, S_c_new, firstIter);
			nIter++;
		}
	}
		return fmaxf(0,z);
}

__global__ void TimeIntegrationKernel(int gridDimX){

	// Global id
	int noborder = 0; //common_ctx.border;
	int dt = tik_ctx.global_dt[0];
	int xid = (cmi_ctx.active_block_indexes.ptr[blockIdx.x] % gridDimX)*blockDim.x + threadIdx.x;
    int yid = (cmi_ctx.active_block_indexes.ptr[blockIdx.x]/gridDimX)*blockDim.y + threadIdx.y;
    xid = fminf(xid, common_ctx.nx-1);
    yid = fminf(yid, common_ctx.ny-1);

    float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, xid, yid, noborder)[0];
    float pv = global_index(tik_ctx.pv.ptr, tik_ctx.pv.pitch, xid, yid, noborder)[0];
    float vol_old, vol_new;
    float S_c_new = 0;
    float S_c_old = global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0];
    vol_old = global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0]*pv;
    float r = global_index(tik_ctx.R.ptr, tik_ctx.R.pitch, xid, yid, noborder)[0];
    vol_new = vol_old - dt*r;
	//if (abs(r) > 0)
		//printf("S_c_new %.3f, vol_new %.3f r_new %.3f xid: %i yid: %i pv %f \n", S_c_new/H, vol_new, r, xid, yid, pv);

    if (pv != 0){
		S_c_new = vol_new/pv;
		/*
		float eps = 0.000001;
		if (S_c_new/H >= 0.9-eps){
			S_c_new = H*0.9;
			vol_new = pv*S_c_new;
		}
		*/
		global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0] = S_c_new;
		float C = global_index(tik_ctx.scaling_parameter_C.ptr, tik_ctx.scaling_parameter_C.pitch,
							   xid, yid, 0)[0];
		global_index(tik_ctx.h.ptr, tik_ctx.h.pitch, xid, yid, 0)[0] = 0;

		// Brute force algorithm
		float h = solveForh(S_c_new, H,common_ctx.p_ci, tik_ctx.dz, common_ctx.delta_rho, common_ctx.g, C);
		global_index(tik_ctx.h.ptr, tik_ctx.h.pitch, xid, yid, noborder)[0] = h;


		global_index(tik_ctx.vol_new.ptr, tik_ctx.vol_new.pitch, xid, yid, noborder)[0] = vol_new;
    }
}

__global__ void TimeIntegrationKernelNewton(int gridDimX){

	// Global id
	int noborder = 0; //common_ctx.border;
	int dt = tik_ctx.global_dt[0];
	int xid = (cmi_ctx.active_block_indexes.ptr[blockIdx.x] % gridDimX)*blockDim.x + threadIdx.x;
    int yid = (cmi_ctx.active_block_indexes.ptr[blockIdx.x]/gridDimX)*blockDim.y + threadIdx.y;
    xid = fminf(xid, common_ctx.nx-1);
    yid = fminf(yid, common_ctx.ny-1);

    float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, xid, yid, noborder)[0];
    float pv = global_index(tik_ctx.pv.ptr, tik_ctx.pv.pitch, xid, yid, noborder)[0];
    float vol_old, vol_new;
    float S_c_new = 0;
    float S_c_old = global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0];
    vol_old = global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0]*pv;

    float r = global_index(tik_ctx.R.ptr, tik_ctx.R.pitch, xid, yid, noborder)[0];
    vol_new = vol_old - dt*r;
    if (pv != 0){
		S_c_new = vol_new/pv;

		global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0] = S_c_new;
		float C = global_index(tik_ctx.scaling_parameter_C.ptr, tik_ctx.scaling_parameter_C.pitch,
							   xid, yid, 0)[0];
		float prev_h = global_index(tik_ctx.h.ptr, tik_ctx.h.pitch, xid, yid, noborder)[0];
		global_index(tik_ctx.h.ptr, tik_ctx.h.pitch, xid, yid, 0)[0] = 0;

		int iter= 0;
		int iter_lim = 7;

		float h = solveForhNewton(S_c_new, S_c_old, H, prev_h, common_ctx.p_ci, 5*tik_ctx.dz, common_ctx.delta_rho, common_ctx.g, C, iter, iter_lim);

		if (iter >= iter_lim){
			tik_ctx.d_isValid[common_ctx.nx*yid +xid] = 1;
		} else {
			tik_ctx.d_isValid[common_ctx.nx*yid +xid] = 0;
			global_index(tik_ctx.h.ptr, tik_ctx.h.pitch, xid, yid, noborder)[0] = h;
		}
		global_index(tik_ctx.vol_new.ptr, tik_ctx.vol_new.pitch, xid, yid, noborder)[0] = vol_new;
    }
}

__global__ void solveForhProblemCellsBruteForce(){
	size_t threadid = blockIdx.x*blockDim.x + threadIdx.x;
	if (threadid > spc_ctx.d_numValid[0])
		return;
	int xid = spc_ctx.d_out[threadid] % common_ctx.nx;
	int yid = spc_ctx.d_out[threadid] / (int) common_ctx.nx;
	int noborder = 0;

    float S_c_new = global_index(spc_ctx.S_c.ptr, spc_ctx.S_c.pitch, xid, yid, noborder)[0];
	float C = global_index(spc_ctx.scaling_parameter_C.ptr, spc_ctx.scaling_parameter_C.pitch,
						   xid, yid, 0)[0];
    float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, xid, yid, noborder)[0];

    float h = solveForh(S_c_new, H, common_ctx.p_ci, spc_ctx.dz, common_ctx.delta_rho, common_ctx.g, C);
    global_index(spc_ctx.h.ptr, spc_ctx.h.pitch, xid, yid, noborder)[0] = h;

}
__global__ void solveForhProblemCellsBisectionNew(){
	int threads_per_cell = 64;
	size_t thread_id = blockIdx.x*blockDim.x + threadIdx.x;
	size_t cell_id = thread_id/threads_per_cell;
	if (cell_id > spc_ctx.d_numValid[0])
		return;
	int cell_block_id = cell_id % N_CELLS_PER_BLOCK;
	int local_tid = thread_id % threads_per_cell;
	int xid = spc_ctx.d_out[cell_id] % common_ctx.nx;
	int yid = spc_ctx.d_out[cell_id] / (int) common_ctx.nx;
	int noborder = 0;

    float S_c_new = global_index(spc_ctx.S_c.ptr, spc_ctx.S_c.pitch, xid, yid, noborder)[0];
	float C = global_index(spc_ctx.scaling_parameter_C.ptr, spc_ctx.scaling_parameter_C.pitch,
						   xid, yid, 0)[0];
    float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, xid, yid, noborder)[0];
    float prev_h = global_index(spc_ctx.h.ptr, spc_ctx.h.pitch, xid, yid, noborder)[0];

	if (S_c_new == 0){
		global_index(spc_ctx.h.ptr, spc_ctx.h.pitch, xid, yid, noborder)[0] = 0;
		return;
	}
    __shared__ float shared_mem[N_CELLS_PER_BLOCK][64];

    // BISECTION METHOD
	float init_tol = 0.000001;
	float TOL = init_tol;
	float l_cap = -((C/sqrt(common_ctx.s_b_res))-C)/(common_ctx.delta_rho*common_ctx.g);
	float a = 0;
	float b = H + l_cap;
	int iter = 0;
	int max_iter = 50;
	float F_mid =100;
	float midpoint;
	float cut_off_old = 0;
	float sum = 0;
	float h_old = 0;
	if (abs(S_c_new) < init_tol ){
		 global_index(spc_ctx.h.ptr, spc_ctx.h.pitch, xid, yid, noborder)[0] = 0;
		 return;
	}
	float F_a = F_parallel_new(S_c_new, H, a, spc_ctx.dz*5, C, cut_off_old,
		       h_old, sum, shared_mem, local_tid, cell_block_id);
	while ( ((b-a)*0.5) > TOL && iter < max_iter ){
		midpoint = (a+b)*0.5f;
		F_mid = F_parallel_new(S_c_new, H, midpoint, spc_ctx.dz*5, C, cut_off_old,
			       h_old, sum, shared_mem, local_tid, cell_block_id);
		if (sign(F_mid) == sign(F_a)){
			a = midpoint;
			F_a = F_mid;
		}  else {
			b = midpoint;
		}
		iter++;
		TOL = fmaxf(midpoint/pow(10.0,7), init_tol);
	}
	/*if (iter > 10 && local_tid == 0){
		printf("iterations %i S_c %.10f h %.10f H %.5f C: %.5f sum: %.10f, xid %i yid %i diff %.10f prev_h %.10f\n", iter, S_c_new, midpoint, H, C, sum, xid, yid, abs(midpoint-abs(prev_h)), prev_h);
	}
	*/
    global_index(spc_ctx.h.ptr, spc_ctx.h.pitch, xid, yid, noborder)[0] = midpoint;
}




void callSolveForhProblemCellsBisection(dim3 grid, dim3 block, SolveForhProblemCellsKernelArgs* args){
	cudaMemcpyToSymbolAsync(spc_ctx, args, sizeof(SolveForhProblemCellsKernelArgs), 0, cudaMemcpyHostToDevice);
	solveForhProblemCellsBisectionNew<<<grid, block>>>();
}

void callSolveForhProblemCells(dim3 grid, dim3 block, SolveForhProblemCellsKernelArgs* args){
	cudaMemcpyToSymbolAsync(spc_ctx, args, sizeof(SolveForhProblemCellsKernelArgs), 0, cudaMemcpyHostToDevice);
	solveForhProblemCellsBruteForce<<<grid, block>>>();
}

void callTimeIntegration(dim3 grid, dim3 block, int gridDimX, TimeIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(tik_ctx, args, sizeof(TimeIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	TimeIntegrationKernel<<<grid, block>>>(gridDimX);
}

void callTimeIntegrationNewton(dim3 grid, dim3 block, int gridDimX, TimeIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(tik_ctx, args, sizeof(TimeIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	TimeIntegrationKernelNewton<<<grid, block>>>(gridDimX);
}


// Function to compute the capillary pressure in the subintervals
__device__ void computeLambda(float* lambda_c_and_b, float p_ci, float g, float delta_rho, float H, float height, float h,
							   float dz, int n, float* k_values, float scaling_parameter_C){

	float curr_p_cap = p_ci + g*(delta_rho)*(dz*0-h);
	float curr_satu_b = computeBrineSaturation(curr_p_cap, scaling_parameter_C);
	float curr_satu_e = (curr_satu_b-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
	float curr_mob_c = computeRelPermCarbon(curr_satu_e, common_ctx.lambda_end_point_c);
	float curr_mob_b = computeRelPermBrine(curr_satu_e, common_ctx.lambda_end_point_b);
	float curr_mobk_c = curr_mob_c*k_values[0];
	float curr_mobk_b = curr_mob_b*k_values[0];
	float prev_mobk_c = curr_mobk_c;
	float prev_mobk_b = curr_mobk_b;
	float sum_c = 0;
	float sum_b = 0;
	if (n>0){
		for (int i = 1; i < n; i++){
			curr_p_cap = p_ci + g*(delta_rho)*(dz*i-h);
			curr_satu_b = computeBrineSaturation(curr_p_cap, scaling_parameter_C);

			curr_satu_e = (curr_satu_b-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
			curr_mob_c = computeRelPermCarbon(curr_satu_e, common_ctx.lambda_end_point_c);
			curr_mob_b = computeRelPermBrine(curr_satu_e, common_ctx.lambda_end_point_b);

			curr_mobk_c = curr_mob_c*k_values[i];
			curr_mobk_b = curr_mob_b*k_values[i];
			sum_c += dz*0.5*(curr_mobk_c+prev_mobk_c);
			sum_b += dz*0.5*(curr_mobk_b+prev_mobk_b);
			prev_mobk_c = curr_mobk_c;
			prev_mobk_b = curr_mobk_b;
		}
			curr_p_cap = p_ci + g*(delta_rho)*(height-h);
			curr_satu_b = computeBrineSaturation(curr_p_cap, scaling_parameter_C);

			curr_satu_e = (curr_satu_b-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
			curr_mob_c = computeRelPermCarbon(curr_satu_e, common_ctx.lambda_end_point_c);
			curr_mob_b = computeRelPermBrine(curr_satu_e, common_ctx.lambda_end_point_b);
			curr_mobk_c = curr_mob_c*k_values[n];
			curr_mobk_b = curr_mob_b*k_values[n];
			sum_c += 0.5*(prev_mobk_c+curr_mobk_c)*(height-dz*(n-1));
			sum_b += 0.5*(prev_mobk_b+curr_mobk_b)*(height-dz*(n-1));

	}

		float K_frac = 0;
		if (height < H){
			float last_bit = fminf(dz*n,H);
			// Add last part of integral to b
			curr_satu_b = 1;
			curr_satu_e = (curr_satu_b-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
			curr_mob_b = computeRelPermBrine(curr_satu_e, common_ctx.lambda_end_point_b);
			curr_mobk_b = curr_mob_b*k_values[n];
			prev_mobk_b = curr_mobk_b;
			sum_b += 0.5*(prev_mobk_b+curr_mobk_b)*(last_bit-height);
			K_frac = trapezoidal(H-last_bit, cpi_ctx.dz, ceil((H-last_bit)/dz), (k_values+n));
		}
		sum_b += K_frac*curr_mob_b;

		lambda_c_and_b[0] = sum_c/common_ctx.mu_c;
		lambda_c_and_b[1] = sum_b/common_ctx.mu_b;
}
__device__ void computeLambdaSimpson(float* lambda_c_and_b, float p_ci, float g, float delta_rho, float H, float height, float h,
							   float dz, int n, float* k_values, float scaling_parameter_C){

	float curr_p_cap = p_ci + g*(delta_rho)*(dz*0-h);
	float curr_satu_b = computeBrineSaturation(curr_p_cap, scaling_parameter_C);
	float curr_satu_e = (curr_satu_b-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
	float curr_mob_c = computeRelPermCarbon(curr_satu_e, common_ctx.lambda_end_point_c);
	float curr_mob_b = computeRelPermBrine(curr_satu_e, common_ctx.lambda_end_point_b);
	float curr_mobk_c_0 = curr_mob_c*k_values[0];
	float curr_mobk_b_0 = curr_mob_b*k_values[0];
	float curr_mobk_c_1, curr_mobk_c_2, curr_mobk_b_1, curr_mobk_b_2;
	float sum_c = 0;
	float sum_b = 0;
	int add = 0;
	if (n%2 != 0){
		n = n+1;
		add = 1;
	}
	dz = height/n;
		for (int i = 1; i <= n; i+=2){
			curr_p_cap = p_ci + g*(delta_rho)*(dz*i-h);
			curr_satu_b = computeBrineSaturation(curr_p_cap, scaling_parameter_C);
			curr_satu_e = (curr_satu_b-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
			curr_mob_c = computeRelPermCarbon(curr_satu_e, common_ctx.lambda_end_point_c);
			curr_mob_b = computeRelPermBrine(curr_satu_e, common_ctx.lambda_end_point_b);

			curr_mobk_c_1 = curr_mob_c*k_values[i];
			curr_mobk_b_1 = curr_mob_b*k_values[i];

			curr_p_cap = p_ci + g*(delta_rho)*(dz*(i+1)-h);
			curr_satu_b = computeBrineSaturation(curr_p_cap, scaling_parameter_C);
			curr_satu_e = (curr_satu_b-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
			curr_mob_c = computeRelPermCarbon(curr_satu_e, common_ctx.lambda_end_point_c);
			curr_mob_b = computeRelPermBrine(curr_satu_e, common_ctx.lambda_end_point_b);

			curr_mobk_c_2 = curr_mob_c*k_values[i+1];
			curr_mobk_b_2 = curr_mob_b*k_values[i+1];

			sum_c += (curr_mobk_c_0+4*curr_mobk_c_1 + curr_mobk_c_2);
			sum_b += (curr_mobk_b_0+4*curr_mobk_b_1 + curr_mobk_b_2);

			curr_mobk_c_0 = curr_mobk_c_2;
			curr_mobk_b_0 = curr_mobk_b_2;
		}
			sum_c = sum_c*(dz/(float)3.0);
			sum_b = sum_b*(dz/(float)3.0);


		float K_frac = 0;
		if (height < H){
			float last_bit = fminf(dz*n,H);
			K_frac = simpson(H-last_bit, cpi_ctx.dz, ceil((H-last_bit)/dz), (k_values+n-add));
		}
		sum_b += K_frac*curr_mob_b;

		lambda_c_and_b[0] = sum_c/common_ctx.mu_c;
		lambda_c_and_b[1] = sum_b/common_ctx.mu_b;
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
		if (H != 0){
			global_index(cpi_ctx.K.ptr, cpi_ctx.K.pitch, xid, yid,0)[0] = K/H;
		}
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
		float height = fminf(H,h);
		int nIntervalsForh = ceil(height/cmi_ctx.dz);

		float K = global_index(cmi_ctx.K.ptr, cmi_ctx.K.pitch, xid, yid, 0)[0];

		float* k_values = global_index(cpi_ctx.perm_distribution, xid, yid, 0, 0);

		float lambda_c_and_b[2];
		float dlambda_b, dlambda_c;

		float C = global_index(cmi_ctx.scaling_parameter_C.ptr, cmi_ctx.scaling_parameter_C.pitch,
							   xid, yid, 0)[0];

		computeLambda(lambda_c_and_b, cmi_ctx.p_ci, common_ctx.g, common_ctx.delta_rho, H, height, h, cmi_ctx.dz,
				nIntervalsForh, k_values, C);
		if (K != 0){

			global_index(cmi_ctx.Lambda_c.ptr, cmi_ctx.Lambda_c.pitch, xid, yid, 0)[0] =  lambda_c_and_b[0]/(K);
			global_index(cmi_ctx.Lambda_b.ptr, cmi_ctx.Lambda_b.pitch, xid, yid, 0)[0] =  lambda_c_and_b[1]/(K);
			float s_e = (1-common_ctx.s_c_res-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
			float rel_perm_b_at_h = computeRelPermBrine(s_e, common_ctx.lambda_end_point_b);
			float rel_perm_c_at_h = computeRelPermCarbon(s_e, common_ctx.lambda_end_point_c);

			if (cmi_ctx.perm_type == PERM_CONSTANT){
				dlambda_c = 1/(common_ctx.mu_c);
				dlambda_b = 1/(common_ctx.mu_b);
			} else if (cmi_ctx.perm_type == PERM_VARIATIONAL){
				dlambda_c = common_ctx.lambda_end_point_c*k_values[nIntervalsForh]/(common_ctx.mu_c*K);
				dlambda_b = common_ctx.lambda_end_point_b*k_values[nIntervalsForh]/(common_ctx.mu_b*K);
			} else {
				printf("ERROR: Perm type no set in config");
			}

			global_index(cmi_ctx.dLambda_c.ptr, cmi_ctx.dLambda_c.pitch, xid, yid, 0)[0] = dlambda_c;
			global_index(cmi_ctx.dLambda_b.ptr, cmi_ctx.dLambda_b.pitch, xid, yid, 0)[0] = dlambda_b;
		}
	}
}

void callCoarseMobIntegrationKernel(dim3 grid, dim3 block, int gridDimX, CoarseMobIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(cmi_ctx, args, sizeof(CoarseMobIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	CoarseMobIntegrationKernel<<<grid, block>>>(gridDimX);
}

__global__ void TimestepReductionKernel(){


	extern __shared__ float sdata[];

	volatile float* sdata_volatile = sdata;
	unsigned int tid = threadIdx.x;
	int threads = trk_ctx.nThreads;
	float dt;

	sdata[tid] = FLT_MAX;

	for (unsigned int i=tid; i<trk_ctx.nElements; i += threads)
		sdata[tid] = min(sdata[tid], trk_ctx.dt_vec[i]);
		__syncthreads();
	//Now, reduce all elements into a single element
	if (threads >= 512) {
		if (tid < 256) sdata[tid] = min(sdata[tid], sdata[tid + 256]);
		__syncthreads();
	}
	if (threads >= 256) {
		if (tid < 128) sdata[tid] = min(sdata[tid], sdata[tid + 128]);
		__syncthreads();
	}
	if (threads >= 128) {
		if (tid < 64) sdata[tid] = min(sdata[tid], sdata[tid + 64]);
		__syncthreads();
	}
	if (tid < 32) {
		if (threads >= 64) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid + 32]);
		if (tid < 16) {
			if (threads >= 32) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid + 16]);
			if (threads >= 16) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  8]);
			if (threads >=  8) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  4]);
			if (threads >=  4) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  2]);
			if (threads >=  2) sdata_volatile[tid] = min(sdata_volatile[tid], sdata_volatile[tid +  1]);
		}

		if (tid == 0) {
			dt = sdata_volatile[tid];
			if (dt == FLT_MAX) {
				dt = 200*24*60*60;
			}
			dt = dt*trk_ctx.cfl_scale*(1-common_ctx.s_b_res-common_ctx.s_c_res);
			float tf = trk_ctx.global_dt[2];
			float t = trk_ctx.global_dt[1];
			trk_ctx.global_dt[0] = fminf(dt, tf-t);
			if ((tf-t) < dt)
					trk_ctx.global_dt[1] = 0;
			else
				trk_ctx.global_dt[1] += trk_ctx.global_dt[0];

		}
	}

}

void callTimestepReductionKernel(int nThreads, TimestepReductionKernelArgs* args){
	cudaMemcpyToSymbolAsync(trk_ctx, args, sizeof(TimestepReductionKernelArgs), 0, cudaMemcpyHostToDevice);
	TimestepReductionKernel<<<1, nThreads, sizeof(float)*nThreads>>>();
}



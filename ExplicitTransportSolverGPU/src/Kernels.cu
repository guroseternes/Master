#include "Kernels.h"

__constant__ FluxKernelArgs fk_ctx;
__constant__ CommonArgs common_ctx;
__constant__ TimeIntegrationKernelArgs tik_ctx;
__constant__ CoarsePermIntegrationKernelArgs cpi_ctx;
__constant__ CoarseMobIntegrationKernelArgs cmi_ctx;
__constant__ TimestepReductionKernelArgs trk_ctx;


void initAllocate(CommonArgs* args1, CoarsePermIntegrationKernelArgs* args2,
				  CoarseMobIntegrationKernelArgs* args3, FluxKernelArgs* args4,
				  TimeIntegrationKernelArgs* args5, TimestepReductionKernelArgs* args6){
	cudaHostAlloc(&args1, sizeof(CommonArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args2, sizeof(CoarsePermIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args3, sizeof(CoarseMobIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args4, sizeof(FluxKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args5, sizeof(TimeIntegrationKernelArgs), cudaHostAllocWriteCombined);
	cudaHostAlloc(&args6, sizeof(TimeIntegrationKernelArgs), cudaHostAllocWriteCombined);
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

__device__ float computeBrineSaturation(float p_cap, float C){
	return fmaxf(C*C/((C+p_cap)*(C+p_cap)), common_ctx.s_b_res);
}

__device__ float computeRelPermBrine(float s_e, float lambda_end_point_b){
		return (pow(s_e, 1)*lambda_end_point_b);
}
__device__ float computeRelPermCarbon(float s_e, float lambda_end_point_c){
		return (pow(1-s_e, 1)*lambda_end_point_c);
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

inline __device__ float computeFluxEast(float (&U)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&lambda_c)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&lambda_b)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&dlambda_c)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&dlambda_b)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&h)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float (&z)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float z_diff,
									   float(&normal_z)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float K_face, float g_vec, float pv, float H,
									   unsigned int i, unsigned int j, float &upwindThis){
	float face_mob_c, face_mob_b, dface_mob_c, dface_mob_b, tot_mob, F_c;
	float U_b, U_c;
	U_c = 0;
	U_b = 0;
	float h_diff, g_flux, b;
	float delta_rho = common_ctx.delta_rho;
	float ff = 0;
	z_diff= z[i][j]-z[i+1][j];
	upwindThis = z_diff;
	h_diff = h[i][j]*normal_z[i][j]-h[i+1][j]*normal_z[i+1][j];
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
			//upwindThis = 0;
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
			//upwindThis = 2;
		} else {
			face_mob_c = lambda_c[i+1][j];
			dface_mob_c = dlambda_c[i+1][j];
		}
	}

	tot_mob = face_mob_c + face_mob_b;
	//if (face_mob_c == 0)
	//	face_mob_c = 0.00000000001;
	F_c = 0;
	float ff1 = 0;
	float ff2 = 0;
	if (tot_mob != 0) {
		F_c = face_mob_c/tot_mob;
		U_c = F_c*(U[i][j]+face_mob_b*g_flux);
		U_b = (1-F_c)*(U[i][j]-face_mob_c*g_flux);
		ff1 = fmaxf(face_mob_b*dface_mob_c*abs(U_c)/(tot_mob*face_mob_c),0);
		ff2 = fmaxf(face_mob_c*dface_mob_b*abs(U_b)/(tot_mob*face_mob_b),0);
		ff = ff1+ff2-(g_vec*delta_rho*K_face*face_mob_b*F_c);
	}
	float dt_temp = FLT_MAX;
	if (pv != 0)
		dt_temp = pv/ff;
	// Reuse of memory
	upwindThis = ff;
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
									   float z_diff,
									   float (&normal_z)[BLOCKDIM_X_FLUX][SM_BLOCKDIM_Y],
									   float K_face, float g_vec, float pv, float H,
									   unsigned int i, unsigned int j, float& upwindThis){
	float face_mob_c, face_mob_b, dface_mob_c, dface_mob_b, tot_mob, F_c;
	float U_b, U_c;
	U_c = 0;
	U_b = 0;
	float h_diff, b, g_flux;
	float delta_rho = common_ctx.delta_rho;
	float ff = 0;
	h_diff = h[i][j]*normal_z[i][j]-h[i][j+1]*normal_z[i][j+1];
	//upwindThis = h_diff;
	z_diff= z[i][j]-z[i][j+1];
	b = z_diff + h_diff;
	g_flux = -g_vec*b*delta_rho*K_face;
	//upwindThis = K_face;
	bool aa = !(U[i][j]<0) & !(g_flux<0);
	bool bb = !(U[i][j]>0) & !(g_flux>0);
	face_mob_c = lambda_c[i][j];
	face_mob_b = lambda_b[i][j];
	upwindThis = U[i][j];

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
			//upwindThis = 1;
		} else {
			face_mob_b = lambda_b[i][j+1];
			dface_mob_b = dlambda_b[i][j+1];
			//upwindThis = 2;
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
			//upwindThis = 2;
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
	if (pv != 0)
		dt_temp = pv/ff;
	// Reuse of memory
	U[i][j] = U_c;
	upwindThis = ff;
	return dt_temp;
}

__global__ void FluxKernel(){

	int border = common_ctx.border;
	int noborder = 0;
	//float r = FLT_MAX;

	// Global id
	int xid = blockIdx.x*TILEDIM_X + threadIdx.x-border;
    int yid = blockIdx.y*TILEDIM_Y + threadIdx.y-border;

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

    //TIMESTEP BLOCK
	const int nthreads = BLOCKDIM_X_FLUX*BLOCKDIM_Y_FLUX;
	__shared__ float timeStep[BLOCKDIM_X_FLUX][BLOCKDIM_Y_FLUX];
	float default_ = FLT_MAX;
	float dt_local = default_;
	timeStep[i][j] = default_;

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
    float z_diff_east = global_index(fk_ctx.z_diff_east.ptr, fk_ctx.z_diff_east.pitch, new_xid, new_yid, noborder)[0];
    float z_diff_north = global_index(fk_ctx.z_diff_north.ptr, fk_ctx.z_diff_north.pitch, new_xid, new_yid, noborder)[0];
    float g_vec_east = global_index(fk_ctx.g_vec_east.ptr, fk_ctx.g_vec_east.pitch, new_xid, new_yid, noborder)[0];
    float g_vec_north = global_index(fk_ctx.g_vec_north.ptr, fk_ctx.g_vec_north.pitch, new_xid, new_yid, noborder)[0];
    float K_face_east  = global_index(fk_ctx.K_face_east.ptr, fk_ctx.K_face_east.pitch, new_xid, new_yid, noborder)[0];
    float K_face_north = global_index(fk_ctx.K_face_north.ptr, fk_ctx.K_face_north.pitch, new_xid, new_yid, noborder)[0];
    float pv = global_index(common_ctx.pv.ptr, common_ctx.pv.pitch, new_xid, new_yid, noborder)[0];
    float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, new_xid, new_yid, noborder)[0];
    float upwindThis = 0;
    float upwindThisNorth = 0;

    __syncthreads();

    if (i < (TILEDIM_X+1) && j < (TILEDIM_Y+1)) {
    	dt_local = computeFluxEast(U_local_x, lambda_c_local, lambda_b_local,
    			dlambda_c_local, dlambda_b_local, h_local, z_local, z_diff_east, normal_z_local, K_face_east, g_vec_east, pv,H, i, j, upwindThis);
    	dt_local = fminf(dt_local, computeFluxNorth(U_local_y, lambda_c_local, lambda_b_local,
    			dlambda_c_local, dlambda_b_local, h_local, z_local, z_diff_north, normal_z_local, K_face_north ,g_vec_north, pv,H, i, j, upwindThisNorth));
    	if (global_index(common_ctx.H.ptr, common_ctx.H.pitch, new_xid, new_yid, noborder)[0] == 0) {
    		dt_local = default_;
    		U_local_x[i][j] = 0;
    		U_local_y[i][j] = 0;
    	}
    	if (xid == 0 || yid == 0 || xid == common_ctx.nx || yid == common_ctx.ny) {
    		dt_local = default_;
    		U_local_x[i][j] = 0;
    		U_local_y[i][j] = 0;
    	}
    	if (active_east == -1){
    		U_local_x[i][j] = 0;
    		//dt_local = default_;
    	}
    	if (xid == 42 && yid <= 43 ){
    		U_local_x[i][j] = 0;
    		//U_local_x[i+1][j] = 0;
    	}
    	if (upwindThisNorth == 1){
    		//U_local_y[i][j] = 0;
    	}
    	if (active_north == -1){
    		//U_local_y[i][j] = 0;
    	    //dt_local = default_;
    	}
    }


    int p = threadIdx.y*blockDim.x+threadIdx.x;

    __syncthreads();

    if (xid > -1 && xid < common_ctx.nx && yid > -1 && yid < common_ctx.ny){
        if (i < TILEDIM_X+1 && i > 0 && j < TILEDIM_Y+1 && j > 0) {
				float r = (U_local_x[i][j] - U_local_x[i-1][j]) + (U_local_y[i][j] - U_local_y[i][j-1]);
				float source = global_index(fk_ctx.source.ptr, fk_ctx.source.pitch , new_xid, new_yid, noborder)[0];
				float F_c = lambda_c/(lambda_c+lambda_b);
				global_index(fk_ctx.test_output.ptr, fk_ctx.test_output.pitch , new_xid, new_yid, noborder)[0] = upwindThis;
				r = r -fmaxf(source, 0) - fminf(source, 0)*F_c;
				global_index(fk_ctx.R.ptr, fk_ctx.R.pitch, new_xid, new_yid, noborder)[0] = r;
				timeStep[i][j] = dt_local;
        }
    }
    //timeStep[i][j] = dt_local;
    __syncthreads();
    		volatile float* B_volatile = timeStep[0];
    		//int p = threadIdx.y*blockDim.x+threadIdx.x; //reuse p for indexing
    	/*	if (xid < flux_ctx.nx && new_yid < flux_ctx.ny && xid >= 0 && yid >=0){
    			timeStep[0][p] = r;
    		}
    	*/
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

void callFluxKernel(dim3 grid, dim3 block, FluxKernelArgs* args){
	cudaMemcpyToSymbolAsync(fk_ctx, args, sizeof(FluxKernelArgs), 0, cudaMemcpyHostToDevice);
	FluxKernel<<<grid, block>>>();
}

inline __device__ float computeSatu(float z, float C){
	float curr_p_cap = common_ctx.p_ci + common_ctx.g*common_ctx.delta_rho*(-z);
	return 1.0-computeBrineSaturation(curr_p_cap, C);
}

inline __device__ void higherResolutionIteration(float& z, float& sum, float C, float& prev_s_c, float& curr_s_c, float& dz, float S_c_new){
	z -= dz;
	int i = 0;
	sum -= 0.5*dz*(curr_s_c + prev_s_c);
	curr_s_c = prev_s_c;
	dz = dz/10;
	while (sum < S_c_new && i <10){
		prev_s_c = curr_s_c;
		z += dz;
		curr_s_c = computeSatu(z, C);
		sum += 0.5*dz*(curr_s_c + prev_s_c);
		i++;
	}
}

inline __device__ void higherResolutionIterationWithShift(float H, float& z, float& z_bottom, float& sum, float C, float& prev_s_c, float& curr_s_c,
														   float& prev_s_c_bottom, float& curr_s_c_bottom, float& dz, float S_c_new, bool& firstIter,
														    float& sum_overflow){
	float diff = 1;
	int nIter;
	if (firstIter){
		sum -= 0.5*(z-H)*(curr_s_c + prev_s_c);
		z -= (z-H);
		nIter = 100;

	} else{
		if (z_bottom > 0){
			z_bottom -= dz;
			sum += 0.5*dz*(curr_s_c_bottom + prev_s_c_bottom);
			curr_s_c_bottom = prev_s_c_bottom;
		}
		z -= dz;
		sum -= 0.5*dz*(curr_s_c + prev_s_c);
		sum_overflow -= 0.5*dz*(curr_s_c_bottom + prev_s_c_bottom);
		curr_s_c = prev_s_c;
		dz = dz/10;
		nIter = 10;
	}
	int i = 0;
	while (sum < S_c_new && i<nIter && diff > 0){
		prev_s_c = curr_s_c;
		prev_s_c_bottom = curr_s_c_bottom;
		z += dz;
		curr_s_c = computeSatu(z, C);
		sum += 0.5*dz*(curr_s_c + prev_s_c);
		if (z > H){
			z_bottom += dz;
			curr_s_c_bottom = computeSatu(z_bottom, C);
			sum -= 0.5*dz*(curr_s_c_bottom + prev_s_c_bottom);
			sum_overflow += 0.5*dz*(curr_s_c_bottom + prev_s_c_bottom);
			diff = 0.5*dz*(curr_s_c + prev_s_c) - 0.5*dz*(curr_s_c_bottom + prev_s_c_bottom);
		}
		//if (S_c_new > 74 && S_c_new <75.5)
			//	printf("sum5: %.6f z: %.6f z_bottom %.6f S_c_new %.6f dz: %.7f i: %i diff %.7f curr_s_c %.5f sum_overflow %.5f\n", sum, z, z_bottom, S_c_new, dz, i, diff, curr_s_c, sum_overflow);
		i++;
	}
	firstIter = false;
}

inline __device__ float solveForh(float S_c_new, float H, float h, float p_ci, float dz, float delta_rho, float g, float C, bool print){
	float eps = dz/(10*10*10*10*10*10); // 7 times
	float sum = 0;
	float z = 0;
	print = false;
	float curr_s_c = computeSatu(z, C);
	float prev_s_c = 0;
	float z_bottom = 0;
	float curr_s_c_bottom = computeSatu(z, C);
	float prev_s_c_bottom = 0;
	float overflow = 0;
	int i = 0;
	if (H > eps && eps < S_c_new ){
		while (sum < S_c_new && i < 200 && z < H + dz){
			prev_s_c = curr_s_c;
			z += dz;
			curr_s_c = computeSatu(z, C);
			sum += 0.5*dz*(curr_s_c + prev_s_c);
			if (print)
				printf("prev_s_c %.9f z: %.9f sum %.9f \n", prev_s_c, z, sum);
			i++;
		}
		int nIter = 0;
		while (z < H && nIter < 5) {
			higherResolutionIteration(z, sum, C, prev_s_c, curr_s_c, dz, S_c_new);
			if (print)
				printf("INSIDE INTEGRAL sum5: %.8f z5: %.8f S_c_new %.8f dz: %.6f h: %.6f nIter: %i \n", sum, z, S_c_new, dz, h, nIter);
			nIter ++;
		}
		bool firstIter = true;
		float sum_overflow = 0;
		while (nIter < 5){
			if (print)
				printf("BEFORE ITER sum5: %.8f z5: %.8f S_c_new %.8f dz: %.6f overflow: %.6f nIter: %i \n", sum, z, S_c_new, dz, overflow, nIter);
			higherResolutionIterationWithShift(H, z, z_bottom, sum, C, prev_s_c, curr_s_c,
									          prev_s_c_bottom, curr_s_c_bottom, dz, S_c_new, firstIter, sum_overflow);
			nIter++;

		}

	}
	if (z > 0){
		return z;
	}
	else
		return 0;

}

__global__ void TimeIntegrationKernel(int gridDimX){

	// Global id
	int noborder = 0; //common_ctx.border;
	int dt = tik_ctx.global_dt[0];
	int xid = (cmi_ctx.active_block_indexes.ptr[blockIdx.x] % gridDimX)*blockDim.x + threadIdx.x;
    int yid = (cmi_ctx.active_block_indexes.ptr[blockIdx.x]/gridDimX)*blockDim.y + threadIdx.y;
    xid = fminf(xid, common_ctx.nx-1);
    yid = fminf(yid, common_ctx.ny-1);
	if (xid == 51 && yid == 51)
		printf("TIME STEP %.2f", tik_ctx.global_dt[0]);
    //global_index(common_ctx.active_cells.ptr, common_ctx.active_cells.pitch, xid, yid, border)[0] = 0;

    float H = global_index(common_ctx.H.ptr, common_ctx.H.pitch, xid, yid, noborder)[0];
    float pv = global_index(tik_ctx.pv.ptr, tik_ctx.pv.pitch, xid, yid, noborder)[0];
    float vol_old, vol_new;
    float S_c_new = 0;
    bool print = false;
    float S_c_old = global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0];
    vol_old = global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0]*pv;

    float r = global_index(tik_ctx.R.ptr, tik_ctx.R.pitch, xid, yid, noborder)[0];
	float h =  global_index(tik_ctx.h.ptr, tik_ctx.h.pitch, xid, yid, 0)[0];
	global_index(tik_ctx.h.ptr, tik_ctx.h.pitch, xid, yid, 0)[0] = 0;
    vol_new = vol_old - dt*r;
    if (pv != 0){
		S_c_new = vol_new/pv;
		global_index(tik_ctx.S_c.ptr, tik_ctx.S_c.pitch, xid, yid, noborder)[0] = S_c_new;

		float C = global_index(tik_ctx.scaling_parameter_C.ptr, tik_ctx.scaling_parameter_C.pitch,
							   xid, yid, 0)[0];

		//global_index(tik_ctx.vol_new.ptr, tik_ctx.vol_new.pitch, xid, yid, noborder)[0] = S_c_new/H;
		if (S_c_new/H > 1)
			printf("SATU WARNING");
		if (xid == 51 && yid == 51){
			printf("Saturation %.8f vol: %.8f pv: %.8f\n", S_c_new/H, vol_new, pv);
			print = true;
		}
		h  = solveForh(S_c_new, H, h, common_ctx.p_ci, tik_ctx.dz, common_ctx.delta_rho, common_ctx.g, C, print);
		global_index(tik_ctx.vol_new.ptr, tik_ctx.vol_new.pitch, xid, yid, noborder)[0] = vol_new;
		global_index(tik_ctx.vol_old.ptr, tik_ctx.vol_old.pitch, xid, yid, noborder)[0] = vol_old;
		global_index(tik_ctx.h.ptr, tik_ctx.h.pitch, xid, yid, noborder)[0] = h;
    }

}

void callTimeIntegration(dim3 grid, dim3 block, int gridDimX, TimeIntegrationKernelArgs* args){
	cudaMemcpyToSymbolAsync(tik_ctx, args, sizeof(TimeIntegrationKernelArgs), 0, cudaMemcpyHostToDevice);
	TimeIntegrationKernel<<<grid, block>>>(gridDimX);
}

// Function to compute the capillary pressure in the subintervals
__device__ float computeLambda(float* lambda_c_and_b, float p_ci, float g, float delta_rho, float H, float height, float h,
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
			/*
			if (abs(H-83)<0.5){
				printf("curr_satu_b %.15f diff %.15f curr_mob_b %.15f sum_b %.15f \n", curr_satu_b, curr_satu_b - common_ctx.s_b_res, curr_mob_b, sum_b);
			}
			*/
			curr_mobk_c = curr_mob_c*k_values[i];
			curr_mobk_b = curr_mob_b*k_values[i];
			sum_c += dz*0.5*(curr_mobk_c+prev_mobk_c);
			sum_b += dz*0.5*(curr_mobk_b+prev_mobk_b);
			prev_mobk_c = curr_mobk_c;
			prev_mobk_b = curr_mobk_b;
		}
			curr_p_cap = p_ci + g*(delta_rho)*(height-h);
			curr_satu_b = computeBrineSaturation(curr_p_cap, scaling_parameter_C);
		/*
			if (abs(H-h)<0.02 && h > H){
				printf("curr_satu_b %.15f diff 5.15f \n", curr_satu_b, curr_satu_b - common_ctx.s_b_res);
			}

			*/
			curr_satu_e = (curr_satu_b-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
			curr_mob_c = computeRelPermCarbon(curr_satu_e, common_ctx.lambda_end_point_c);
			curr_mob_b = computeRelPermBrine(curr_satu_e, common_ctx.lambda_end_point_b);
			curr_mobk_c = curr_mob_c*k_values[n];
			curr_mobk_b = curr_mob_b*k_values[n];
			sum_c += 0.5*(prev_mobk_c+curr_mobk_c)*(height-dz*(n-1));
			sum_b += 0.5*(prev_mobk_b+curr_mobk_b)*(height-dz*(n-1));
			/*
			if (abs(H-83)<0.5){
				printf("curr_satu_b %.15f diff %.15f curr_mob_b %.15f sum_b %.15f \n", curr_satu_b, curr_satu_b - common_ctx.s_b_res, curr_mob_b, sum_b);
			}
			*/

	}
		if (abs(H-83.42)<0.006){
			printf("curr_satu_b %.15f diff %.15f curr_mob_b %.15f sum_b %.15f \n", curr_satu_b, curr_satu_b - common_ctx.s_b_res, curr_mob_b, sum_b);
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

		if (abs(H-83.42)<0.006){
			printf("curr_satu_b %.15f diff %.15f curr_mob_b %.15f sum_b %.15f H %.15f \n", curr_satu_b, curr_satu_b - common_ctx.s_b_res, curr_mob_b, sum_b,H);
		}

		K_frac = trapezoidal(H-last_bit, cpi_ctx.dz, ceil((H-last_bit)/dz), (k_values+n));
		}
		/*if (abs(H-83.425045)<0.02 && h > H){
					printf("K_frac %.15f \n", K_frac);
		}
		*/
		sum_b += K_frac*curr_mob_b;
		//sum_b = K_frac*curr_mob_b;
		sum_b = sum_b/common_ctx.mu_b;
		float sum_b_2 = (H-height)*0.85f*k_values[0]/common_ctx.mu_b;
		float diff = (sum_b -sum_b_2)/sum_b_2;
		if (diff > 0.00001){
			printf("Difference between the two %.15f sum_b %.15f sum_b %.15f H %.4f h %.9f H-dz*n %.9f\n", diff, sum_b, sum_b_2, H, height, H-dz*n);
		}
		lambda_c_and_b[0] = sum_c/common_ctx.mu_c;
		lambda_c_and_b[1] = sum_b;
		//if (lambda_c_and_b[1] < 2.6)
			//lambda_c_and_b[1] = 0;
		return (sum_b);
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

		float C = global_index(cmi_ctx.scaling_parameter_C.ptr, cmi_ctx.scaling_parameter_C.pitch,
							   xid, yid, 0)[0];

		float L = computeLambda(lambda_c_and_b, cmi_ctx.p_ci, common_ctx.g, common_ctx.delta_rho, H, height, h, cmi_ctx.dz,
				nIntervalsForh, k_values, C);
		if (K != 0){
			global_index(cmi_ctx.Lambda_c.ptr, cmi_ctx.Lambda_c.pitch, xid, yid, 0)[0] =  lambda_c_and_b[0]/(K);
			global_index(cmi_ctx.Lambda_b.ptr, cmi_ctx.Lambda_b.pitch, xid, yid, 0)[0] =  lambda_c_and_b[1]/(K);
			if (xid == 51 && yid ==51) // && abs(H-h) < 0.02)
				printf("H-h:%.10f lambda_b %.4f lambda_c %.4f\n",H-h, lambda_c_and_b[1]/(K), lambda_c_and_b[0]/(K));
			float s_e = (1-common_ctx.s_c_res-common_ctx.s_b_res)/(1-common_ctx.s_b_res);
			float rel_perm_b_at_h = computeRelPermBrine(s_e, common_ctx.lambda_end_point_b);
			float rel_perm_c_at_h = computeRelPermCarbon(s_e, common_ctx.lambda_end_point_c);
			float dlambda_c = 1/(common_ctx.mu_c);
			float dlambda_b = 1/(common_ctx.mu_b);
			//float dlambda_c = rel_perm_c_at_h*k_values[nIntervalsForh]/(common_ctx.mu_c*K);
			//float dlambda_b = rel_perm_b_at_h*k_values[nIntervalsForh]/(common_ctx.mu_b*K);
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
			if (dt == FLT_MAX || dt < 200*24*60*60) {
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

		//	printf("TID %i",tid);
		}
	}

}

void callTimestepReductionKernel(int nThreads, TimestepReductionKernelArgs* args){
	cudaMemcpyToSymbolAsync(trk_ctx, args, sizeof(TimestepReductionKernelArgs), 0, cudaMemcpyHostToDevice);
	TimestepReductionKernel<<<1, nThreads, sizeof(float)*nThreads>>>();
}





#include <iostream>
#include <stdio.h>
#include "GpuPtr.h"
#include "CpuPtr.h"
#include "Kernels.h"
#include "Functions.h"
#include "Util.h"
#include "cuda.h"
#include "InitialConditions.h"
#include "matio.h"
#include "float.h"
#include <memory.h>

int main() {
	int nx, ny, nz, border;
	float dt, cfl_scale;
	float t, tf;
	t = 0;
	tf = 157680000;
	cfl_scale = 0.5;
	border = 1;
	dt = 4.9861 * pow((float)10, 7);
	char* filename = "dimensions.mat";
	cudaSetDevice(1);
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, device);
	printf("Name: %s\n", p.name);
	readDimensionsFromMATLABFile(filename, nx, ny, nz);
	printf("nx: %i, ny: %i nz: %i dt: %.10f", nx, ny, nz, dt);
	size_t free_mem;
	size_t total_mem;
	//cudaMemGetInfo(&free_mem, &total_mem);
	//printf("Memory free: %i total: %i", free_mem, total_mem);

	CpuPtr_2D H(nx, ny, 0, true);
	CpuPtr_2D top_surface(nx, ny, 0, true);
	CpuPtr_2D h(nx, ny, 0, true);
	CpuPtr_2D normal_z(nx, ny, 0, true);
	CpuPtr_3D perm3D(nx, ny, nz + 1, 0, true);
	CpuPtr_3D poro3D(nx, ny, nz + 1, 0, true);
	CpuPtr_2D pv(nx, ny, 0, true);
	CpuPtr_2D flux_north(nx, ny, border, true);
	CpuPtr_2D flux_east(nx, ny, border, true);
	CpuPtr_2D grav_north(nx, ny, 0, true);
	CpuPtr_2D grav_east(nx, ny, 0, true);
	CpuPtr_2D K_face_north(nx, ny, 0, true);
	CpuPtr_2D K_face_east(nx, ny, 0, true);
	CpuPtr_2D active_cells(nx, ny, 0, true);
	float dz;
	filename = "johansendata.mat";
	readFormationDataFromMATLABFile(filename, H.getPtr(), top_surface.getPtr(),
			h.getPtr(), normal_z.getPtr(), perm3D.getPtr(), poro3D.getPtr(),
			pv.getPtr(), flux_north.getPtr(), flux_east.getPtr(),
			grav_north.getPtr(), grav_east.getPtr(), K_face_north.getPtr(),
			K_face_east.getPtr(), dz);
	filename = "active_cells.mat";
	readActiveCellsFromMATLABFile(filename, active_cells.getPtr());
	//Test reading
	//readPermFromMATLABFile(filename2, perm3D);
	printf("dz: %.3f nz: %i\n", dz, nz);
	printf("H(29,25) %.3f east_grav(45,50): %.3f north_flux(50,50): %.16f\n",
			H(29, 25), grav_east(45, 50), flux_north(45, 50));
	printf("poro3d(45, 50, 1) : %.15f\n", poro3D(45, 50, 1));
	printf("perm3d(45, 50, 1) : %.15f\n", perm3D(45, 50, 1));


	// Files with results
	FILE* Lambda_integration_file;
	Lambda_integration_file = fopen("Lambda_integration_sparse.txt", "w");
	FILE* matlab_file;
	matlab_file = fopen("/home/guro/mrst-bitbucket/mrst-other/co2lab/toMATLAB.txt", "w");
	FILE* Check_results_file;
	Check_results_file = fopen("Check_results.txt", "w");

	// Cpu Pointer to store the results
	CpuPtr_2D CheckResults(nx, ny, 0, true);
	CpuPtr_2D zeros(nx, ny, 0, true);
	CpuPtr_2D zerosWithBorder(nx+2*border, ny+2*border,0,true);
	CpuPtr_2D Lambda(nx, ny, 0, true);

	//Initial Conditions
	InitialConditions IC(nx, ny, 5);
	printf("mu_c %.7f mu_b %.7f\n", IC.mu_c, IC.mu_b);
	IC.dz = dz;
	IC.createScalingParameterTable(H);
	IC.createInitialCoarseSatu(H, h);
	CpuPtr_2D scaling_parameter(nx, ny, 0, true);
	CpuPtr_2D initial_coarse_satu(nx, ny, 0, true);
	for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			scaling_parameter(i, j) = IC.scaling_parameter(i, j);
			initial_coarse_satu(i, j) = IC.initial_coarse_satu_c(i, j);
		}
	}

	// GPU
	// Block sizes
	int block_x = 16;
	int block_y = 16;
	// Set block and grid sizes and initialize gpu pointer
	dim3 grid;
	dim3 block;
	computeGridBlock(grid, block, nx, ny, block_x, block_y);
	dim3 grid_flux;
	dim3 block_flux;
	computeGridBlock(grid_flux, block_flux, nx + 2*border, ny + 2*border, BLOCKDIM_X,
			BLOCKDIM_Y, TILEDIM_X, TILEDIM_Y);

	// Fix dt vector
	int nElements = grid_flux.x*grid_flux.y;
	float dt_vector[nElements];
	for (int k = 0; k < nElements; k++){
		dt_vector[k] = FLT_MAX;
	}
	float global_time_data[3];
	global_time_data[0] = -1;
	global_time_data[1] = t;
	global_time_data[2] = tf;

	// Create mask for sparse grid on GPU
	std::vector<int> active_block_indexes;
	int n_active_blocks = 0;
	createGridMask(H, grid, block, nx, ny, active_block_indexes,
			n_active_blocks);
	printf("nBlocks: %i nActiveBlocks: %i fraction: %.5f\n", grid.x * grid.y,
			n_active_blocks, (float) n_active_blocks / (grid.x * grid.y));
	printf("dz: %.3f\n", IC.dz);
	dim3 new_sparse_grid(n_active_blocks, 1, 1);

	CpuPtr_2D n_interval_dist(nx, ny, 0, true);
	for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			n_interval_dist(i, j) = ceil(H(i, j) / IC.dz);
		}
	}

	CommonArgs common_args;
	CoarseMobIntegrationKernelArgs coarse_mob_int_args;
	CoarsePermIntegrationKernelArgs coarse_perm_int_args;
	FluxKernelArgs flux_kernel_args;
	TimeIntegrationKernelArgs time_int_kernel_args;
	TimestepReductionKernelArgs time_red_kernel_args;

	initAllocate(&common_args, &coarse_perm_int_args, &coarse_mob_int_args,
			&flux_kernel_args, &time_int_kernel_args, &time_red_kernel_args);

	// Allocate and set data on the GPU
	GpuPtr_3D perm3D_device(nx, ny, nz + 1, 0, perm3D.getPtr());
	GpuPtr_2D Lambda_c_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D Lambda_b_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D dLambda_c_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D dLambda_b_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D scaling_parameter_C_device(nx, ny, 0, scaling_parameter.getPtr());
	GpuPtr_2D K_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D H_device(nx, ny, 0, H.getPtr());
	GpuPtr_2D h_device(nx, ny, 0, h.getPtr());
	GpuPtr_2D top_surface_device(nx, ny, 0, top_surface.getPtr());
	GpuPtr_2D nInterval_device(nx, ny, 0, n_interval_dist.getPtr());
	GpuPtr_2D U_x_device(nx, ny, border, flux_east.getPtr());
	GpuPtr_2D U_y_device(nx, ny, border, flux_north.getPtr());
	GpuPtr_2D K_face_east_device(nx, ny, 0, K_face_east.getPtr());
	GpuPtr_2D K_face_north_device(nx, ny, 0, K_face_north.getPtr());
	GpuPtr_2D grav_east_device(nx, ny, 0, grav_east.getPtr());
	GpuPtr_2D grav_north_device(nx, ny, 0, grav_north.getPtr());
	GpuPtr_2D normal_z_device(nx, ny, 0, normal_z.getPtr());
	GpuPtr_2D R_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D pv_device(nx, ny, 0, pv.getPtr());
	GpuPtr_2D active_cells_device(nx, ny, 0, active_cells.getPtr());
	GpuPtr_2D zeros_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D coarse_satu_device(nx, ny, 0, initial_coarse_satu.getPtr());
	GpuPtr_1D global_dt_device(3, global_time_data);
	GpuPtr_1D dt_vector_device(nElements, dt_vector);
	GpuPtrInt_1D active_block_indexes_device(n_active_blocks,
			&active_block_indexes[0]);

	setCommonArgs(&common_args, IC.p_ci, IC.delta_rho, IC.g, IC.mu_c, IC.mu_b,
			IC.s_c_res, IC.s_b_res, IC.lambda_end_point_c,
			IC.lambda_end_point_b, active_cells_device.getRawPtr(),
			H_device.getRawPtr(), pv_device.getRawPtr(),
			nx, ny, border);
	setupGPU(&common_args);

	printf("Coarse perm integration error: %s\n",
			cudaGetErrorString(cudaGetLastError()));
	// Set arguments and run coarse permeability integration kernel
	setCoarsePermIntegrationKernelArgs(&coarse_perm_int_args,
			K_device.getRawPtr(), perm3D_device.getRawPtr(),
			nInterval_device.getRawPtr(), IC.dz);
	callCoarsePermIntegrationKernel(grid, block, &coarse_perm_int_args);
	printf("Coarse perm integration error: %s\n",
			cudaGetErrorString(cudaGetLastError()));
	setTimeIntegrationKernelArgs(&time_int_kernel_args, global_dt_device.getRawPtr(),
			IC.integral_res,pv_device.getRawPtr(), h_device.getRawPtr(),
			R_device.getRawPtr(),coarse_satu_device.getRawPtr(),
			scaling_parameter_C_device.getRawPtr(), zeros_device.getRawPtr());

	// Set arguments and run coarse mobilty integration kernel
	setCoarseMobIntegrationKernelArgs(&coarse_mob_int_args,
			Lambda_c_device.getRawPtr(), Lambda_b_device.getRawPtr(),
			dLambda_c_device.getRawPtr(), dLambda_b_device.getRawPtr(),
			h_device.getRawPtr(), perm3D_device.getRawPtr(),
			K_device.getRawPtr(), nInterval_device.getRawPtr(),
			scaling_parameter_C_device.getRawPtr(),
			active_block_indexes_device.getRawPtr(), IC.p_ci, IC.dz);

	setFluxKernelArgs(&flux_kernel_args,
			Lambda_c_device.getRawPtr(),Lambda_b_device.getRawPtr(),
			dLambda_c_device.getRawPtr(), dLambda_b_device.getRawPtr(),
			U_x_device.getRawPtr(), U_y_device.getRawPtr(), h_device.getRawPtr(),
			top_surface_device.getRawPtr(), normal_z_device.getRawPtr(),
			K_face_east_device.getRawPtr(), K_face_north_device.getRawPtr(),
			grav_east_device.getRawPtr(), grav_north_device.getRawPtr(),
			R_device.getRawPtr(), dt_vector_device.getRawPtr());

	setTimestepReductionKernelArgs(&time_red_kernel_args, TIME_THREADS, nElements, global_dt_device.getRawPtr(),
								   cfl_scale, dt_vector_device.getRawPtr());

	int iter = 0;
	while (t < tf && iter < 10){
		callCoarseMobIntegrationKernel(new_sparse_grid, block, grid.x, &coarse_mob_int_args);

		callFluxKernel(grid_flux, block_flux, &flux_kernel_args);

		callTimestepReductionKernel(TIME_THREADS, &time_red_kernel_args);

		callTimeIntegration(grid, block, &time_int_kernel_args);

		cudaMemcpy(global_time_data, global_dt_device.getRawPtr(), sizeof(float)*3, cudaMemcpyDeviceToHost);

		printf("Total time: %.3f timestep %.3f\n", global_time_data[1]/(60*60*24), global_time_data[0]/(60*60*24));

		t += global_time_data[0];
		iter++;
	}

	// Run function with timer
	double time_start = getWallTime();

	printf("Elapsed time %.5f", getWallTime() - time_start);

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	// Print to file

	printf("Load error: %s\n", cudaGetErrorString(cudaGetLastError()));

	h_device.download(zeros.getPtr(), 0, 0, nx, ny);
	//R_device.download(zeros.getPtr(), 0, 0, nx, ny);
	/*for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			H(i, j) = ceil(H(i, j) / IC.dz);
		}
	}
	*/
	zeros.printToFileComparison(Check_results_file, Lambda);
	zeros.printToFile(matlab_file);
	printf("Load error: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(dt_vector, dt_vector_device.getRawPtr(), sizeof(float)*(nElements), cudaMemcpyDeviceToHost);
	float mini = FLT_MAX;
	for (int i =0; i < nElements; i++){
		printf(" %.2f ", dt_vector[i]);
		if (dt_vector[i] < mini){
			mini = dt_vector[i];
		}
	}
	cudaMemcpy(global_time_data, global_dt_device.getRawPtr(), sizeof(float), cudaMemcpyDeviceToHost);

	printf("\nCudaMemcpy error: %s", cudaGetErrorString(cudaGetLastError()));
	printf("\n Global timestep manual: %.2f Global timestep from Kernel: %.2f\n", mini, global_time_data[0]);
	printf("dt from MATLAB: %.3f global_dt from kernel: %.3f\n", dt/(60*60*24), global_time_data[0]/(60*60*24));
	printf("FINITO");

	//printf("Check results %.3f", CheckResults(6,0));

}


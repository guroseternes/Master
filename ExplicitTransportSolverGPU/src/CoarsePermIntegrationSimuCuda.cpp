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
#include <stdlib.h>
#include <string.h>
#include "engine.h"
#include <memory.h>

int main() {
	int nx, ny, nz;
	float dt, dz;
	float t, tf;
	t = 0;
	tf = 157680000;
	char* filename = "dimensions.mat";
	cudaSetDevice(1);
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, device);
	printf("Name: %s\n", p.name);
	readDimensionsFromMATLABFile(filename, nx, ny, nz);
	InitialConditions IC(nx, ny, 5);
	printf("nx: %i, ny: %i nz: %i dt: %.10f", nx, ny, nz, dt);

	CpuPtr_2D H(nx, ny, 0, true);
	CpuPtr_2D top_surface(nx, ny, 0, true);
	CpuPtr_2D h(nx, ny, 0, true);
	CpuPtr_2D normal_z(nx, ny, 0, true);
	CpuPtr_3D perm3D(nx, ny, nz + 1, 0, true);
	CpuPtr_3D poro3D(nx, ny, nz + 1, 0, true);
	CpuPtr_2D pv(nx, ny, 0, true);
	CpuPtr_2D flux_north(nx, ny, IC.border, true);
	CpuPtr_2D flux_east(nx, ny, IC.border, true);
	CpuPtr_2D flux_north_2(nx, ny, IC.border, true);
	CpuPtr_2D flux_east_2(nx, ny, IC.border, true);
	CpuPtr_2D grav_north(nx, ny, 0, true);
	CpuPtr_2D grav_east(nx, ny, 0, true);
	CpuPtr_2D K_face_north(nx, ny, 0, true);
	CpuPtr_2D K_face_east(nx, ny, 0, true);
	CpuPtr_2D active_east(nx, ny, 0, true);
	CpuPtr_2D active_north(nx, ny, 0, true);

	filename = "johansendata.mat";
	readFormationDataFromMATLABFile(filename, H.getPtr(), top_surface.getPtr(),
			h.getPtr(), normal_z.getPtr(), perm3D.getPtr(), poro3D.getPtr(),
			pv.getPtr(), flux_north.getPtr(), flux_east.getPtr(),
			grav_north.getPtr(), grav_east.getPtr(), K_face_north.getPtr(),
			K_face_east.getPtr(), dz);
	filename = "active_cells.mat";
	readActiveCellsFromMATLABFile(filename, active_east.getPtr(), active_north.getPtr());

	printf("dz: %.3f nz: %i\n", dz, nz);
	printf("h(52,53) %.3f east_grav(45,50): %.3f north_flux(50,50): %.16f\n",
			h(52, 53), grav_east(45, 50), flux_north(45, 50));
	printf("poro3d(45, 50, 1) : %.15f\n", poro3D(45, 50, 1));
	printf("perm3d(45, 50, 1) : %.15f\n", perm3D(45, 50, 1));

	// Files with results
	FILE* Lambda_integration_file;
	Lambda_integration_file = fopen("Lambda_integration_sparse.txt", "w");
	FILE* matlab_file;
	matlab_file = fopen("/home/guro/mrst-bitbucket/mrst-other/co2lab/toMATLAB.txt", "w");
	FILE* Check_results_file;
	Check_results_file = fopen("Check_results.txt", "w");

	Engine *ep;
	mxArray *T = NULL, *result = NULL;

	if (!(ep = engOpen(""))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		return EXIT_FAILURE;
	}

	double * h_matlab_matrix;
	h_matlab_matrix = new double[nx*ny];
	T = mxCreateDoubleMatrix(nx, ny, mxREAL);
	engEvalString(ep, "cd ~/mrst-bitbucket/mrst-core;");
    engEvalString(ep, "startup;");
	engEvalString(ep, "startup_user;");
	engEvalString(ep, "cd ~/mrst-bitbucket/mrst-other/co2lab;");
	engEvalString(ep, "startuplocal");
	engEvalString(ep, "cd ~/mrst-bitbucket/mrst-other/co2lab/guro_code;");

	// Cpu Pointer to store the results
	CpuPtr_2D CheckResults(nx, ny, 0, true);
	CpuPtr_2D zeros(nx, ny, 0, true);
	CpuPtr_2D zerosWithBorder(nx+2*IC.border, ny+2*IC.border, 0, true);

	//Initial Conditions
	printf("mu_c %.7f mu_b %.7f\n", IC.mu_c, IC.mu_b);
	IC.dz = dz;
	IC.createnIntervalsTable(H);
	IC.createScalingParameterTable(H);
	IC.createInitialCoarseSatu(H, h);
	IC.computeAllGridBlocks();
	IC.createDtVec();

	float global_time_data[3];
	global_time_data[0] = IC.dt_test/3;
	global_time_data[1] = t;
	global_time_data[2] = tf;

	// Create mask for sparse grid on GPU
	std::vector<int> active_block_indexes;
	int n_active_blocks = 0;
	createGridMask(H, IC.grid, IC.block, nx, ny, active_block_indexes,
			n_active_blocks);
	printf("nBlocks: %i nActiveBlocks: %i fraction: %.5f\n", IC.grid.x * IC.grid.y,
			n_active_blocks, (float) n_active_blocks / (IC.grid.x * IC.grid.y));
	printf("dz: %.3f\n", IC.dz);
	dim3 new_sparse_grid(n_active_blocks, 1, 1);

	CommonArgs common_args;
	CoarseMobIntegrationKernelArgs coarse_mob_int_args;
	CoarsePermIntegrationKernelArgs coarse_perm_int_args;
	FluxKernelArgs flux_kernel_args;
	TimeIntegrationKernelArgs time_int_kernel_args;
	TimestepReductionKernelArgs time_red_kernel_args;

	initAllocate(&common_args, &coarse_perm_int_args, &coarse_mob_int_args,
			&flux_kernel_args, &time_int_kernel_args, &time_red_kernel_args);

	h.convertToDoublePointer(h_matlab_matrix);
	memcpy((void *)mxGetPr(T), (void *)h_matlab_matrix, sizeof(double)*nx*ny);
	engPutVariable(ep, "T", T);
    filename = "fluxes.mat";
	readFluxesFromMATLABFile(filename, flux_east_2.getPtr(), flux_north_2.getPtr());

	// Allocate and set data on the GPU
	GpuPtr_3D perm3D_device(nx, ny, nz + 1, 0, perm3D.getPtr());
	GpuPtr_2D Lambda_c_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D Lambda_b_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D dLambda_c_device(nx, ny, 0, zeros.getPtr());

	GpuPtr_2D dLambda_b_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D scaling_parameter_C_device(nx, ny, 0, IC.scaling_parameter.getPtr());
	GpuPtr_2D K_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D H_device(nx, ny, 0, H.getPtr());
	GpuPtr_2D h_device(nx, ny, 0, h.getPtr());
	GpuPtr_2D top_surface_device(nx, ny, 0, top_surface.getPtr());
	GpuPtr_2D nInterval_device(nx, ny, 0, IC.nIntervals.getPtr());
	GpuPtr_2D U_x_device(nx, ny, IC.border, flux_east_2.getPtr());
	GpuPtr_2D U_y_device(nx, ny, IC.border, flux_north_2.getPtr());
	GpuPtr_2D K_face_east_device(nx, ny, 0, K_face_east.getPtr());
	GpuPtr_2D K_face_north_device(nx, ny, 0, K_face_north.getPtr());
	GpuPtr_2D grav_east_device(nx, ny, 0, grav_east.getPtr());
	GpuPtr_2D grav_north_device(nx, ny, 0, grav_north.getPtr());
	GpuPtr_2D normal_z_device(nx, ny, 0, normal_z.getPtr());
	GpuPtr_2D R_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D pv_device(nx, ny, 0, pv.getPtr());
	GpuPtr_2D output_test_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D active_east_device(nx, ny, 0, active_east.getPtr());
	GpuPtr_2D active_north_device(nx, ny, 0, active_north.getPtr());
	GpuPtr_2D vol_old_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D vol_new_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D coarse_satu_device(nx, ny, 0, IC.initial_coarse_satu_c.getPtr());
	GpuPtr_1D global_dt_device(3, global_time_data);
	GpuPtr_1D dt_vector_device(IC.nElements, IC.dt_vector);
	GpuPtrInt_1D active_block_indexes_device(n_active_blocks,
			&active_block_indexes[0]);

	setCommonArgs(&common_args, IC.p_ci, IC.delta_rho, IC.g, IC.mu_c, IC.mu_b,
			IC.s_c_res, IC.s_b_res, IC.lambda_end_point_c, IC.lambda_end_point_b,
			active_east_device.getRawPtr(), active_north_device.getRawPtr(),
			H_device.getRawPtr(), pv_device.getRawPtr(),
			nx, ny, IC.border);
	setupGPU(&common_args);

	setCoarsePermIntegrationKernelArgs(&coarse_perm_int_args,
			K_device.getRawPtr(), perm3D_device.getRawPtr(),
			nInterval_device.getRawPtr(), IC.dz);
	callCoarsePermIntegrationKernel(IC.grid, IC.block, &coarse_perm_int_args);

	setTimeIntegrationKernelArgs(&time_int_kernel_args, global_dt_device.getRawPtr(),
			IC.integral_res,pv_device.getRawPtr(), h_device.getRawPtr(),
			R_device.getRawPtr(),coarse_satu_device.getRawPtr(),
			scaling_parameter_C_device.getRawPtr(),
			vol_old_device.getRawPtr(), vol_new_device.getRawPtr());

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
			R_device.getRawPtr(), dt_vector_device.getRawPtr(),
			output_test_device.getRawPtr());

	setTimestepReductionKernelArgs(&time_red_kernel_args, TIME_THREADS, IC.nElements, global_dt_device.getRawPtr(),
								   IC.cfl_scale, dt_vector_device.getRawPtr());

	int iter = 0;
	int iter2 = 0;
	float totTime = 0;
	//total time in years
	float totalTime = 500;
	while (totTime < totalTime && iter2 <50){
		t = 0;
		iter = 0;
		while (t < tf && iter < 20){
			callCoarseMobIntegrationKernel(new_sparse_grid, IC.block, IC.grid.x, &coarse_mob_int_args);

			callFluxKernel(IC.grid_flux, IC.block_flux, &flux_kernel_args);

			callTimestepReductionKernel(TIME_THREADS, &time_red_kernel_args);

			callTimeIntegration(IC.grid, IC.block, &time_int_kernel_args);

			cudaMemcpy(global_time_data, global_dt_device.getRawPtr(), sizeof(float)*3, cudaMemcpyDeviceToHost);

			//printf("Total time in years: %.3f time in this round %.3f timestep %.3f\n", totTime, t/(60*60*24), global_time_data[0]/(60*60*24));

			t += global_time_data[0];
			iter++;
		}
		//printf("Finished Iter Total time in years: %.3f time in this round %.3f timestep %.3f\n", totTime, global_time_data[1]/(60*60*24), global_time_data[0]/(60*60*24));
		totTime += tf/(60*60*24*365);
		h_device.download(h.getPtr(), 0, 0, nx, ny);
		h.convertToDoublePointer(h_matlab_matrix);
		memcpy((void *)mxGetPr(T), (void *)h_matlab_matrix, sizeof(double)*nx*ny);
		engPutVariable(ep, "T", T);
		engEvalString(ep, "pressureFunctionToRunfromCpp(T);");
	    filename = "fluxes.mat";
		readFluxesFromMATLABFile(filename, flux_east_2.getPtr(), flux_north_2.getPtr());
		U_x_device.upload(flux_east_2.getPtr(), 0, 0, nx, ny);
		U_y_device.upload(flux_north_2.getPtr(), 0, 0, nx, ny);
		//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
		iter2++;
	}

    engClose(ep);

	// Run function with timer
	double time_start = getWallTime();

	printf("Elapsed time %.5f", getWallTime() - time_start);

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	h_device.download(zeros.getPtr(), 0, 0, nx, ny);
	zeros.printToFileComparison(Check_results_file, h);
	zeros.printToFile(matlab_file);

	float total_volume_old = computeTotalVolume(vol_old_device, nx, ny);
	float total_volume_new = computeTotalVolume(vol_new_device, nx, ny);

	printf("total volume new %.2f total volume old %.2f frac: %.7f",
			total_volume_new, total_volume_old, (total_volume_old-total_volume_new)/(total_volume_old));

	printf("Load error: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(IC.dt_vector, dt_vector_device.getRawPtr(), sizeof(float)*(IC.nElements), cudaMemcpyDeviceToHost);
	float mini = FLT_MAX;
	for (int i =0; i < IC.nElements; i++){
		//printf("dt: %.3f\n", IC.dt_vector[i]);
		if (IC.dt_vector[i] < mini){
			mini = IC.dt_vector[i];
		}
	}

	printf("\nCudaMemcpy error: %s", cudaGetErrorString(cudaGetLastError()));
	printf("\n Global timestep manual: %.2f Global timestep from Kernel: %.2f\n", mini, global_time_data[0]);
	printf("dt from MATLAB: %.3f global_dt from kernel: %.3f\n", dt/(60*60*24), global_time_data[0]/(60*60*24));
	printf("FINITO");

	//printf("Check results %.3f", CheckResults(6,0));

}


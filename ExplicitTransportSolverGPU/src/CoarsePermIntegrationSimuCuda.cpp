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
	int year = 60*60*24*365;
	int stop_inject = 100;
	bool run = true;
	char* filename = "dimensions.mat";
	cudaSetDevice(0);
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
	CpuPtr_2D source(nx, ny, 0, true);
	CpuPtr_2D grav_north(nx, ny, 0, true);
	CpuPtr_2D grav_east(nx, ny, 0, true);
	CpuPtr_2D K_face_north(nx, ny, 0, true);
	CpuPtr_2D K_face_east(nx, ny, 0, true);
	CpuPtr_2D active_east(nx, ny, 0, true);
	CpuPtr_2D active_north(nx, ny, 0, true);
	CpuPtr_2D volume(nx, ny, 0,true);

	filename = "johansendata_at_0.mat";
	readFormationDataFromMATLABFile(filename, H.getPtr(), top_surface.getPtr(),
			h.getPtr(), normal_z.getPtr(), perm3D.getPtr(), poro3D.getPtr(),
			pv.getPtr(), flux_north.getPtr(), flux_east.getPtr(),
			grav_north.getPtr(), grav_east.getPtr(), K_face_north.getPtr(),
			K_face_east.getPtr(), dz);
	filename = "active_cells.mat";
	readActiveCellsFromMATLABFile(filename, active_east.getPtr(), active_north.getPtr());

	//readSourceFromMATLABFile(filename, source.getPtr());

	// Files with results
	FILE* Lambda_integration_file;
	Lambda_integration_file = fopen("Lambda_integration_sparse.txt", "w");
	FILE* matlab_file;
	matlab_file = fopen("/home/guro/mrst-bitbucket/mrst-other/co2lab/toMATLAB.txt", "w");
	FILE* matlab_file_2;
	matlab_file_2 = fopen("/home/guro/mrst-bitbucket/mrst-other/co2lab/toMATLAB1.txt", "w");
	FILE* Check_results_file;
	Check_results_file = fopen("Check_results.txt", "w");

	float dt_table[302];
	int size_dt_table = 0;

	filename = "dt_table.mat";
	readDtTableFromMATLABFile(filename, dt_table, size_dt_table);

	filename = "source.mat";
	Engine *ep;
	//startMatlabEngine();
	if (!(ep = engOpen(""))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		return EXIT_FAILURE;
	}

	startMatlabEngine(ep);

	mxArray *h_matrix = NULL, *flux_east_matrix = NULL, *flux_north_matrix=NULL, *source_matrix = NULL, *open_well = NULL;
	open_well = mxCreateLogicalScalar(true);
	engPutVariable(ep, "open_well", open_well);
	double * h_matlab_matrix;
	h_matlab_matrix = new double[nx*ny];
	h_matrix = mxCreateDoubleMatrix(nx, ny, mxREAL);
	flux_east_matrix = mxCreateDoubleMatrix(nx+2*IC.border,ny+2*IC.border,mxREAL);
	flux_north_matrix = mxCreateDoubleMatrix(nx+2*IC.border,ny+2*IC.border,mxREAL);
	source_matrix = mxCreateDoubleMatrix(nx,ny,mxREAL);

	// Cpu Pointer to store the results
	CpuPtr_2D CheckResults(nx, ny, 0, true);
	CpuPtr_2D zeros(nx, ny, 0, true);
	CpuPtr_2D zerosWithBorder(nx+2*IC.border, ny+2*IC.border, 0, true);

	//Initial Conditions
	IC.dz = dz;
	IC.createnIntervalsTable(H);
	IC.createScalingParameterTable(H);
	IC.createInitialCoarseSatu(H, h);
	IC.computeAllGridBlocks();
	IC.createDtVec();

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

	printf("Cuda error 0.5: %s\n", cudaGetErrorString(cudaGetLastError()));
	initAllocate(&common_args, &coarse_perm_int_args, &coarse_mob_int_args,
			&flux_kernel_args, &time_int_kernel_args, &time_red_kernel_args);

	h.convertToDoublePointer(h_matlab_matrix);
	memcpy((void *)mxGetPr(h_matrix), (void *)h_matlab_matrix, sizeof(double)*nx*ny);
	engPutVariable(ep, "h_matrix", h_matrix);
	printf("Cuda error 1: %s\n", cudaGetErrorString(cudaGetLastError()));

	// Allocate and set data on the GPU
	GpuPtr_3D perm3D_device(nx, ny, nz + 1, 0, perm3D.getPtr());
	GpuPtr_2D Lambda_c_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D Lambda_b_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D dLambda_c_device(nx, ny, 0, zeros.getPtr());
	printf("Cuda error 2: %s\n", cudaGetErrorString(cudaGetLastError()));

	GpuPtr_2D dLambda_b_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D scaling_parameter_C_device(nx, ny, 0, IC.scaling_parameter.getPtr());
	GpuPtr_2D K_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D H_device(nx, ny, 0, H.getPtr());
	GpuPtr_2D h_device(nx, ny, 0, h.getPtr());
	GpuPtr_2D top_surface_device(nx, ny, 0, top_surface.getPtr());
	GpuPtr_2D z_diff_east_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D z_diff_north_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D nInterval_device(nx, ny, 0, IC.nIntervals.getPtr());

	GpuPtr_2D U_x_device(nx, ny, IC.border, flux_east.getPtr());
	GpuPtr_2D U_y_device(nx, ny, IC.border, flux_north.getPtr());
	GpuPtr_2D source_device(nx, ny, 0, source.getPtr());
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
	GpuPtr_1D global_dt_device(3, IC.global_time_data);
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
			U_x_device.getRawPtr(), U_y_device.getRawPtr(), source_device.getRawPtr(),
			h_device.getRawPtr(),top_surface_device.getRawPtr(),
			z_diff_east_device.getRawPtr(), z_diff_north_device.getRawPtr(),
			normal_z_device.getRawPtr(),
			K_face_east_device.getRawPtr(), K_face_north_device.getRawPtr(),
			grav_east_device.getRawPtr(), grav_north_device.getRawPtr(),
			R_device.getRawPtr(), dt_vector_device.getRawPtr(),
			output_test_device.getRawPtr());

	setTimestepReductionKernelArgs(&time_red_kernel_args, TIME_THREADS, IC.nElements, global_dt_device.getRawPtr(),
								   IC.cfl_scale, dt_vector_device.getRawPtr());

	//Compute start volume
	/*
	callCoarseMobIntegrationKernel(new_sparse_grid, IC.block, IC.grid.x, &coarse_mob_int_args);
	callFluxKernel(IC.grid_flux, IC.block_flux, &flux_kernel_args);
	callTimeIntegration(new_sparse_grid, IC.block, IC.grid.x, &time_int_kernel_args);
	*/
	vol_old_device.download(volume.getPtr(), 0, 0, nx, ny);
	float total_volume_old = computeTotalVolume(volume, nx, ny);
	//total_volume_old = stop_inject*0.162037037037037*year;
	float injection_rate = 0.162037037037037;

	t = 0;
	double t2 = 0;
	tf = IC.global_time_data[2];
	int iter_outer_loop = 0;
	int iter_inner_loop = 0;
	int iter_total = 0;
	float time = 0;
	float injected = 0;
	int table_index = 1;
	//total time in years
	float totalTime = 500;
	float temp_time_data[3];
	double time_start = getWallTime();
	double time_start_iter;
	double total_time_gpu = 0;
	if (run){
		while (time < totalTime) {// && iter_total < 91){ // && table_index < size_dt_table){ //iter_total < 400){
			t = 0;
			iter_inner_loop = 0;

			h_device.download(h.getPtr(), 0, 0, nx, ny);
			h.convertToDoublePointer(h_matlab_matrix);
			memcpy((void *)mxGetPr(h_matrix), (void *)h_matlab_matrix, sizeof(double)*nx*ny);
			engPutVariable(ep, "h_matrix", h_matrix);
			if (time >= stop_inject){
				// Total time in years: %.3f time in this round %.3f timestep %.3f\n", totTime, t/(60*60*24), IC.global_time_data[0]/(60*60*24));
				open_well = mxCreateLogicalScalar(false);
				engPutVariable(ep, "open_well", open_well);
				IC.global_time_data[2] = 31536000*2;//157680000*0.5;
				tf = IC.global_time_data[2];
				cudaMemcpy(global_dt_device.getRawPtr(), IC.global_time_data, sizeof(float)*3, cudaMemcpyHostToDevice);
			}
			engEvalString(ep, "[source, east_flux, north_flux] = pressureFunctionToRunfromCpp(h_matrix, variables, open_well);");
			flux_east_matrix = engGetVariable(ep, "east_flux");
			flux_north_matrix = engGetVariable(ep, "north_flux");
			source_matrix = engGetVariable(ep, "source");

			memcpy((void *)flux_east.getPtr(), (void *)mxGetPr(flux_east_matrix), sizeof(float)*(nx+2*IC.border)*(ny+2*IC.border));
			memcpy((void *)flux_north.getPtr(), (void *)mxGetPr(flux_north_matrix), sizeof(float)*(nx+2*IC.border)*(ny+2*IC.border));
			memcpy((void *)source.getPtr(), (void *)mxGetPr(source_matrix), sizeof(float)*nx*ny);

			source_device.upload(source.getPtr(), 0, 0, nx, ny);
			U_x_device.upload(flux_east.getPtr(), 0, 0, nx+2*IC.border, ny+2*IC.border);
			U_y_device.upload(flux_north.getPtr(), 0, 0,nx+2*IC.border, ny+2*IC.border);
			time_start_iter = getWallTime();
			while (t < tf) { //&& iter_total < 91){
				callCoarseMobIntegrationKernel(new_sparse_grid, IC.block, IC.grid.x, &coarse_mob_int_args);

				callFluxKernel(IC.grid_flux, IC.block_flux, &flux_kernel_args);

				callTimestepReductionKernel(TIME_THREADS, &time_red_kernel_args);


				if (iter_outer_loop < 1 && iter_inner_loop == 2){
					IC.global_time_data[0] = 25066810;
					IC.global_time_data[1] += 25066810;
					cudaMemcpy(global_dt_device.getRawPtr(), IC.global_time_data, sizeof(float)*3, cudaMemcpyHostToDevice);
				}


				if (iter_outer_loop < 1 && iter_inner_loop == 1){
					IC.global_time_data[0] = 25413443.94;
					IC.global_time_data[1] += 25413443.94;
					cudaMemcpy(global_dt_device.getRawPtr(), IC.global_time_data, sizeof(float)*3, cudaMemcpyHostToDevice);
				}

				if (iter_outer_loop < 1 && iter_inner_loop == 0){
					IC.global_time_data[0] = 12591745.0588;
					IC.global_time_data[1] += 12591745.0588;
					cudaMemcpy(global_dt_device.getRawPtr(), IC.global_time_data, sizeof(float)*3, cudaMemcpyHostToDevice);
				}


				/*
				IC.global_time_data[0] = (float)dt_table[table_index];//157680000/20;
				IC.global_time_data[1] += (float)dt_table[table_index];//157680000/20;
				cudaMemcpy(global_dt_device.getRawPtr(), IC.global_time_data, sizeof(float)*3, cudaMemcpyHostToDevice);
				 */
				callTimeIntegration(new_sparse_grid, IC.block, IC.grid.x, &time_int_kernel_args);

				cudaMemcpy(IC.global_time_data, global_dt_device.getRawPtr(), sizeof(float)*3, cudaMemcpyDeviceToHost);


				injected += IC.global_time_data[0]*source(50,50);

				t += IC.global_time_data[0];

				//printf("Total time in years: %.3f time in this round %.3f timestep %.3f\n", time, t, IC.global_time_data[0]);
				table_index++;
				iter_inner_loop++;
				iter_total++;
				t2 += (double)IC.global_time_data[0]/year;

			}
			total_time_gpu += getWallTime() - time_start_iter;
			//printf("Finished Iter Total time in years: %.3f time in this round %.3f timestep %.3f\n", totTime, global_time_data[1]/(60*60*24), global_time_data[0]/(60*60*24));
			time += t/(year);
			iter_outer_loop++;
			//engEvalString(ep, "pressureFunctionToRunfromCpp(h_matrix, variables);");
			//printf("%s\n", cudaGetErrorString(cudaGetLastError()));

		}
	}
	printf("Elapsed time program: %.5f gpu part: %.5f", getWallTime() - time_start, total_time_gpu);
    engClose(ep);

	// Run function with timer
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	vol_new_device.download(zeros.getPtr(), 0, 0, nx, ny);
	zeros.printToFile(matlab_file);
    //h_device.download(zeros.getPtr(), 0, 0, nx, ny);
    output_test_device.download(zeros.getPtr(), 0, 0, nx, ny);
	zeros.printToFile(matlab_file_2);
	printf("Load error: %s\n", cudaGetErrorString(cudaGetLastError()));

	vol_new_device.download(zeros.getPtr(), 0, 0, nx, ny);
	float total_volume_new = computeTotalVolume(zeros, nx, ny);

	printf("total volume new %.2f total volume old %.2f frac: %.10f injected %.1f injected fraction %.10f",
			total_volume_new, total_volume_old, (total_volume_new-total_volume_old)/(total_volume_old),
			injected, (total_volume_new-injected)/(injected));

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
	//printf("\n Global timestep manual: %.2f Global timestep from Kernel: %.2f\n", mini, IC.global_time_data[0]);
	printf("Total time: %.3f global_dt from kernel: %.3f\n", time, IC.global_time_data[0]/(60*60*24));

	printf("FINITO precise time %.6f iter_total %i", t2, iter_total);



}


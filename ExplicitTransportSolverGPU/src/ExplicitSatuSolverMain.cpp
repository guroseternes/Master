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
#include "string.h"
#include <memory.h>
#include <algorithm>
#include "cudpp.h"
#include "cudpp_manager.h"
#include "cudpp_plan.h"
#include "cudpp_scan.h"
#include "Config.h"
#include "unistd.h"

#define cudaCheckError() { \
cudaError_t e=cudaGetLastError(); \
if(e!=cudaSuccess) { \
printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
exit(0); \
} \
}

int main() {

	//SPECIFY PARMETERS, SEE CONFIG.H FILE FOR ENUM OPTIONS
	Configurations config;

	//Algorithm for solving for h
	config.algorithm_solve_h = NEWTON_BRUTE_FORCE;
	//Total time in years
	config.total_time = 20;
	// Initial time step in seconds
	config.initial_time_step = 30000;//12592000;
	// Injection stop
	config.injection_time = 100;
	// Pressure update interval in years
	config.pressure_update_injection = 2;
	config.pressure_update_migration = 5;
	// Permeability type
	config.perm_type = PERM_CONSTANT;
	// Name of formation
	config.formation_name = UTSIRA;
	// Parameter beta for the Corey permeability function
	config.beta = 0.4;

	//////////////////////////////////////////////////////////////////////////

	if (config.formation_name == UTSIRA){
		config.formation = "utsira";
		config.formation_dir = "Utsira";
	} else if(config.formation_name == JOHANSEN){
		config.formation = "johansen";
		config.formation_dir = "Johansen";
	} else {
		printf("ERROR: No data for this formation");
	}

	//Initialize some variables
	int nx, ny, nz;
	float dt, dz;
	float t, tf;
	int year = 60*60*24*365;


	// Device properties
	cudaSetDevice(0);
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, device);
	printf("Device name: %s\n", p.name);

	// Set directory path of output and input files
	size_t buff_size= 100;
	char buffer[buff_size];
	const char* path;
	readlink("/proc/self/exe", buffer, buff_size);
	printf("PATH %s", buffer);
	char* output_dir_path = "FullyIntegratedVESimulatorMATLAB/SimulationData/ResultData/";
	char* input_dir_path = "FullyIntegratedVESimulatorMATLAB/SimulationData/FormationData/";

	std::cout << "Trying to open " << input_dir_path << std::endl;
	std::cout << "Output will end up in " << output_dir_path << std::endl;

	// Filename strings
	char dir_input[300];
	char filename_input[300];
	char dir_output[300];
	strcpy(dir_input, input_dir_path);
	strcat(dir_input, config.formation_dir);
	strcat(dir_input, "/");
	strcpy(dir_output, output_dir_path);
	strcat(dir_output, config.formation_dir);
	strcat(dir_output, "/");

	strcpy(filename_input, dir_input);
	strcat (filename_input, "dimensions.mat");

	// Output txt files with results
	FILE* matlab_file_h;
	FILE* matlab_file_coarse_satu;
	FILE* matlab_file_volume;

	// Create output files for coarse saturation, interface height and volume
	// Files are stored in the directory
	createOutputFiles(matlab_file_h, matlab_file_coarse_satu, matlab_file_volume, dir_output);

	readDimensionsFromMATLABFile(filename_input, nx, ny, nz);

	InitialConditions IC(nx, ny, 5);
	printf("nx: %i, ny: %i nz: %i dt: %.10f", nx, ny, nz, dt);

	// Cpu pointers to store formation data from MATLAB
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

	strcpy(filename_input, dir_input);
	strcat (filename_input, "data.mat");

	readFormationDataFromMATLABFile(filename_input, H.getPtr(), top_surface.getPtr(),
			h.getPtr(), normal_z.getPtr(), perm3D.getPtr(), poro3D.getPtr(),
			pv.getPtr(), flux_north.getPtr(), flux_east.getPtr(),
			grav_north.getPtr(), grav_east.getPtr(), K_face_north.getPtr(),
			K_face_east.getPtr(), dz);
	strcpy(filename_input, dir_input);
	strcat (filename_input, "active_cells.mat");
	readActiveCellsFromMATLABFile(filename_input, active_east.getPtr(), active_north.getPtr());

	//readDtTableFromMATLABFile(filename, dt_table, size_dt_table);

	Engine *ep;
	if (!(ep = engOpen(""))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		return EXIT_FAILURE;
	}

	startMatlabEngine(ep, config.formation_dir);

	// Create double precision array for the data exchange between the GPU program and the MATLAB program
	mxArray *h_matrix = NULL, *flux_east_matrix = NULL, *flux_north_matrix=NULL;
	mxArray *source_matrix = NULL, *open_well = NULL;
	open_well = mxCreateLogicalScalar(true);
	engPutVariable(ep, "open_well", open_well);
	double * h_matlab_matrix;
	h_matlab_matrix = new double[nx*ny];
	h_matrix = mxCreateDoubleMatrix(nx, ny, mxREAL);
	flux_east_matrix = mxCreateDoubleMatrix(nx+2*IC.border,ny+2*IC.border,mxREAL);
	flux_north_matrix = mxCreateDoubleMatrix(nx+2*IC.border,ny+2*IC.border,mxREAL);
	source_matrix = mxCreateDoubleMatrix(nx, ny, mxREAL);

	// Cpu Pointer to store the results
	CpuPtr_2D zeros(nx, ny, 0, true);

	//Initial Conditions
	IC.dz = dz;
	IC.createnIntervalsTable(H);
	IC.createScalingParameterTable(H, config.beta);

	IC.createInitialCoarseSatu(H, h);
	IC.computeAllGridBlocks();
	IC.createDtVec();

	// Create mask for sparse grid on GPU
	std::vector<int> active_block_indexes;
	std::vector<int> active_block_indexes_flux;
	int n_active_blocks = 0;
	int n_active_blocks_flux = 0;
	createGridMask(H, IC.grid, IC.block, nx, ny, active_block_indexes,
			n_active_blocks);
	createGridMaskFlux(H, IC.grid_flux, IC.block_flux, nx, ny, active_block_indexes_flux,
			n_active_blocks_flux);

	// Print grid mask properties
	printf("\n nBlocks: %i nActiveBlocks: %i fraction: %.5f\n", IC.grid.x * IC.grid.y,
			n_active_blocks, (float) n_active_blocks / (IC.grid.x * IC.grid.y));
	printf("nBlocks: %i nActiveBlocks: %i fraction: %.5f\n", IC.grid_flux.x * IC.grid_flux.y,
			n_active_blocks_flux, (float) n_active_blocks_flux / (IC.grid_flux.x * IC.grid_flux.y));

	printf("dz: %.3f\n", IC.dz);
	dim3 new_sparse_grid(n_active_blocks, 1, 1);
	dim3 new_sparse_grid_flux(n_active_blocks_flux, 1, 1);

	CommonArgs common_args;
	CoarseMobIntegrationKernelArgs coarse_mob_int_args;
	CoarsePermIntegrationKernelArgs coarse_perm_int_args;
	FluxKernelArgs flux_kernel_args;
	TimeIntegrationKernelArgs time_int_kernel_args;
	TimestepReductionKernelArgs time_red_kernel_args;
	SolveForhProblemCellsKernelArgs solve_problem_cells_args;

	printf("Cuda error 0.5: %s\n", cudaGetErrorString(cudaGetLastError()));
	initAllocate(&common_args, &coarse_perm_int_args, &coarse_mob_int_args,
			&flux_kernel_args, &time_int_kernel_args, &time_red_kernel_args, &solve_problem_cells_args);

	h.convertToDoublePointer(h_matlab_matrix);
	memcpy((void *)mxGetPr(h_matrix), (void *)h_matlab_matrix, sizeof(double)*nx*ny);
	engPutVariable(ep, "h_matrix", h_matrix);
	printf("Cuda error 1: %s\n", cudaGetErrorString(cudaGetLastError()));

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

	GpuPtrInt_1D active_block_indexes_flux_device(n_active_blocks_flux,
			&active_block_indexes_flux[0]);

	cudaCheckError();
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

	setCoarseMobIntegrationKernelArgs(&coarse_mob_int_args,
			Lambda_c_device.getRawPtr(), Lambda_b_device.getRawPtr(),
			dLambda_c_device.getRawPtr(), dLambda_b_device.getRawPtr(),
			h_device.getRawPtr(), perm3D_device.getRawPtr(),
			K_device.getRawPtr(), nInterval_device.getRawPtr(),
			scaling_parameter_C_device.getRawPtr(),
			active_block_indexes_device.getRawPtr(), IC.p_ci, IC.dz, config.perm_type);

	setFluxKernelArgs(&flux_kernel_args,
			Lambda_c_device.getRawPtr(),Lambda_b_device.getRawPtr(),
			dLambda_c_device.getRawPtr(), dLambda_b_device.getRawPtr(),
			U_x_device.getRawPtr(), U_y_device.getRawPtr(), source_device.getRawPtr(),
			h_device.getRawPtr(),top_surface_device.getRawPtr(),
			normal_z_device.getRawPtr(),
			K_face_east_device.getRawPtr(), K_face_north_device.getRawPtr(),
			grav_east_device.getRawPtr(), grav_north_device.getRawPtr(),
			R_device.getRawPtr(), dt_vector_device.getRawPtr(), active_block_indexes_flux_device.getRawPtr(),
			output_test_device.getRawPtr());

	setTimestepReductionKernelArgs(&time_red_kernel_args, TIME_THREADS, IC.nElements, global_dt_device.getRawPtr(),
								   IC.cfl_scale, dt_vector_device.getRawPtr());

	//CUDPP
	CUDPPHandle plan;
	unsigned int* d_isValid;
	int* d_in;
	int* d_out;
	size_t* d_numValid = NULL;
	size_t numValidElements;
	unsigned int numElements=nx*ny;
	cudaCheckError();
	cudaMalloc((void**) &d_isValid, sizeof(unsigned int)*numElements);
	cudaMalloc((void**) &d_in, sizeof(int)*numElements);
	cudaMalloc((void**) &d_out, sizeof(int)*numElements);
	cudaMalloc((void**) &d_numValid, sizeof(size_t));
	cudaCheckError();

	CUDPPHandle theCudpp;
	cudppCreate(&theCudpp);

	setUpCUDPP(theCudpp, plan, nx, ny, d_isValid, d_in, d_out, numElements);
	printf("\nCudaMemcpy error: %s", cudaGetErrorString(cudaGetLastError()));
	cudaCheckError();

	setTimeIntegrationKernelArgs(&time_int_kernel_args, global_dt_device.getRawPtr(),
			IC.integral_res,pv_device.getRawPtr(), h_device.getRawPtr(),
			R_device.getRawPtr(),coarse_satu_device.getRawPtr(),
			scaling_parameter_C_device.getRawPtr(),
			vol_old_device.getRawPtr(), vol_new_device.getRawPtr(),
			d_isValid, d_in);

	cudaCheckError();

	setSolveForhProblemCellsKernelArgs(&solve_problem_cells_args, h_device.getRawPtr(),
									   coarse_satu_device.getRawPtr(), scaling_parameter_C_device.getRawPtr(),
									   d_out, IC.integral_res, d_numValid);
	cudaCheckError();

	//Compute start volume
	callCoarseMobIntegrationKernel(new_sparse_grid, IC.block, IC.grid.x, &coarse_mob_int_args);
	callFluxKernel(new_sparse_grid_flux, IC.block_flux, IC.grid_flux.x, &flux_kernel_args);
	callTimeIntegration(new_sparse_grid, IC.block, IC.grid.x, &time_int_kernel_args);

	cudaCheckError();
	vol_old_device.download(volume.getPtr(), 0, 0, nx, ny);
	float total_volume_old = computeTotalVolume(volume, nx, ny);

	t = 0;
	double t2 = 0;
	tf = IC.global_time_data[2];
	int iter_outer_loop = 0;
	int iter_inner_loop = 0;
	int iter_total = 0;
	int iter_total_lim = 5000;
	float time = 0;
	float injected = 0;
	int table_index = 1;
	double time_start = getWallTime();
	double time_start_iter;
	double total_time_gpu = 0;
	while (time < config.total_time && iter_total < iter_total_lim){
		t = 0;
		iter_inner_loop = 0;

		h_device.download(h.getPtr(), 0, 0, nx, ny);
		h.convertToDoublePointer(h_matlab_matrix);

		memcpy((void *)mxGetPr(h_matrix), (void *)h_matlab_matrix, sizeof(double)*nx*ny);
		engPutVariable(ep, "h_matrix", h_matrix);
		if (time >= config.injection_time){
			open_well = mxCreateLogicalScalar(false);
			engPutVariable(ep, "open_well", open_well);
			IC.global_time_data[2] = 31536000*config.pressure_update_migration;
			tf = IC.global_time_data[2];
			cudaMemcpy(global_dt_device.getRawPtr(), IC.global_time_data, sizeof(float)*3, cudaMemcpyHostToDevice);
		}

		// MATLAB call
		engEvalString(ep, "[source, east_flux, north_flux] = pressureFunctionToRunfromCpp(h_matrix, variables, open_well);");

		// Get variables from MATLABs pressure solver
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
		while (t < tf && iter_total < iter_total_lim){
			cudaCheckError();
			callCoarseMobIntegrationKernel(new_sparse_grid, IC.block, IC.grid.x, &coarse_mob_int_args);
			cudaCheckError();

			callFluxKernel(new_sparse_grid_flux, IC.block_flux, IC.grid_flux.x, &flux_kernel_args);

			cudaCheckError();
			callTimestepReductionKernel(TIME_THREADS, &time_red_kernel_args);
			cudaCheckError();

			// Set the initial time step
			if (iter_total < 1 && iter_inner_loop == 0){
				IC.global_time_data[0] = config.initial_time_step;
				IC.global_time_data[1] += config.initial_time_step;
				cudaMemcpy(global_dt_device.getRawPtr(), IC.global_time_data, sizeof(float)*3, cudaMemcpyHostToDevice);
			}

			//For precomputed time step insertion
			/*
			IC.global_time_data[0] = (float)dt_table[table_index];
			IC.global_time_data[1] += (float)dt_table[table_index];
			cudaMemcpy(global_dt_device.getRawPtr(), IC.global_time_data, sizeof(float)*3, cudaMemcpyHostToDevice);
			*/
			cudaCheckError();

			if (config.algorithm_solve_h == BRUTE_FORCE)
				callTimeIntegration(new_sparse_grid, IC.block, IC.grid.x, &time_int_kernel_args);
			else if (config.algorithm_solve_h == NEWTON_BRUTE_FORCE){
				callTimeIntegrationNewton(new_sparse_grid, IC.block, IC.grid.x, &time_int_kernel_args);
				cudppCompact(plan, d_out, d_numValid, d_in, d_isValid, numElements);
				callSolveForhProblemCells(IC.grid_pc, IC.block_pc, &solve_problem_cells_args);
			} else if (config.algorithm_solve_h == NEWTON_BISECTION){
				callTimeIntegrationNewton(new_sparse_grid, IC.block, IC.grid.x, &time_int_kernel_args);
				cudppCompact(plan, d_out, d_numValid, d_in, d_isValid, numElements);
				callSolveForhProblemCellsBisection(IC.grid_pc, IC.block_pc, &solve_problem_cells_args);
			}
			cudaCheckError();

			cudaMemcpy(IC.global_time_data, global_dt_device.getRawPtr(), sizeof(float)*3, cudaMemcpyDeviceToHost);
			cudaCheckError();

			// Keep track of injected volume, insert injection coordinate and rate
			injected += IC.global_time_data[0]*source(50,50);
			t += IC.global_time_data[0];

			//table_index++;
			iter_inner_loop++;
			iter_total++;

		}
		total_time_gpu += getWallTime() - time_start_iter;
		printf("Total time in years: %.3f time in this round %.3f timestep %.3f GPU time %.3f time per iter %.8f \n", time, t, IC.global_time_data[0], getWallTime() - time_start_iter, (getWallTime() - time_start_iter)/iter_inner_loop);

		time += t/(year);
		iter_outer_loop++;
	}

	printf("Elapsed time program: %.5f gpu part: %.5f", getWallTime() - time_start, total_time_gpu);
    engClose(ep);

	h_device.download(zeros.getPtr(), 0, 0, nx, ny);
	zeros.printToFile(matlab_file_h);
    coarse_satu_device.download(zeros.getPtr(), 0, 0, nx, ny);
    // Divide by the formation thickness H to get the actual coarse satu
	for (int i = 0; i < nx; i++){
		for (int j = 0; j < ny; j++){
			if (H(i,j) != 0)
				zeros(i,j) = zeros(i,j)/H(i,j);
		}
	}
	zeros.printToFile(matlab_file_coarse_satu);
	vol_new_device.download(zeros.getPtr(), 0, 0, nx, ny);
	zeros.printToFile(matlab_file_volume);

	float total_volume_new = computeTotalVolume(zeros, nx, ny);
	printf("total volume new %.2f total volume old %.2f injected %.1f injected fraction %.10f",
			total_volume_new, total_volume_old, injected, (total_volume_new-injected)/(injected));
	printf("volume fraction %.10f",(total_volume_new-total_volume_old)/(total_volume_old));

	printf("\nCudaMemcpy error: %s", cudaGetErrorString(cudaGetLastError()));
	printf("FINITO precise time %.6f iter_total %i", time, iter_total);

}


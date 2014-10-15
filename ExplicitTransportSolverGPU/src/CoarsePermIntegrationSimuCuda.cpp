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
#include <memory.h>

int main() {

	int nx, ny, nz, border;
	border = 1;
	char* filename = "dimensions.mat";
	readDimensionsFromMATLABFile(filename, nx, ny, nz);
	printf("nx: %i, ny: %i nz: %i", nx, ny, nz);
	CpuPtr_2D H(nx,ny,0,true);
	CpuPtr_2D top_surface(nx,ny,0,true);
	CpuPtr_2D h(nx,ny,0,true);
	CpuPtr_2D normal_z(nx,ny,0,true);
	CpuPtr_3D perm3D(nx, ny, nz+1, 0 , true);
	CpuPtr_3D poro3D(nx, ny, nz+1, 0 , true);
	CpuPtr_2D pv(nx,ny,0,true);
	CpuPtr_2D north_flux(nx, ny, border,true);
	CpuPtr_2D east_flux(nx, ny, border,true);
	CpuPtr_2D north_grav(nx, ny, 0, true);
	CpuPtr_2D east_grav(nx, ny, 0,true);
	CpuPtr_2D north_K_face(nx, ny, 0, true);
	CpuPtr_2D east_K_face(nx, ny, 0,true);
	float dz;
	filename = "johansendata.mat";
	readFormationDataFromMATLABFile(filename, H.getPtr(), top_surface.getPtr(), h.getPtr(),
									normal_z.getPtr(), perm3D.getPtr(), poro3D.getPtr(), pv.getPtr(),
									north_flux.getPtr(), east_flux.getPtr(),
									north_grav.getPtr(), east_grav.getPtr(),
									north_K_face.getPtr(), east_K_face.getPtr(), dz);
	//Test reading
	//readPermFromMATLABFile(filename2, perm3D);
	printf("dz: %.3f nz: %i\n", dz, nz);
	printf("H(25,25) %.3f east_grav(50,50): %.3f north_K_face(50,50): %.16f\n", H(25,25), east_grav(50,50), north_K_face(50,50));
	printf("poro3d(45, 50, 1) : %.15f", poro3D(45,50,1));
	printf("perm3d(45, 50, 1) : %.15f", perm3D(45,50,1));

	// Files with results
	FILE* Lambda_integration_file;
	Lambda_integration_file = fopen("Lambda_integration_sparse.txt", "w");
	FILE* height_file;
	height_file = fopen("heights.txt", "w");
	FILE* Check_results_file;
	Check_results_file = fopen("Check_results.txt", "w");


	// Cpu Pointer to store the results
	CpuPtr_2D CheckResults(nx, ny, 0, true);
	CpuPtr_2D zeros(nx, ny, 0, true);
	CpuPtr_2D Lambda(nx, ny, 0, true);

	// Initial Conditions
	InitialConditions IC(nx, ny, 5);
	IC.createReferenceTable();
	IC.dz = dz;

	// GPU
	// Block sizes
	int block_x = 16;
	int block_y = 16;
	// Set block and grid sizes and initialize gpu pointer
	dim3 grid;
	dim3 block;
	computeGridBlock(grid, block, nx, ny, block_x, block_y);

	// Create mask for sparse grid on GPU
	std::vector<int> active_block_indexes;
	int n_active_blocks = 0;
	createGridMask(H, grid, block, nx, ny, active_block_indexes, n_active_blocks);
	printf("nBlocks: %i nActiveBlocks: %i fraction: %.5f\n", grid.x*grid.y, n_active_blocks, (float)n_active_blocks/(grid.x*grid.y));
	printf("dz: %.3f\n", IC.dz);
	dim3 new_sparse_grid(n_active_blocks, 1, 1);

	CpuPtr_2D n_interval_dist(nx, ny, 0, true);
	for (int j = 0; j < ny; j++){
		for (int i = 0; i < nx; i++){
			n_interval_dist(i,j) = ceil(H(i,j)/IC.dz);
		}
	}

	CommonArgs common_args;
	CoarseMobIntegrationKernelArgs coarse_mob_int_args;
	CoarsePermIntegrationKernelArgs coarse_perm_int_args;
	FluxKernelArgs flux_kernel_args;
	TimeIntegrationKernelArgs time_int_kernel_args;
	initAllocate(&common_args, &coarse_perm_int_args, &coarse_mob_int_args, &flux_kernel_args, &time_int_kernel_args);

	// Allocate and set data on the GPU
	GpuPtr_3D perm3D_device(nx, ny, nz+1, 0, perm3D.getPtr());
	GpuPtr_2D Lambda_c_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D Lambda_b_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D K_device(nx, ny, 0, zeros.getPtr());
	GpuPtr_2D H_device(nx, ny, 0, H.getPtr());
	GpuPtr_2D h_device(nx, ny, 0, h.getPtr());
	GpuPtr_2D top_surface_device(nx, ny, 0, top_surface.getPtr());
	GpuPtr_2D nInterval_dist_device(nx, ny, 0, n_interval_dist.getPtr());
	GpuPtr_2D U_x_device(nx, ny, border, east_flux.getPtr());
	GpuPtr_2D U_y_device(nx, ny, border, north_flux.getPtr());
	GpuPtr_1D p_cap_ref_table_device(IC.size_tables, IC.p_cap_ref_table);
	GpuPtr_1D s_c_ref_table_device(IC.size_tables, IC.s_c_ref_table);
	GpuPtrInt_1D active_block_indexes_device(n_active_blocks, &active_block_indexes[0]);

	setCommonArgs(&common_args, IC.delta_rho, IC.g, IC.mu_c, IC.mu_b,
				  H_device.getRawPtr(), p_cap_ref_table_device.getRawPtr(),
				  s_c_ref_table_device.getRawPtr(), nx, ny, border);
	setupGPU(&common_args);
	// Set arguments and run coarse permeability integration kernel
	setCoarsePermIntegrationKernelArgs(&coarse_perm_int_args,
									   K_device.getRawPtr(),
									   perm3D_device.getRawPtr(),
									   nInterval_dist_device.getRawPtr(),
									   IC.dz);
	callCoarsePermIntegrationKernel(grid, block, &coarse_perm_int_args);

	// Set arguments and run coarse mobilty integration kernel
	setCoarseMobIntegrationKernelArgs(&coarse_mob_int_args,
								Lambda_c_device.getRawPtr(),
								h_device.getRawPtr(),
								perm3D_device.getRawPtr(),
								K_device.getRawPtr(),
								nInterval_dist_device.getRawPtr(),
								active_block_indexes_device.getRawPtr(),
								IC.p_ci, IC.dz);

	setFluxKernelArgs(&flux_kernel_args,
					  Lambda_c_device.getRawPtr(), Lambda_b_device.getRawPtr(),
					  U_x_device.getRawPtr(), U_y_device.getRawPtr(),
					  h_device.getRawPtr(),top_surface_device.getRawPtr());

	// Run function with timer
	double time_start = getWallTime();
	//callCoarseMobIntegrationKernel(new_sparse_grid, block, grid.x, &coarse_mob_int_args);
	printf("Elapsed time %.5f", getWallTime()-time_start);

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));

	// Print to file
	/*Lambda_c_device.download(CheckResults.getPtr(), 0, 0, nx, ny);
	printf("Load error: %s\n", cudaGetErrorString(cudaGetLastError()));
	CheckResults.printToFile(Lambda_integration_file);
	H.printToFile(height_file);
	printf("Check results %.3f", CheckResults(6,0));
*/
}


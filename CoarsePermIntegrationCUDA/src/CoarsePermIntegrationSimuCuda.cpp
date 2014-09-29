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

// Max number of intervals
const int N = 100;

int main() {

	char* filename = "test.mat";
	mat_t *matfp;
	matvar_t *matvar;
	matfp = Mat_Open("test.mat",MAT_ACC_RDONLY);
	if ( NULL == matfp ) {
		fprintf(stderr,"Error opening MAT file");
		return EXIT_FAILURE;
	}

	matvar = Mat_VarReadNextInfo(matfp);
	printf("Variable: %s\n",matvar->name);
	Mat_VarReadDataAll(matfp, matvar);
	int nx = matvar->dims[0];
	int ny = matvar->dims[1];
	int nz = 100;
	CpuPtr_2D H(nx, ny, 0, true);
	int size = nx*ny;
	memcpy(H.getPtr(), matvar->data,sizeof(float)*size);
	Mat_VarFree(matvar);
	matvar = NULL;

	matvar = Mat_VarReadNextInfo(matfp);
	printf("Variable: %s\n",matvar->name);
	Mat_VarReadDataAll(matfp, matvar);
	CpuPtr_2D top_surface(nx, ny, 0, true);
	memcpy(top_surface.getPtr(), matvar->data,sizeof(float)*size);
	Mat_VarFree(matvar);
	matvar = NULL;

	printf("%.4f", H(11,15));
	printf("%.4f", top_surface(3,3));

	float max_height = maximum(1032, H.getPtr());

// Files with results
FILE* Lambda_integration_file;
Lambda_integration_file = fopen("Lambda_integration.txt", "w");
FILE* Check_results_file;
Check_results_file = fopen("Check_results.txt", "w");

// Cpu Pointer to store the results
CpuPtr_2D CheckResults(nx, ny, 0, true);
CpuPtr_2D zeros(nx, ny, 0, true);
CpuPtr_2D Lambda(nx, ny, 0, true);

// Initial Conditions
InitialConditions IC(nx, ny, max_height);
//IC.H = H;
IC.createReferenceTable();

// GPU
// Block sizes
int block_x = 16;
int block_y = 16;
// Set block and grid sizes and initialize gpu pointer
dim3 grid;
dim3 block;
computeGridBlock(grid, block, nx, ny, block_x, block_y);

CpuPtr_2D nInterval_dist(nx, ny, 0, true);
for (int j = 0; j < ny; j++){
	for (int i = 0; i < nx; i++){
		nInterval_dist(i,j) = (int) (H(i,j)/IC.dz);
	}
}

// Create a 3D array with permeability data
CpuPtr_3D permeability_dist(nx, ny, nz, 0 , true);
// Fill array with fake data
int H_current_col = 0;
for (int j = 0; j < ny; j++){
	for (int i = 0; i < nx; i++){
		for (int k = 0; k < nInterval_dist(i,j); k++){
			permeability_dist(i,j,k) = 1; //IC.k_data[]
		}
	}
}

// Allocate and set data on the GPU
GpuPtr_3D perm_dist_device(nx, ny, nz, 0, permeability_dist.getPtr());
printf("%s\n", cudaGetErrorString(cudaGetLastError()));
GpuPtr_2D Lambda_device(nx, ny, 0, NULL);
GpuPtr_2D K_device(nx, ny, 0, zeros.getPtr());
GpuPtr_2D H_distribution_device(nx, ny, 0, H.getPtr());
GpuPtr_2D nInterval_dist_device(nx, ny, 0, nInterval_dist.getPtr());
GpuPtr_1D k_data_device(10, IC.k_data);
GpuPtr_1D k_heights_device(10, IC.k_heights);
GpuPtr_1D p_cap_ref_table_device(IC.size_tables, IC.p_cap_ref_table);
GpuPtr_1D s_b_ref_table_device(IC.size_tables, IC.s_b_ref_table);

// Set arguemnts and run coarse permeability integration kernel
CoarsePermIntegrationKernelArgs coarse_perm_int_args;
setCoarsePermIntegrationKernelArgs(&coarse_perm_int_args,
								   K_device.getRawPtr(),
								   H_distribution_device.getRawPtr(),
								   perm_dist_device.getRawPtr(),
								   nInterval_dist_device.getRawPtr(),
								   IC.dz, nx, ny, nz, 0);

// Set arguments and run coarse mobilty integration kernel
CoarseMobIntegrationKernelArgs coarse_mob_int_args;
setCoarseMobIntegrationArgs(&coarse_mob_int_args,
							Lambda_device.getRawPtr(),
							H_distribution_device.getRawPtr(),
							k_data_device.getRawPtr(),
							k_heights_device.getRawPtr(),
							p_cap_ref_table_device.getRawPtr(),
							s_b_ref_table_device.getRawPtr(),
							IC.p_ci, IC.g, IC.delta_rho,
							IC.dz, nx, ny, 0);

initAllocate(&coarse_perm_int_args, &coarse_mob_int_args);

printf("%s\n", cudaGetErrorString(cudaGetLastError()));

callCoarsePermIntegrationKernel(grid, block, &coarse_perm_int_args);

//callCoarsePermIntegrationKernel(grid, block, &coarse_perm_int_args);

printf("%s\n", cudaGetErrorString(cudaGetLastError()));

K_device.download(CheckResults.getPtr(), 0, 0, nx, ny);
printf("%s\n", cudaGetErrorString(cudaGetLastError()));
CheckResults.printToFile(Check_results_file);
H.printToFile(Lambda_integration_file);
printf("Check results %.3f", CheckResults(0,0));
//IC.height_distribution.printToFile(height_distribution_file);
//Lambda_device.download(Lambda.getPtr(),0,0,IC.nx,IC.ny);
//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
//Lambda.printToFile(Lambda_integration_file);

}


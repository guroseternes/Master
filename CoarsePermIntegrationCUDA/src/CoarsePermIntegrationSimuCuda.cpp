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
	int size = nx*ny;
	float H[size];
	memcpy(H, matvar->data,sizeof(float)*size);
	Mat_VarFree(matvar);
	matvar = NULL;

	matvar = Mat_VarReadNextInfo(matfp);
	printf("Variable: %s\n",matvar->name);
	Mat_VarReadDataAll(matfp, matvar);
	float top_surface[size];
	memcpy(top_surface, matvar->data,sizeof(float)*size);
	Mat_VarFree(matvar);
	matvar = NULL;

	printf("%.4f", H[6]);
	printf("%.4f", top_surface[6]);

	float max_height = maximum(1032, H);

// Files with results
FILE* Lambda_integration_file;
Lambda_integration_file = fopen("Lambda_integration.txt", "w");

// Cpu Pointer to stor the results
CpuPtr_2D Lambda(nx, ny, 0, true);

// Initial Conditions
InitialConditions IC(nx, ny, max_height);
IC.H = H;
IC.createReferenceTable();

// GPU
// Block sizes
int block_x = 16;
int block_y = 16;
// Set block and grid sizes and initialize gpu pointer
dim3 grid;
dim3 block;
computeGridBlock(grid, block, IC.nx, IC.ny, block_x, block_y);

// Allocate and set data on the GPU
GpuPtr_2D Lambda_device(IC.nx, IC.ny, 0, NULL);
GpuPtr_2D H_distribution_device(IC.nx, IC.ny, 0, IC.H);
GpuPtr_1D k_data_device(10, IC.k_data);
GpuPtr_1D k_heights_device(10, IC.k_heights);
GpuPtr_1D p_cap_ref_table_device(IC.size_tables, IC.p_cap_ref_table);
GpuPtr_1D s_b_ref_table_device(IC.size_tables, IC.s_b_ref_table);

// Set arguments and run coarse integration kernel
CoarsePermIntegrationKernelArgs coarse_perm_int_args;
setCoarsePermIntegrationArgs(&coarse_perm_int_args,
							Lambda_device.getRawPtr(),
							H_distribution_device.getRawPtr(),
							k_data_device.getRawPtr(),
							k_heights_device.getRawPtr(),
							p_cap_ref_table_device.getRawPtr(),
							s_b_ref_table_device.getRawPtr(),
							IC.p_ci, IC.g, IC.delta_rho,
							IC.dz, IC.nx, IC.ny, 0);

initAllocate(&coarse_perm_int_args);

callCoarsePermIntegrationKernel(grid, block, &coarse_perm_int_args);

printf("%s\n", cudaGetErrorString(cudaGetLastError()));

//IC.height_distribution.printToFile(height_distribution_file);
Lambda_device.download(Lambda.getPtr(),0,0,IC.nx,IC.ny);
printf("%s\n", cudaGetErrorString(cudaGetLastError()));
Lambda.printToFile(Lambda_integration_file);

}


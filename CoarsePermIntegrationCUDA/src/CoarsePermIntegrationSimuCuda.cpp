#include <iostream>
#include <stdio.h>
#include "GpuPtr.h"
#include "CpuPtr.h"
#include "Kernels.h"
#include "Functions.h"
#include "Util.h"
#include "cuda.h"
#include "InitialConditions.h"

// Max number of intervals
const int N = 100;

int main() {

// Files with results
FILE* K_integration_file;
K_integration_file = fopen("K_integration.txt", "w");
FILE* height_distribution_file;
height_distribution_file = fopen("height_distribution.txt", "w");

// Initial Conditions
InitialConditions IC(10, 10, 80);
IC.computeRandomHeights();
IC.createReferenceTable();

// GPU
// Block sizes
int block_x = 16;
int block_y = 16;
// Set block and grid sizes and initialize gpu pointer
dim3 grid;
dim3 block;
computeGridBlock(grid, block, IC.nx, IC.ny, block_x, block_y);
for (int i = 0; i < 10; i++){
	//for (int j = 0; j < 10; j++){
		printf("%.3f ", IC.p_cap_ref_table[i]);
	//}
}

// Allocate and set data on the GPU
GpuPtr_2D Lambda_device(IC.nx, IC.ny, 0, NULL);
GpuPtr_2D height_distribution_device(IC.nx, IC.ny, 0, IC.height_distribution.getPtr());
GpuPtr_1D k_data_device(10, IC.k_data);
GpuPtr_1D k_heights_device(10, IC.k_heights);
GpuPtr_1D p_cap_ref_table_device(IC.size_tables, IC.p_cap_ref_table);
GpuPtr_1D s_b_ref_table_device(IC.size_tables, IC.s_b_ref_table);

// Set arguments and run coarse integration kernel
CoarsePermIntegrationKernelArgs coarse_perm_int_args;
setCoarsePermIntegrationArgs(&coarse_perm_int_args,
							Lambda_device.getRawPtr(),
							height_distribution_device.getRawPtr(),
							k_data_device.getRawPtr(),
							k_heights_device.getRawPtr(),
							p_cap_ref_table_device.getRawPtr(),
							s_b_ref_table_device.getRawPtr(),
							IC.p_ci, IC.g, IC.delta_rho,
							IC.dz, IC.nx, IC.ny, 0);

initAllocate(&coarse_perm_int_args);

callCoarsePermIntegrationKernel(grid, block, &coarse_perm_int_args);

printf("%s\n", cudaGetErrorString(cudaGetLastError()));

IC.height_distribution.printToFile(height_distribution_file);
Lambda_device.download(IC.height_distribution.getPtr(),0,0,IC.nx,IC.ny);
printf("%s\n", cudaGetErrorString(cudaGetLastError()));
printf("hd: %.6f", IC.height_distribution(0,0));
IC.height_distribution.printToFile(K_integration_file);

}


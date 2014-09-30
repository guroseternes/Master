#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "KernelArgStructs.h"
#include "vector_types.h"
#include "Functions.h"
#include "Util.h"
#include <iostream>
#include <stdio.h>

void initAllocate(CoarsePermIntegrationKernelArgs* args1, CoarseMobIntegrationKernelArgs* args2);

void callCoarseMobIntegrationKernel(dim3 grid, dim3 block, CoarseMobIntegrationKernelArgs* args);

void callCoarsePermIntegrationKernel(dim3 grid, dim3 block, CoarsePermIntegrationKernelArgs* args);


#endif /* KERNELS_H_ */

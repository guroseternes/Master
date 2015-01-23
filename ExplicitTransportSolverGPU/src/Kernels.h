#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "KernelArgStructs.h"
#include "DeviceFunctions.h"
#include "vector_types.h"
#include "Functions.h"
#include "Util.h"
#include <iostream>
#include <stdio.h>
#include "float.h"

const int BLOCKDIM_X_FLUX = 16;
const int BLOCKDIM_Y_FLUX = 16;
const int TILEDIM_X = BLOCKDIM_X_FLUX-2;
const int SM_BLOCKDIM_Y = BLOCKDIM_Y_FLUX;
const int TILEDIM_Y = BLOCKDIM_Y_FLUX-2;

const int BLOCKDIM_X = 8;
const int BLOCKDIM_Y = 8;

const int TIME_THREADS = 64;

const int N_CELLS_PER_BLOCK = 4;

const int PROBLEM_CELL_THREADS = 64;

void setupGPU(CommonArgs* args);

void callCoarseMobIntegrationKernel(dim3 grid, dim3 block, int gridDimX, CoarseMobIntegrationKernelArgs* args);

void callCoarsePermIntegrationKernel(dim3 grid, dim3 block, CoarsePermIntegrationKernelArgs* args);

void initAllocate(CommonArgs* args1, CoarsePermIntegrationKernelArgs* args2,
				  CoarseMobIntegrationKernelArgs* args3, FluxKernelArgs* args4,
				  TimeIntegrationKernelArgs* args5, TimestepReductionKernelArgs* args6,
				  SolveForhProblemCellsKernelArgs* args7);

void callFluxKernel(dim3 grid, dim3 block, int gridDimX, FluxKernelArgs* args);

void callTimeIntegration(dim3 grid, dim3 block, int gridDimX, TimeIntegrationKernelArgs* args);

void callSolveForhProblemCells(dim3 grid, dim3 block, SolveForhProblemCellsKernelArgs* args);

void callSolveForhProblemCellsBisection(dim3 grid, dim3 block, SolveForhProblemCellsKernelArgs* args);

void callTimestepReductionKernel(int nThreads, TimestepReductionKernelArgs* args);

#endif /* KERNELS_H_ */

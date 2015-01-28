#ifndef UTIL_H_
#define UTIL_H_

#include "GpuPtr.h"
#include "CpuPtr.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "KernelArgStructs.h"
#include <vector>
#include "math.h"

void computeGridBlock(dim3& grid, dim3& block, int NX, int block_x);
void computeGridBlockBisection(dim3& grid, dim3& block, int NX, int n_cells_per_block);
void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y);
void computeGridBlock(dim3& grid, dim3& block, int NX, int NY, int block_x, int block_y, int tile_x, int tile_y);

void createGridMask(CpuPtr_2D H, dim3 grid, dim3 block, int nx, int ny,
				    std::vector<int> &activeBlockIndexes, int& nActiveBlocks);
void createGridMaskFlux(CpuPtr_2D H, dim3 grid, dim3 block, int nx, int ny,
					std::vector<int> &activeBlockIndexes, int& nActiveBlocks);


void setCommonArgs(CommonArgs* args, float p_ci, float delta_rho, float g, float mu_c, float mu_b,
				   float s_c_res, float s_b_res, float l_e_p_c, float l_e_p_b,
				   GpuRawPtr active_east, GpuRawPtr active_north,
				   GpuRawPtr H, GpuRawPtr pv,
			       unsigned int nx, unsigned int ny, unsigned int border);

void setCoarsePermIntegrationKernelArgs(CoarsePermIntegrationKernelArgs* args, GpuRawPtr K,
										cudaPitchedPtr k, GpuRawPtr nI,
										float dz);

// set arguments for the coarse permeability integration
void setCoarseMobIntegrationKernelArgs(CoarseMobIntegrationKernelArgs* args,
								  GpuRawPtr Lambda_c, GpuRawPtr Lambda_b,
								  GpuRawPtr dLambda_c, GpuRawPtr dLambda_b,
								  GpuRawPtr h, cudaPitchedPtr k,
								  GpuRawPtr K, GpuRawPtr nI, GpuRawPtr scaling_para_C,
								  GpuRawPtrInt a_b_i,
								  float p_ci,float dz, Perm perm_type);

void setFluxKernelArgs(FluxKernelArgs* args,
					   GpuRawPtr Lambda_c, GpuRawPtr Lambda_b,
					   GpuRawPtr dLambda_c, GpuRawPtr dLambda_b,
					   GpuRawPtr U_x, GpuRawPtr U_y, GpuRawPtr source,
					   GpuRawPtr h, GpuRawPtr z,
					   GpuRawPtr normal_z,
					   GpuRawPtr K_face_east, GpuRawPtr K_face_north,
					   GpuRawPtr g_vec_east, GpuRawPtr g_vec_north,
					   GpuRawPtr R, float* dt_vector, GpuRawPtrInt a_b_i,
					   GpuRawPtr test_output);


void setTimeIntegrationKernelArgs(TimeIntegrationKernelArgs* args, float* global_dt, float dz,
								  GpuRawPtr pv, GpuRawPtr h, GpuRawPtr F,
								  GpuRawPtr S_c, GpuRawPtr scaling_para_C,
								  GpuRawPtr vol_old, GpuRawPtr vol_new,
								  unsigned int* d_isValid, int* d_in);

void setTimestepReductionKernelArgs(TimestepReductionKernelArgs* args, int nThreads, int nElements,
									float* global_dt, float cfl_scale, float* dt_vec);

void setSolveForhProblemCellsKernelArgs(SolveForhProblemCellsKernelArgs* args, GpuRawPtr h,
									    GpuRawPtr S_c, GpuRawPtr scaling_parameter_C, int* d_out, float dz, size_t* d_numValid);

float computeBrineSaturation(float p_cap, float C, float s_b_res);

float computeCoarseSaturation(float p_ci, float g, float delta_rho, float s_b_res, float h, float dz, int n,
					    float scaling_parameter_C, float H);
float computeCoarseSaturationSharpInterface(float h, float H);

#endif /* UTIL_H_ */

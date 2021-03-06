#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <math.h>
#include "CpuPtr.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <iostream>
#include "matio.h"
#include "GpuPtr.h"
#include "engine.h"
#include "cudpp.h"
#include "cudpp_manager.h"
#include "cudpp_plan.h"
#include "cudpp_scan.h"
using namespace std;

void createOutputFiles(FILE* &matlab_file_h, FILE* &matlab_file_coarse_satu,
		               FILE* &matlab_file_volume, char* filename_output);
void print_properties();
void startMatlabEngine(Engine* ep, char* formation);
void setUpCUDPP(CUDPPHandle &theCudpp, CUDPPHandle &plan, int nx, int ny, unsigned int* d_isValid, int* d_in, int* d_out, unsigned int& num_elements);
float maximum(int n, float* array);
float computeTotalVolume(CpuPtr_2D vol, int nx, int ny);
void readSourceFromMATLABFile(const char* filename, float* source);
void readDtTableFromMATLABFile(const char* filename, float* dt_table, int& size);
void readZDiffFromMATLABFile(const char* filename, float* east_z_diff, float* north_z_diff);
void printArray(int n, float* array);
double getWallTime();
void readTextFile(const char* filename, CpuPtr_2D& matrix);
void readFormationDataFromMATLABFile(const char* filename, float* H, float* top_surface, float* h,
		float* z_normal, float* perm3D, float* poro3D, float* pv,
		float* north_flux, float* east_flux,
		float* north_grav, float* east_grav,
		float* north_K_face, float* east_K_face , float& dz);
void readFluxesFromMATLABFile(const char* filename, float* east_flux, float* north_flux);
void readHeightAndTopSurfaceFromMATLABFile(const char* filename, CpuPtr_2D& H, CpuPtr_2D& topSurface, int& nx, int& ny);
void readActiveCellsFromMATLABFile(const char* filename, float* active_east, float* active_north);
void readDimensionsFromMATLABFile(const char* filename, int& nx, int& ny, int& nz);
#endif

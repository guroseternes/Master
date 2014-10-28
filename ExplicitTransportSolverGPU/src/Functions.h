#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <math.h>
#include "CpuPtr.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory.h>
#include "matio.h"
using namespace std;

void print_properties();
float maximum(int n, float* array);
void printArray(int n, float* array);
double getWallTime();
void readFormationDataFromMATLABFile(const char* filename, float* H, float* top_surface, float* h,
		float* z_normal, float* perm3D, float* poro3D, float* pv,
		float* north_flux, float* east_flux,
		float* north_grav, float* east_grav,
		float* north_K_face, float* east_K_face , float& dz);
void readHeightAndTopSurfaceFromMATLABFile(const char* filename, CpuPtr_2D& H, CpuPtr_2D& topSurface, int& nx, int& ny);
void readActiveCellsFromMATLABFile(const char* filename, float* active_cells);
void readDimensionsFromMATLABFile(const char* filename, int& nx, int& ny, int& nz);
#endif

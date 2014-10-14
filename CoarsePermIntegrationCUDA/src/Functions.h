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
void readFormationDataFromMATLABFile(const char* filename, CpuPtr_2D& H, CpuPtr_2D& top_surface, CpuPtr_2D& h,
									CpuPtr_2D& normal_z, CpuPtr_3D& perm, float* poro, CpuPtr_2D& pv,
									CpuPtr_2D& north_flux, CpuPtr_2D& east_flux,
									CpuPtr_2D& north_grav, CpuPtr_2D& east_grav,
									float& dz, int&nz, int& nx, int& ny);
void readHeightAndTopSurfaceFromMATLABFile(const char* filename, CpuPtr_2D& H, CpuPtr_2D& topSurface, int& nx, int& ny);
void readPermFromMATLABFile(const char* filename, CpuPtr_3D& perm3D);

#endif

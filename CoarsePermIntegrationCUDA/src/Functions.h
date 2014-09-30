#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <math.h>
#include "CpuPtr.h"
#include "cuda.h"
#include <memory.h>
#include "matio.h"
using namespace std;

float maximum(int n, float* array);
void printArray(int n, float* array);
double getWallTime();
void readHeightAndTopSurfaceFromMATLABFile(const char* filename, CpuPtr_2D& H, CpuPtr_2D& topSurface, int& nx, int& ny);


#endif

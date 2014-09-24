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
void readHeightAndTopSurfaceFromMATLABFile(const char* filename, float* heights, float* topSurface, int& nx, int& ny);

#endif

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <math.h>
#include "CpuPtr.h"
#include "cuda.h"
using namespace std;

void computeRandomHeights(float min, float max, CpuPtr_2D &domain);
void createReferenceTable(float g, float h, float delta_rho, float c_cap, float resolution, float* p_cap_ref_table, float* s_b_ref_table);
void multiply(int n, float* x_values, float* y_values, float* product);
void printArray(int n, float* array);

#endif

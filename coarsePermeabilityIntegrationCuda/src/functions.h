#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <math.h>
using namespace std;

void computeRandomHeights(float min, float max, cpu_ptr_2D domain)
void createReferenceTable(float g, float h, float delta_rho, float c_cap, float resolution, float* p_cap_ref_table, float* s_b_ref_table);
void kDistribution(float dz, float h, float* k_heights, float* k_data, float* k_values);
void computeCapillaryPressure(float p_ci, float g, float delta_rho, float h, float dz, int n, float* p_cap_values);
void inverseCapillaryPressure(int n, float g, float h, float delta_rho, float c_cap, float* p_cap_values, float* s_b_values);
void computeMobility(int n, float* s_b_values, float lambda_end_point, float* mobility_values);
void multiply(int n, float* x_values, float* y_values, float* product);
float trapezoidal(float dz, int n, float* function_values);
void printArray(int n, float* array);

#endif

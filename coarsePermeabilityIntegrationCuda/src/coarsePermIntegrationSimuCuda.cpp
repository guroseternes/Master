#include "gpu_ptr.h"
#include "cpu_ptr.h"
#include "kernels.h"
#include "functions.h"

// Dimensions
// Grid
int nx = 200;
int ny = 200;

//Length of sub intervals
float dz = 1;
// Maximum height of aquifer
float H;

// Height of the capillary interface for the grid
cpu_ptr_2D height_distribution(nx, ny, 0, true);
computeRandomHeights(0, H, height_distribution);

int main() {

// Density difference between brine and CO2
float delta_rho = 500;
// Gravitational acceleration
float g = 9.87;
// Non-dimensional constant that scales the strength of the capillary forces
float c_cap = 1.0/6.0;
// Permeability data (In real simulations this will be a table based on rock data, here we use a random distribution )
float k_data[10] = {0.9352, 1.0444, 0.9947, 0.9305, 0.9682, 1.0215, 0.9383, 1.0477, 0.9486, 1.0835};
float k_heights[10] = {10, 20, 25, 100, 155, 193, 205, 245, 267, 300};

//Inside Kernel
// Converting the permeability data into a table of even subintervals in the z-directions
//float k_values[n+1];
//kDistribution(dz, h, k_heights, k_data, k_values);

// MOBILITY
// The mobility is a function of the saturation, which is directly related to the capillary pressure
// Pressure at capillary interface, which is known
float p_ci = 1;
// Table of capillary pressure values for our subintervals along the z-axis ranging from 0 to h
float resolution = 0.01;
float p_cap_ref_table[(int)(1/resolution+1)];
float s_b_ref_table[(int)(1/resolution+1)];
createReferenceTable(g, H, delta_rho, c_cap, resolution, p_cap_ref_table, s_b_ref_table);

float p_cap_values[n+1];
computeCapillaryPressure(p_ci, g, delta_rho, h, dz, n, p_cap_values);
float s_b_values[n+1];
inverseCapillaryPressure(n, g, h, delta_rho, c_cap, p_cap_values, s_b_values);
printArray(n+1, s_b_values);
// End point mobility lambda'_b, a known quantity
float lambda_end_point = 1;
float lambda_values[n+1];
computeMobility(n, s_b_values, lambda_end_point, lambda_values);

// Multiply permeability values with lambda values
float f_values[n+1];
multiply(n+1, lambda_values, k_values, f_values);

//Numerical integral with trapezoidal
float K = trapezoidal(dz, n, k_values);
float L = trapezoidal(dz, n, f_values)/K;
printf("Value of integral K. %.4f", K);
printf("Value of integral L. %.4f", L);

}


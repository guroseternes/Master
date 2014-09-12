//============================================================================
// Name        : coarsePermeabilityIntegration.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================


#include <iostream>
#include <stdio.h>
#include "functions.h"
using namespace std;

// Dimensions
// Length of subintervals (resolution)
float dz = 1;

// height of the capillary interface
float h = 76;
int n = h/dz;

// Coarse scale variables
float K = 0;
float L = 0;

int main() {

// CAPILLARY PRESSURE - SATURATION
// Creation of p_cap-saturation reference table based on an analytical expression
// (In real simulations this table is based on measurement results)
// Density difference between brine and CO2
float delta_rho = 500;
float g = 9.87;

// Non-dimensional constant that scales the strength of the capillary forces
float c_cap = 1.0/6.0;

// pCap-saturation reference table
float resolution = 0.01;
float p_cap_ref_table[(int)(1/resolution+1)];
float s_b_ref_table[(int)(1/resolution+1)];
createReferenceTable(g, h, delta_rho, c_cap, resolution, p_cap_ref_table, s_b_ref_table);


// PERMEABILITY
// Permeability data (In real simulations this will be a table based on rock data, here we use a random distribution )
float k_data[10] = {0.9352, 1.0444, 0.9947, 0.9305, 0.9682, 1.0215, 0.9383, 1.0477, 0.9486, 1.0835};
float k_heights[10] = {10, 20, 25, 32, 55, 63, 77, 86, 93, 100};

// Converting the permeability data into a table of even subintervals in the z-directions
float k_values[n+1];
kDistribution(dz, h, k_heights, k_data, k_values);

// MOBILITY
// Pressure at capillary interface, which is known
float p_ci = 1;

// Table of capillary pressure values for our subintervals along the z-axis ranging from 0 to h
float p_cap_values[n+1];
computeCapillaryPressure(p_ci, g, delta_rho, h, dz, n, p_cap_values);
float s_b_values[n+1];
inverseCapillaryPressure(n, p_cap_values, p_cap_ref_table, s_b_ref_table, s_b_values);
printArray(n+1, s_b_values);

// End point mobility lambda'_b, a known quantity
float lambda_end_point = 1;
float lambda_values[n+1];
computeMobility(n, s_b_values, lambda_end_point, lambda_values);

float f_values[n+1];
multiply(n+1, lambda_values, k_values, f_values);

float K = trapezoidal(dz, n, k_values);
float L = trapezoidal(dz, n, f_values)/K;
printf("Value of integral K. %.4f", K);
printf("Value of integral L. %.4f", L);

}


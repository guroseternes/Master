//============================================================================
// Name        : coarsePermeabilityIntegration.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================


#include <iostream>
#include "functions.h"
using namespace std;

// Dimensions
// Length of subintervals (resolution)
float dz = 1;
// Height of the capillary interface
float h = 76;
// Number of subintervals
int n = h/dz;

// Coarse scale variables
float K = 0;
float L = 0;

// Creation of p_cap-saturation reference table based on an analytical expression
// (In real simulations this table is based on measurement results)
// Density difference between brine and CO2
float delta_rho = 500;
float g = 9.87;
// Non-dimensional constant that scales the strength of the capillary forces
float c_cap = 1.0/6.0;
// pCap-saturation reference table
float p_cap_ref_table[n+1];
float s_b_ref_table[n+1];
float table_resolution = 0.01;
createReferenceTable(g, h, delta_rho, table_resolution, p_cap_ref_table, s_b_ref_table);


// Permeability
// Permeability data (In real simulations this will be a table based on rock data, here we use a random distribution )
float k_data[10] = {0.9352 1.0444 0.9947 0.9305 0.9682 1.0215 0.9383 1.0477 0.9486 1.0835};





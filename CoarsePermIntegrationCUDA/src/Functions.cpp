#include "Functions.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>

float maximum(int n, float* array){
	float max = 0;
	for (int i = 0; i < n; i++){
		if (array[i] > max){
			max = array[i];
		}
	}
	return max;
}

void printArray(int n, float* array){
	for (int i = 0; i < n; i++){
		printf("%.3f ", array[i]);
				if (i % 10 == 0){
					printf("\n");
				}
	}
}

void readHeightAndTopSurfaceFromMATLABFile(const char* filename, float* heights, float* topSurface, int& nx, int& ny){

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}

matvar = Mat_VarReadNextInfo(matfp);
printf("Variable: %s\n",matvar->name);
Mat_VarReadDataAll(matfp, matvar);
nx = matvar->dims[0];
ny = matvar->dims[1];
int size = nx*ny;
heights = new float[size];
memcpy(heights, matvar->data,sizeof(float)*size);
//Mat_VarFree(matvar);
//matvar = NULL;
printf("%.4f", heights[7]);

topSurface = new float[3];

/*
matvar = Mat_VarReadNextInfo(matfp);
printf("Variable: %s\n",matvar->name);
printf("Halla");
Mat_VarReadDataAll(matfp, matvar);
topSurface = new float[size];
memcpy(topSurface, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;
//printf("lazla\n%.4f", ts[9]);
*/
}

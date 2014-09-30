#include "Functions.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <sys/time.h>
#include "CpuPtr.h"

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

double getWallTime(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void readHeightAndTopSurfaceFromMATLABFile(const char* filename, CpuPtr_2D& H, CpuPtr_2D& top_surface, int& nx, int& ny){

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
H = CpuPtr_2D(nx, ny, 0, true);
memcpy(H.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;
printf("%.4f", H(42,0));

matvar = Mat_VarReadNextInfo(matfp);
printf("Variable: %s\n",matvar->name);
printf("Halla");
Mat_VarReadDataAll(matfp, matvar);
top_surface = CpuPtr_2D(nx, ny, 0, true);
memcpy(top_surface.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;
printf("%.4f\n", top_surface(42,0));
}


#include "Functions.h"
#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <sys/time.h>
#include "CpuPtr.h"

// Print GPU properties
void print_properties(){
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
         printf("Device count: %d\n", deviceCount);

        cudaDeviceProp p;
        cudaSetDevice(0);
        cudaGetDeviceProperties (&p, 0);

        printf("Name: %s\n" , p.name);
        printf("Compute capability: %d.%d\n", p.major, p.minor);
        printf("Compute concurrency %i\n", p.concurrentKernels);
        printf("\n\n");
}

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

void readActiveCellsFromMATLABFile(const char* filename, float* active_cells) {

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}
int nx, ny, nz;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nx = matvar->dims[0];
ny = matvar->dims[1];
int size = nx*ny;
memcpy(active_cells, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;
}

void readFormationDataFromMATLABFile(const char* filename, float* H, float* top_surface, float* h,
									float* z_normal, float* perm3D, float* poro3D, float* pv,
									float* north_flux, float* east_flux,
									float* north_grav, float* east_grav,
									float* north_K_face, float* east_K_face , float& dz){

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}
int nx, ny, nz;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nx = matvar->dims[0];
ny = matvar->dims[1];
int size = nx*ny;
memcpy(H, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(h, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(top_surface, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(pv, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
int nx_flux = matvar->dims[0];
int ny_flux = matvar->dims[1];
//printf("Variable %s nx: %i ny %i")
memcpy(east_flux, matvar->data,sizeof(float)*(nx+2)*(ny+2));
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(north_flux, matvar->data,sizeof(float)*(nx+2)*(ny+2));
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(east_grav, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(north_grav, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(north_K_face, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(east_K_face, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(z_normal, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

//perm3D = CpuPtr_3D(nx, ny, nz, 0, true);
matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nz = matvar->dims[0];
ny = matvar->dims[1];
nx = matvar->dims[2];
printf("3D dims: nx %i ny %i nz %i\n", nx, ny, nz);
memcpy(perm3D, matvar->data,sizeof(float)*nx*ny*nz);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
memcpy(poro3D, matvar->data,sizeof(float)*nx*ny*nz);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
dz = *(float*)matvar->data;
Mat_VarFree(matvar);
matvar = NULL;
}

void readDimensionsFromMATLABFile(const char* filename, int& nx, int& ny, int& nz){

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nz = *(float*)matvar->data;
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nx = *(float*)matvar->data;
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
ny = *(float*)matvar->data;
Mat_VarFree(matvar);
matvar = NULL;
}

void readHeightAndTopSurfaceFromMATLABFile(const char* filename, CpuPtr_2D& H, CpuPtr_2D& top_surface, int& nx, int& ny){

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nx = matvar->dims[0];
ny = matvar->dims[1];
int size = nx*ny;
H = CpuPtr_2D(nx, ny, 0, true);
memcpy(H.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
top_surface = CpuPtr_2D(nx, ny, 0, true);
memcpy(top_surface.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;
}


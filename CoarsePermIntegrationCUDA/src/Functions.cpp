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
        printf("Compute capability: %d.%d\n", p.major, p.minor);
        printf("Name: %s\n" , p.name);
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

void readFormationDataFromMATLABFile(const char* filename, CpuPtr_2D& H, CpuPtr_2D& top_surface, CpuPtr_2D& h,
									CpuPtr_2D& z_normal, CpuPtr_3D& perm3D, float* poro3D, CpuPtr_2D& pv,
									CpuPtr_2D& north_flux, CpuPtr_2D& east_flux,
									CpuPtr_2D& north_grav, CpuPtr_2D& east_grav,
									float& dz, int&nz, int& nx, int& ny){

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
nz = *(float*)matvar->data;
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
nx = matvar->dims[0];
ny = matvar->dims[1];
int size = nx*ny;
H = CpuPtr_2D(nx, ny, 0, true);
memcpy(H.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
h = CpuPtr_2D(nx, ny, 0, true);
memcpy(h.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
top_surface = CpuPtr_2D(nx, ny, 0, true);
memcpy(top_surface.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
pv = CpuPtr_2D(nx, ny, 0, true);
memcpy(pv.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
east_flux = CpuPtr_2D(nx, ny, 1, true);
memcpy(east_flux.getPtr(), matvar->data,sizeof(float)*(nx+1)*(ny+1));
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
north_flux = CpuPtr_2D(nx, ny, 1, true);
memcpy(north_flux.getPtr(), matvar->data,sizeof(float)*(nx+1)*(ny+1));
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
z_normal = CpuPtr_2D(nx, ny, 0, true);
memcpy(z_normal.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

//perm3D = CpuPtr_3D(nx, ny, nz, 0, true);
matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
nx = matvar->dims[0];
ny = matvar->dims[1];
nz = matvar->dims[2];
printf("3D dims: nx %i ny %i nz %i\n", nx, ny, nz);
//memcpy(perm3D.getPtr(), matvar->data,sizeof(float)*nx*ny*nz);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
//poro3D = CpuPtr_3D(nx, ny, nz, 0, true);
memcpy(poro3D, matvar->data,sizeof(float)*nx*ny*nz);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
dz = *(float*)matvar->data;
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
north_grav = CpuPtr_2D(nx, ny, 0, true);
memcpy(north_grav.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("Variable: %s\n",matvar->name);
east_grav = CpuPtr_2D(nx, ny, 0, true);
memcpy(east_grav.getPtr(), matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

}
void readPermFromMATLABFile(const char* filename, CpuPtr_3D& perm3D){

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
int nx = matvar->dims[0];
int ny = matvar->dims[1];
int nz = matvar->dims[2];
nz = 20;
int size = nx*ny*nz;
perm3D = CpuPtr_3D(nx, ny, nz, 0, true);
memcpy(perm3D.getPtr(), matvar->data,sizeof(float)*size);
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


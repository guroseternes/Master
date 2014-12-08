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

void startMatlabEngine(Engine* ep){
	engEvalString(ep, "cd ~/mrst-bitbucket/mrst-core;");
	engEvalString(ep, "startup;");
	engEvalString(ep, "startup_user;");
	engEvalString(ep, "cd ~/mrst-bitbucket/mrst-other/co2lab;");
	engEvalString(ep, "startuplocal");
	engEvalString(ep, "cd ~/mrst-bitbucket/mrst-other/co2lab/guro_code;");
	engEvalString(ep, "variables = loadDataForCpp;");
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

float computeTotalVolume(CpuPtr_2D vol, int nx, int ny){
	//CpuPtr_2D vol(nx,ny,0,true);
	//vol_device.download(vol.getPtr(), 0, 0, nx, ny);
	float tot_vol = 0;
	for (int j=0; j<ny; j++){
		for (int i=0; i<nx; i++){
			if (!isnan(vol(i,j)))
				tot_vol += vol(i,j);
		}
	}
	return tot_vol;
}

double getWallTime(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void readTextFile(const char* filename, CpuPtr_2D& matrix) {
	FILE * pFile;
	int i, j;
	double value;
	pFile = fopen (filename , "r");
	if (pFile == NULL) perror ("Error opening file");

	else {
		while(fscanf (pFile, "%i %i %lf\n", &i, &j, &value) != EOF ){
			if (i==51 && j == 52)
				printf("Copy from MATLAB file %.15f", value);
			matrix(i,j) = value;
		}
	}

}

void readDtTableFromMATLABFile(const char* filename, float* dt_table, int& size) {

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}
int nx;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
printf("name %s size %i", matvar->name, matvar->dims[0]);
nx = matvar->dims[0];
size = nx;
memcpy(dt_table, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

}

void readFluxesFromMATLABFile(const char* filename, float* east_flux, float* north_flux) {

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
memcpy(east_flux, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nx = matvar->dims[0];
ny = matvar->dims[1];
size = nx*ny;
memcpy(north_flux, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;
}

void readZDiffFromMATLABFile(const char* filename, float* east_z_diff, float* north_z_diff) {

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
memcpy(east_z_diff, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nx = matvar->dims[0];
ny = matvar->dims[1];
size = nx*ny;
memcpy(north_z_diff, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;
}

void readSourceFromMATLABFile(const char* filename, float* source) {

mat_t *matfp;
matvar_t *matvar;
matfp = Mat_Open(filename, MAT_ACC_RDONLY);
if ( NULL == matfp ) {
	fprintf(stderr,"Error opening MAT file");
}
int nx, ny;
matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nx = matvar->dims[0];
ny = matvar->dims[1];
int size = nx*ny;
memcpy(source, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;
}


void readActiveCellsFromMATLABFile(const char* filename, float* active_east, float* active_north) {

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
memcpy(active_east, matvar->data,sizeof(float)*size);
Mat_VarFree(matvar);
matvar = NULL;

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nx = matvar->dims[0];
ny = matvar->dims[1];
memcpy(active_north, matvar->data,sizeof(float)*size);
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

matvar = Mat_VarReadNextInfo(matfp);
Mat_VarReadDataAll(matfp, matvar);
nz = matvar->dims[0];
ny = matvar->dims[1];
nx = matvar->dims[2];
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


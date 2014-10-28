#include <cassert>
#include <iostream>
#include <vector>
#include <cuda.h>
#include "GpuPtr.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

GpuPtr_3D::GpuPtr_3D(unsigned int nx, unsigned int ny, unsigned int nz, int border, float* cpu_ptr) {
/*	data_width = width + 2*border;
	data_height = height + 2*border;
	data_depth = depth;
*/	cudaExtent extent;
	extent.depth = nx;
	extent.height = ny;
	extent.width = nz*sizeof(float);
	data_border = border;
	data.ptr = 0;
	data.pitch = 0;
	cudaMalloc3D(&data, extent);
	printf("Malloc 3D %s\n", cudaGetErrorString(cudaGetLastError()));
	if (cpu_ptr != NULL) upload(cpu_ptr, nx, ny, nz);
}

void GpuPtr_3D::upload(float* cpu_ptr, unsigned int nx, unsigned int ny, unsigned int nz){
	/*width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;
	depth = (depth == 0) ? data_depth : depth;
	*/
	cudaMemcpy3DParms params = {0};
	cudaPitchedPtr srcPtr; // = make_cudaPitchedPtr(cpu_ptr, width*sizeof(float), width, height);
	srcPtr.ptr = cpu_ptr;
	srcPtr.pitch = nz*sizeof(float);
	srcPtr.xsize = nz;
	srcPtr.ysize = ny;
	params.srcPtr = srcPtr;
	params.dstPtr = data;

	cudaExtent extent;
	extent.width = nz*sizeof(float);
	extent.height = ny;
	extent.depth = nx;
	params.extent = extent;
	params.kind = cudaMemcpyHostToDevice;

	printf("nz in upload %i\n", nz);
	cudaMemcpy3D(&params);
	printf("Error upload %s\n", cudaGetErrorString(cudaGetLastError()));
}


GpuPtr_3D::~GpuPtr_3D() {
	cudaFree(data.ptr);
}

void GpuPtr_3D::download(float* cpu_ptr, unsigned int nx, unsigned int ny, unsigned int nz){
	/*width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;
	depth = (depth == 0) ? data_depth : depth;
*/
	cudaMemcpy3DParms params;
	params.srcPtr = data;
	cudaPitchedPtr dstPtr;
	dstPtr.ptr = cpu_ptr;
	dstPtr.pitch = nz*sizeof(float);
	dstPtr.xsize = nz;
	dstPtr.ysize = ny;
	params.dstPtr = dstPtr;
	cudaExtent extent;
	extent.depth = nx;
	extent.height = ny;
	extent.width = nz*sizeof(float);
	params.extent = extent;
	params.kind = cudaMemcpyDeviceToHost;

	cudaMemcpy3D(&params);
	printf("Error %s\n", cudaGetErrorString(cudaGetLastError()));

}

GpuPtr_2D::GpuPtr_2D(unsigned int width, unsigned int height, int border, float* cpu_ptr) {
	data_width = width + 2*border;
	data_height = height + 2*border;
	data_border = border;
	data.ptr = 0;
	data.pitch = 0;

	cudaMallocPitch((void**) &data.ptr, &data.pitch, data_width*sizeof(float), data_height);
	if (cpu_ptr != NULL)
		upload(cpu_ptr);
}

GpuPtr_2D::~GpuPtr_2D() {
	cudaFree(data.ptr);
}

void GpuPtr_2D::copy(const GpuPtr_2D& other, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height, int border){
//	width = (width == 0) ? data_width : width;
//	height = (height == 0) ? data_height : height;

	data_width = width + 2*border;
	data_height = height + 2*border;
	data_border = border;

	size_t pitch1 = data.pitch;
	float* ptr1 = (float*) ((char*) data.ptr+y_offset*pitch1) + x_offset;

	size_t pitch2 = other.getRawPtr().pitch;
	float* ptr2 = (float*) ((char*) other.getRawPtr().ptr+y_offset*pitch2) + x_offset;

	assert(data_width == other.getWidth() && data_height == other.getHeight() && ptr1 != ptr2);

	cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2, width*sizeof(float), height, cudaMemcpyDeviceToDevice);
}

void GpuPtr_2D::download(float* cpu_ptr, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height){
	width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;

	size_t pitch1 = width*sizeof(float);
	float* ptr1 = cpu_ptr;

	size_t pitch2 = data.pitch;
	float* ptr2 = (float*) ((char*) data.ptr+y_offset*pitch2) + x_offset;

	cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2, width*sizeof(float), height, cudaMemcpyDeviceToHost);
}

void GpuPtr_2D::upload(const float* cpu_ptr, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height){
	width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;

	size_t pitch1 = data.pitch;
	float* ptr1 = (float*) ((char*) data.ptr+y_offset*pitch1) + x_offset;

	size_t pitch2 = width*sizeof(float);
	const float* ptr2 = cpu_ptr;

	cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2, width*sizeof(float), height, cudaMemcpyHostToDevice);
}

void GpuPtr_2D::set(int value, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height, int border){
	width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;

	data_width = width + 2*border;
	data_height = height + 2*border;

	data_border = border;

	size_t pitch = data.pitch;
	float* ptr = (float*) ((char*) data.ptr+y_offset*pitch) + x_offset;

	cudaMemset2D(ptr, pitch, value, width*sizeof(float), height);
}

void GpuPtr_2D::set(float value, unsigned int x_offset, unsigned int y_offset, unsigned int width, unsigned int height){
	width = (width == 0) ? data_width :width;
	height = (height == 0) ? data_height : height;

	std::vector<float> tmp(width*height, value);

	size_t pitch1 = data.pitch;
	float* ptr1 = (float*) ((char*) data.ptr+y_offset*pitch1) + x_offset;

	size_t pitch2 = width*sizeof(float);
	const float* ptr2 = &tmp[0];

	cudaMemcpy2D(ptr1, pitch1, ptr2, pitch2, width*sizeof(float), height, cudaMemcpyHostToDevice);
}

void GpuPtr_1D::upload(const float* cpu_ptr, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	float* ptr1 = data.ptr + x_offset;
	const float* ptr2 = cpu_ptr;

	cudaMemcpy(ptr1, ptr2, width*sizeof(float), cudaMemcpyHostToDevice);
}

GpuPtr_1D::GpuPtr_1D(unsigned int width, float* cpu_ptr) {
	data_width = width;
	data.ptr = 0;
	cudaMalloc((void**) &data.ptr, data_width*sizeof(float));
	if (cpu_ptr != NULL) upload(cpu_ptr);
}

GpuPtr_1D::~GpuPtr_1D() {
	cudaFree(data.ptr);
}

void GpuPtr_1D::set(float value, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	float* ptr = data.ptr + x_offset;
	std::vector<float> tmp(width, value);

	cudaMemcpy(ptr, &tmp[0], width*sizeof(float), cudaMemcpyHostToDevice);
}

GpuPtrInt_1D::GpuPtrInt_1D(unsigned int width, int* cpu_ptr) {
	data_width = width;
	data.ptr = 0;
	cudaMalloc((void**) &data.ptr, data_width*sizeof(int));
	if (cpu_ptr != NULL) upload(cpu_ptr);
}

GpuPtrInt_1D::~GpuPtrInt_1D() {
	cudaFree(data.ptr);
}

void GpuPtrInt_1D::upload(const int* cpu_ptr, unsigned int x_offset, unsigned int width) {
	width = (width == 0) ? data_width : width;

	int* ptr1 = data.ptr + x_offset;
	const int* ptr2 = cpu_ptr;

	cudaMemcpy(ptr1, ptr2, width*sizeof(int), cudaMemcpyHostToDevice);
}

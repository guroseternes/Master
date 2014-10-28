#ifndef GPU_PTR_H_
#define GPU_PTR_H_

#include <cassert>
#include <iostream>
#include <vector>
#include <cuda.h>
#include "GpuPtr.h"
#include <cuda_runtime_api.h>
#include <cstddef>

/**
 * Very simple class that suits the GPU fine for accessing memory
 */
class GpuRawPtr {
public:
	float* ptr;   //!< pointer to allocated memory
	size_t pitch; //!< Pitch in bytes of allocated m
};

class GpuRawPtrInt {
public:
	int* ptr;   //!< pointer to allocated memory
	size_t pitch; //!< Pitch in bytes of allocated m
};

class GpuRawPtr_3D {
public:
	cudaPitchedPtr data;   //!< pointer to allocated memory
//!< Pitch in bytes of allocated m
};

class GpuPtr_3D{
public:
	// Allocating data on the GPU
	GpuPtr_3D(unsigned int width, unsigned int height, unsigned int depth, int border = 0, float* cpu_ptr=NULL);

	//Deallocates the data
	~GpuPtr_3D();

	const cudaPitchedPtr& getRawPtr() const {
		return data;
	}
	const unsigned int& getWidth() const {
		return data_width;
	}
	const unsigned int& getHeight() const {
        return data_height;
    }
    const int& getBorder() const {
        return data_border;
    }

	void download(float* cpu_ptr, unsigned int width, unsigned int height, unsigned int depth);

	// Perform CPU to GPU copy of a witdh x height domain starting at x_offset and y_offset on the cpu_ptr.
	void upload(float* cpu_ptr, unsigned int x_offset=0, unsigned int y_offset=0, unsigned int z_offset=0);

private:
	cudaPitchedPtr data;
	unsigned int data_width;
	unsigned int  data_height;
	unsigned int data_depth;
	int data_border;
};

class GpuPtr_2D {
public:
	// Allocating data on the GPU
	GpuPtr_2D(unsigned int width, unsigned int height, int border = 0, float* cpu_ptr=NULL);

	//Deallocates the data
	~GpuPtr_2D();

	const GpuRawPtr& getRawPtr() const {
		return data;
	}
	const unsigned int& getWidth() const {
		return data_width;
	}
	const unsigned int& getHeight() const {
        return data_height;
    }
    const int& getBorder() const {
        return data_border;
    }

	// Performs GPU-GPU copy of a width x height domain starting at x_offset, y_offset from a different GpuPtr
	void copy(const GpuPtr_2D& other,
			unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0, int border=0);

	void download(float* cpu_ptr,
			unsigned int x_offset, unsigned int y_offset,
			unsigned int width, unsigned int height);

	// Perform CPU to GPU copy of a witdh x height domain starting at x_offset and y_offset on the cpu_ptr.
	void upload(const float* cpu_ptr, unsigned int x_offset=0, unsigned int y_offset=0,
    		unsigned int width=0, unsigned int height=0);


	// Performs GPU "memset" of a width x height domain starting at x-offset, y_offset. Only really useful for setting all bits to 0 or 1.
	void set(int value, unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0, int border=0);

	// Same as above, only with float.
	void set(float value, unsigned int x_offset=0, unsigned int y_offset=0,
			unsigned int width=0, unsigned int height=0);

	private:
		GpuRawPtr data;
		size_t pitch;
		unsigned int data_width;
		unsigned int  data_height;
		int data_border;
};

class GpuPtr_1D{
public:
	GpuPtr_1D(unsigned int width, float* cpu_ptr=NULL);
	~GpuPtr_1D();
	//void download(float* cpu_ptr, unsigned int x_offset=0, unsigned int width=0);
	void upload(const float* cpu_ptr, unsigned int x_offset=0, unsigned int width=0);
	//void set(int value, unsigned int x_offset=0, unsigned int width=0);
	void set(float value, unsigned int x_offset=0, unsigned int width=0);
	float* getRawPtr() const {
		return data.ptr;
	}
	const unsigned int& getWidth() const {
		return data_width;
	}

private:
	GpuRawPtr data;
	unsigned int data_width;
};

class GpuPtrInt_1D{
public:
	GpuPtrInt_1D(unsigned int width, int* cpu_ptr=NULL);
	~GpuPtrInt_1D();
	void upload(const int* cpu_ptr, unsigned int x_offset=0, unsigned int width=0);
	GpuRawPtrInt getRawPtr() const {
		return data;
	}
private:
	GpuRawPtrInt data;
	unsigned int data_width;
};


#endif /* GPU_PTR_H_ */

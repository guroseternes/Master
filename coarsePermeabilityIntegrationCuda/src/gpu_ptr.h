#ifndef GPU_PTR_H_
#define GPU_PTR_H_

/**
 * Very simple class that suits the GPU fine for accessing memory
 */
class gpu_raw_ptr {
public:
	float* ptr;   //!< pointer to allocated memory
	size_t pitch; //!< Pitch in bytes of allocated m
};

class gpu_ptr_2D {
public:
	// Allocating data on the GPU
	gpu_ptr_2D(unsigned int width, unsigned int height, int border = 0, float* cpu_ptr=NULL);

	//Deallocates the data
	~gpu_ptr_2D();

	const gpu_raw_ptr& getRawPtr() const {
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
};




#endif /* GPU_PTR_H_ */

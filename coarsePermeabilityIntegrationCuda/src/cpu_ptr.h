#ifndef CPU_PTR_H_
#define CPU_PTR_H_

class cpu_ptr_2D{
public:
	// Regular constructor
	cpu_ptr_2D(unsigned int nx, unsigned int ny, unsigned int border, bool setToZero = false);

	// Copy constructor
	cpu_ptr_2D(const cpu_ptr_2D& other);

	// Deconstructor
	~cpu_ptr_2D();

	float xmin, xmax, ymin, ymax;

	int get_nx();
	int get_ny();

	float get_dx();
	float get_dy();

	float* get_ptr(){return data;};

	void set_time(float time);

	cpu_ptr_2D& operator=(const cpu_ptr_2D& rhs);

	// Access elements
	float &operator()(unsigned int i, unsigned int j);

	void printToFile(FILE* filePtr, bool withHeader = false, bool withBorder = false);

private:
	unsigned int nx, ny, border, NX, NY;
	float time;
	float *data;
 	void allocateMemory();
};


#endif /* CPU_PTR_H_ */

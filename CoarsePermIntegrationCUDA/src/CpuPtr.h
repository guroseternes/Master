#ifndef CPU_PTR_H_
#define CPU_PTR_H_

#include <iostream>
#include <stdio.h>

class CpuPtr_2D{
public:
	// Regular constructor
	CpuPtr_2D(unsigned int nx, unsigned int ny, unsigned int border, bool setToZero);

	// Copy constructor
	CpuPtr_2D(const CpuPtr_2D& other);

	// Deconstructor
	~CpuPtr_2D();

	float xmin, xmax, ymin, ymax;

	int get_nx();
	int get_ny();

	float get_dx();
	float get_dy();

	float* getPtr(){return data;};

	void setTime(float time);

	CpuPtr_2D& operator=(const CpuPtr_2D& rhs);

	// Access elements
	float &operator()(unsigned int i, unsigned int j);

	void printToFile(FILE* filePtr); //, bool withHeader = false, bool withBorder = false);

	private:
		unsigned int nx, ny, border, NX, NY;
		float time;
		float *data;
		void allocateMemory();
};


#endif /* CPU_PTR_H_ */

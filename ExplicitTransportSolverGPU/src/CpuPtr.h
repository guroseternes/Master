#ifndef CPU_PTR_H_
#define CPU_PTR_H_

#include <iostream>
#include <stdio.h>

class CpuPtr_3D{
public:
	// Trivial Constructor
	CpuPtr_3D();

	// Regular constructor
	CpuPtr_3D(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int border, bool setToZero);

	// Deconstructor
	~CpuPtr_3D();

	float xmin, xmax, ymin, ymax; // zmin, zmax;

	int getNx();
	int getNy();
	int getNz();

	float getDx();
	float getDy();
	float getDz();

	float* getPtr(){return data;};

	void setTime(float time);

	// Access elements
	float &operator()(unsigned int i, unsigned int j, unsigned int k);

	void printToFile(FILE* filePtr); //, bool withHeader = false, bool withBorder = false);

	private:
		unsigned int nx, ny, nz, border, NX, NY, NZ;
		float time;
		float *data;
		void allocateMemory();
};


class CpuPtr_2D{
public:
	// Trivial Constructor
	CpuPtr_2D();

	// Regular constructor
	CpuPtr_2D(unsigned int nx, unsigned int ny, unsigned int border, bool setToZero);

	// Copy constructor
	CpuPtr_2D(const CpuPtr_2D& other);

	// Deconstructor
	~CpuPtr_2D();

	float xmin, xmax, ymin, ymax;

	int getNx();
	int getNy();

	float getDx();
	float getDy();

	float* getPtr(){return data;};

	void setTime(float time);

	CpuPtr_2D& operator=(const CpuPtr_2D& rhs);

	// Access elements
	float &operator()(unsigned int i, unsigned int j);

	void printToFile(FILE* filePtr); //, bool withHeader = false, bool withBorder = false);

	void printToFileComparison(FILE* filePtr, CpuPtr_2D other);

	private:
		unsigned int nx, ny, border, NX, NY;
		float time;
		float *data;
		void allocateMemory();
};


#endif /* CPU_PTR_H_ */

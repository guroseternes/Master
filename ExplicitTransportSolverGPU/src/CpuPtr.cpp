#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <vector>
#include "CpuPtr.h"

CpuPtr_3D::CpuPtr_3D(unsigned int nx, unsigned int ny, unsigned int nz, unsigned int border, bool setToZero)
: nx(nx), ny(ny), nz(nz), border(border), NX(nx+2*border), NY(ny+2*border), NZ(nz), xmin(0), ymin(0), xmax(1.0), ymax(1.0),time(0){
	allocateMemory();
	if (setToZero){
		for (int k = 0; k < NZ; k++){
			for (int j = 0; j < NY; j++){
				for (int i = 0; i < NX; i++){
					this->operator()(i-border,j-border, k) = 0.0;
				}
			}
		}
	}
}

CpuPtr_3D::~CpuPtr_3D(){
	delete [] data;
}

float &CpuPtr_3D::operator() (unsigned int i, unsigned int j, unsigned int k){
	return data[(i+border)*(ny+2*border)*(nz) + nz*(j+border) + k];
}

void CpuPtr_3D::allocateMemory(){
	data = new float[NX*NY*NZ];
}

int CpuPtr_3D::getNx(){
	return nx;
}

int CpuPtr_3D::getNy(){
	return ny;
}

int CpuPtr_3D::getNz(){
	return nz;
}

float CpuPtr_3D::getDx(){
	return ((xmax - xmin)/(float)nx);
}

float CpuPtr_3D::getDy(){
	return ((ymax - ymin)/(float)ny);
}

/*float CpuPtr_3D::getDz(){
	return ((zmax - zmin)/(float)nz);
}*/


void CpuPtr_3D::setTime(float t){
	time = t;
}


CpuPtr_2D::CpuPtr_2D()
: nx(0), ny(0), border(0), NX(nx+2*border), NY(ny+2*border), xmin(0), ymin(0), xmax(1.0), ymax(1.0),time(0){
	allocateMemory();
		for (int j = 0; j < NY; j++){
			for (int i = 0; i < NX; i++){
				this->operator()(i-border,j-border) = 0.0;
			}
		}
}

CpuPtr_2D::CpuPtr_2D(unsigned int nx, unsigned int ny, unsigned int border, bool setToZero)
: nx(nx), ny(ny), border(border), NX(nx+2*border), NY(ny+2*border), xmin(0), ymin(0), xmax(1.0), ymax(1.0),time(0){
	allocateMemory();
	if (setToZero){
		for (int j = 0; j < NY; j++){
			for (int i = 0; i < NX; i++){ 
				this->operator()(i-border,j-border) = 0.0;
			}	
		}
	}
}

CpuPtr_2D::CpuPtr_2D(const CpuPtr_2D& other):nx(other.nx),ny(other.ny),border(other.border),NX(other.NX),NY(other.NY),xmin(other.xmin), ymin(other.ymin),xmax(other.xmax),ymax(other.ymax),time(other.time){
	allocateMemory();
	for (int i = 0; i < NX*NY; i++){
		data[i] = other.data[i];
	}
}

CpuPtr_2D::~CpuPtr_2D(){
	//delete [] data;
}

float &CpuPtr_2D::operator() (unsigned int i, unsigned int j){
	return data[((j) + border)*(nx + 2*border) + i + border];
}

void CpuPtr_2D::allocateMemory(){
	data = new float[NX*NY];
}

int CpuPtr_2D::getNx(){
	return nx;
}

int CpuPtr_2D::getNy(){
	return ny;
}

float CpuPtr_2D::getDx(){
	return ((xmax - xmin)/(float)nx);
}

float CpuPtr_2D::getDy(){
        return ((ymax - ymin)/(float)ny);
}

void CpuPtr_2D::setTime(float t){
	time = t;
}


CpuPtr_2D &CpuPtr_2D::operator = (const CpuPtr_2D &rhs){
	if(this == &rhs){
		return *this;
	}else{
		if (NX != rhs.NX || NY != rhs.NY){
			this->~CpuPtr_2D();
			nx = rhs.nx; ny = rhs.ny; border=rhs.border;
			NX = rhs.NX; NY = rhs.NY;
			allocateMemory();
		}
		for (int i = 0; i < NX*NY; i++){
			data[i]=rhs.data[i];
		}	
	}
}

void CpuPtr_2D::printToFile(FILE* filePtr){
	fprintf(filePtr, "nx: %i ny: %i\n", nx, ny);
	for (int j=0; j<ny; j++){
		for (int i=0; i<nx; i++){
			fprintf(filePtr, "%i\t%i\t%.15f\n", i, j, this->operator()(i,j));
		}
	}
}

void CpuPtr_2D::printToFileComparison(FILE* filePtr, CpuPtr_2D other){
	fprintf(filePtr, "NX: %i NY: %i\n", nx, ny);
	for (int j=0; j<ny; j++){
		for (int i=0; i<nx; i++){
			fprintf(filePtr, "%i\t%i\t%.19f\t%.19f\n", i, j, this->operator()(i,j), other(i,j));
		}
	}
}

void CpuPtr_2D::convertToDoublePointer(double* ptr){
	for (int i = 0; i < NX*NY; i++){
		ptr[i] = (double)data[i];
	}
}

/*
void CpuPtr_2D::printToFile(FILE* filePtr, bool withHeader, bool withBorder){

	float dx = (xmax - xmin)/(float)nx;
	float dy = (ymax -ymin)/(float)ny;
	if (not withBorder){	
		if (withHeader){
			fprintf(filePtr, "nx: %i ny: %i\nborder: %i\t time: %f\n", nx, ny, border,time);
			fprintf(filePtr, "xmin: %.1f xmax: %.1f\nymin: %.1f ymax: %.1f\n", xmin, xmax, ymin, ymax);
		}
		for (int j=0; j<nx; j++){
			for (int i=0; i<nx; i++){
				fprintf(filePtr, "%.3f\t%.3f\t%.3f\t%.3f\n", time, xmin + dx*i, ymin + dy*j, this->operator()(i,j));
			}
		}
	}else{
		if (withHeader){
			fprintf(filePtr, "nx: %i ny: %i\nborder: %i\t time: %f\n", NX, NY, border,time);
			fprintf(filePtr, "xmin: %.1f xmax: %.1f\nymin: %.1f ymax: %.1f\n", xmin, xmax, ymin, ymax);
		}
		for (int j=0; j<NY; j++){
			for (int i=0; i<NX; i++){
				fprintf(filePtr, "%.3f\t%.3f\t%.3f\t%.3f\n", time, xmin + dx*i-dx*border, ymin + dy*j-dy*border, this->operator()(i-border,j-border));
			}
		}

	}
*/



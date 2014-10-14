#ifndef CPU_PTR_H_
#define CPU_PTR_H_

#include <iostream>
#include <stdio.h>
#include "GpuPtr.h"

struct GpuData {
	GpuPtr_3D perm_dist_device;
	GpuPtr_2D Lambda_device;
	GpuPtr_2D K_device;
};
#endif

project(ExplicitTransportSolverGPU)
cmake_minimum_required(VERSION 2.8)

find_package("CUDA" REQUIRED)
find_package("Matlab" REQUIRED)
find_package("CUDPP" REQUIRED)

find_library(MATIO_LIBRARY NAMES matio)
find_path(MATIO_INCLUDE_DIR matio.h)

INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR} ${MATIO_INCLUDE_DIR} ${cudpp_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

set( CUDA_NVCC_FLAGS "-arch=sm_20" CACHE STRING "nvcc flags" FORCE)

file(GLOB SOURCE_FILES "./*.cpp" )
file(GLOB HEADER_FILES "./*.h" )
file(GLOB CUDA_FILES "./*.cu" )

cuda_compile(solver_cu ${CUDA_FILES})
add_executable(solver ${SOURCE_FILES} ${solver_cu})
target_link_libraries(solver ${CUDA_LIBRARIES} ${MATLAB_LIBRARIES} ${MATIO_LIBRARY} cudpp)
# Copyright (c) 2015 Robert Maier
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(DvoCuda C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE RelWithDebInfo FORCE)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
#SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

#==============================================================================
# Dependencies
#==============================================================================

# include sophus and Eigen
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/third_party/include/ )

# Eigen
SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/)
FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
#SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/cuda/lib64)
#INCLUDE_DIRECTORIES(/usr/local/cuda/lib64)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode=arch=compute_20,code=sm_20 --ptxas-options=-v --use_fast_math --compiler-options -Wall)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch sm_20 --ptxas-options=-v --use_fast_math --compiler-options -Wall -rdc=true -lcudaevrt)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch sm_20 --ptxas-options=-v --use_fast_math --compiler-options -Wall)

# misc cuda and c++ compiler flags
#IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
#	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_20 --ptxas-options=-v)
#ENDIF()
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -stdlib=libstdc++; -Xlinker -stdlib=libstdc++; --ptxas-options=-v)
#SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wall" )
#SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++" )

# OpenCV
FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS opencv_core opencv_highgui opencv_imgproc REQUIRED)

#==============================================================================
# Application
#==============================================================================

# preprocessor define: source directory
ADD_DEFINITIONS(-DDVO_CUDA_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# applications
FILE(GLOB incs "src/*.h")
FILE(GLOB incs "src/*.hpp")
FILE(GLOB srcs "src/*.cpp")
FILE(GLOB cuda_incs "src/*.cuh")
FILE(GLOB cuda_srcs "src/*.cu")
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE(dvo_cuda ${incs} ${srcs} ${cuda_incs} ${cuda_srcs})
CUDA_ADD_CUBLAS_TO_TARGET(dvo_cuda)
#SET(CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(dvo_cuda ${OpenCV_LIBS})
#SET_TARGET_PROPERTIES(dvo_cuda PROPERTIES LINKER_LANGUAGE CXX)

# Copyright (c) 2015 Robert Maier
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(DvoCuda C CXX)

# RelWithDebInfo, Release or Debug
SET(CMAKE_BUILD_TYPE RelWithDebInfo FORCE)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#==============================================================================
# Dependencies
#==============================================================================

# include sophus and Eigen
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/third_party/include/ )

# Eigen
SET(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/)
FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# OpenCV
FIND_PACKAGE(OpenCV 2.4.8 COMPONENTS opencv_core opencv_highgui opencv_imgproc REQUIRED)

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode=arch=compute_20,code=sm_50 --ptxas-options=-v --compiler-options -Wall -lcublas)

# Boost
FIND_PACKAGE(Boost COMPONENTS thread REQUIRED)

#PCL
#FIND_PACKAGE(PCL REQUIRED COMPONENTS common io visualization)
#INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
#LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
#ADD_DEFINITIONS(${PCL_DEFINITIONS})


#==============================================================================
# Application
#==============================================================================

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -funroll-loops" )

# preprocessor define: source directory
ADD_DEFINITIONS(-DDVO_SOURCE_DIR=${PROJECT_SOURCE_DIR})

# applications
FILE(GLOB incs "src/*.hpp")
FILE(GLOB srcs "src/*.cpp")
FILE(GLOB cuda_incs "src/*.cuh")
FILE(GLOB cuda_srcs "src/*.cu")

CUDA_ADD_EXECUTABLE(dvo_cuda ${incs} ${srcs} ${cuda_incs} ${cuda_srcs} )
CUDA_ADD_CUBLAS_TO_TARGET(dvo_cuda)
TARGET_LINK_LIBRARIES(dvo_cuda ${OpenCV_LIBS} ${PCL_LIBRARIES})

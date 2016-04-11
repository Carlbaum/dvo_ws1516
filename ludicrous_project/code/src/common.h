/**
 * \file
 * \brief   Global declarations.
 */

#pragma once

#include <Eigen/Dense>
#include <cuda_runtime.h>

using namespace Eigen;

typedef Matrix<float,6,1> Vector6f;
typedef Matrix<float,6,6> Matrix6f;

// global variables
const int MAX_LEVELS = 7;   // TODO: potential bug for more levels

// CUDA related
int devID;
cudaDeviceProp props;
int g_CUDA_maxSharedMemSize;
const int g_CUDA_blockSize2DX = 16;
const int g_CUDA_blockSize2DY = 16;
const int BORDER_ZERO = 1;
const int BORDER_REPLICATE = 2;

// tracker uses these global variables, so it has to be included after them
__constant__ float const_K_pyr[9*MAX_LEVELS];     // Allocates constant memory in excess for K and K downscaled. Stored column-wise and matrix after matrix
__constant__ float const_RK_inv[9];     // Allocates space for the concatenation of a rotation and an intrinsic matrix. Stored column-wise
__constant__ float const_translation[3];     // Allocates space for a translation vector
texture <float, 2, cudaReadModeElementType> texRef_grayImg;
texture <float, 2, cudaReadModeElementType> texRef_gray_dx;
texture <float, 2, cudaReadModeElementType> texRef_gray_dy;

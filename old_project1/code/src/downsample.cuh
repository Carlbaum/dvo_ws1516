#pragma once
#include <Eigen/Dense>
#include "helpers.hpp"

typedef Eigen::Matrix<float, 3, 3> Matrix3f;

__global__ void downsampleGrayKernel(float *imgDst, float *imgSrc, int w, int h) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < w && y < h) {
    float acc = 0;

    acc += imgSrc[2*x   + 2*y     * 2*w];
    acc += imgSrc[2*x+1 + 2*y     * 2*w];
    acc += imgSrc[2*x   + (2*y+1) * 2*w];
    acc += imgSrc[2*x+1 + (2*y+1) * 2*w];

    imgDst[x + y*w] = 0.25f * acc;
  }
}

__global__ void downsampleDepthKernel(float *imgDst, float *imgSrc, int w, int h) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < w && y < h) {
    float a00 = imgSrc[2*x   + 2*y     * 2*w];
    float a01 = imgSrc[2*x+1 + 2*y     * 2*w];
    float a10 = imgSrc[2*x   + (2*y+1) * 2*w];
    float a11 = imgSrc[2*x+1 + (2*y+1) * 2*w];

    imgDst[x + y*w] = a00 > 0 && a01 > 0 && a10 > 0 && a11 > 0
      ? 0.25f*(a00+a01+a10+a11)
      : 0;
  }
}

void downsampleGray(float *d_grayImgCoarse, float *d_grayImgFine, int lw, int lh) {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(ceilDivide(lw, blockDim.x), ceilDivide(lh, blockDim.y), 1);

    downsampleGrayKernel<<<gridDim, blockDim>>>(d_grayImgCoarse, d_grayImgFine, lw, lh); CUDA_CHECK;
}

void downsampleDepth(float *d_depthImgCoarse, float *d_depthImgFine, int lw, int lh) {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(ceilDivide(lw, blockDim.x), ceilDivide(lh, blockDim.y), 1);

    downsampleDepthKernel<<<gridDim, blockDim>>>(d_depthImgCoarse, d_depthImgFine, lw, lh); CUDA_CHECK;
}

Matrix3f downsampleK(const Matrix3f K) {
  Matrix3f K_d = K;
  K_d(0, 2) += 0.5; K_d(1, 2) += 0.5;
  K_d.topLeftCorner(2, 3) *= 0.5;
  K_d(0, 2) -= 0.5; K_d(1, 2) -= 0.5;
  return K_d;
}

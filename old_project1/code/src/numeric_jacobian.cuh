#include "aux.h"
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include "kernel_math.cuh"
#include <math.h>       /* sqrt */
#include "helpers.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

typedef KernelMatrix3f KMat3;
typedef KernelVector3f KVec3;
#define HUBER_DELTA  (4.0/100)
#define TDIST_DOF 5
#define TDIST_SCALE0 0.025
#define INVALID __int_as_float(0x7fffffff)
inline __device__ bool isValid(float x) { return !isnan(x); }

enum ResidualWeight {
    NONE   = 0,
    HUBER  = 1,
    TDIST  = 2
};

struct NumericJacobianStuff {
  float eps;
  KMat3 kMat;
  KVec3 tVec, tPermVecs[6];
  KMat3 rkInvMat, rkInvPermMats[6];
};

inline __device__ float interp2(float *img, float x, float y, int w, int h) {
  int x1 = (int)x, y1 = (int)y;
  int x2 = x1 + 1, y2 = y1 + 1;

  return img[x1 + y1*w] * (x2-x) *  (y2-y ) + img[x2 + y1*w] * (x-x1) * (y2-y )
       + img[x1 + y2*w] * (x2-x) *  (y -y1) + img[x2 + y2*w] * (x-x1)  *(y -y1);
}

inline __device__ float calcWeight(float r, ResidualWeight wType){
  float weight = 1.0;
  if (wType == HUBER) {
    float absPixResidual = abs(r);
    weight = absPixResidual > HUBER_DELTA ? (1.0 * HUBER_DELTA / absPixResidual) : 1.0;
  }
  return weight;
}

inline __device__ float calcResidual(
  float *grayPreImg, float* depthPreImg, float *grayCurImg,
  KMat3 K,
  KMat3 RKInv, KVec3 t,
  int x, int y,
  int w, int h, ResidualWeight wType
) {
  float d = depthPreImg[x + y*w];

  KVec3 p = { x * d, y * d, d };
  KVec3 pTrans = K * (RKInv * p + t);

  if(pTrans.a2 > 0 && d > 0) {
    float xCur = pTrans.a0 / pTrans.a2;
    float yCur = pTrans.a1 / pTrans.a2;
    if (xCur >= 0 && xCur <= w-1 && yCur >= 0 && yCur <= h-1) {
      float pixelCur = interp2(grayCurImg, xCur, yCur, w, h);
      float pixelPre = grayPreImg[x + y*w];
      float pixelResidual = pixelPre - pixelCur;
      pixelResidual *= calcWeight(pixelResidual, wType);
      return pixelResidual;
    }
  }

  return INVALID;
}

__global__ void d_calcResidualAndJacobian(
  float *jacobian, float *residual, int *n, float *visualResidual, float *grayPreImg,
  float *depthPreImg, float *grayCurImg, NumericJacobianStuff stuff, int w, int h, ResidualWeight wType
) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < w && y < h) {
    float r = calcResidual(grayPreImg, depthPreImg, grayCurImg, stuff.kMat, stuff.rkInvMat, stuff.tVec, x, y, w, h, wType);
    float rPerm[6];

    for (int i = 0; i < 6; i++) {
      rPerm[i] = calcResidual(
        grayPreImg, depthPreImg, grayCurImg, stuff.kMat, stuff.rkInvPermMats[i], stuff.tPermVecs[i], x, y, w, h, wType);
    }

    if (isValid(rPerm[0]) && isValid(rPerm[1]) && isValid(rPerm[2]) && isValid(r) &&
        isValid(rPerm[3]) && isValid(rPerm[4]) && isValid(rPerm[5])) {
      int pixelIdx = atomicAdd(n, 1);
      residual[pixelIdx] = r;
      for(int i = 0; i < 6; i++) {
        jacobian[w*h*i + pixelIdx] = (rPerm[i] - r) / stuff.eps;
      }
    }

    visualResidual[x + y*w] = isValid(r) ? r : 0;
  }
}

__global__ void d_TdistSquareTerms(float* d_wSquareterm, float* d_residual, int w, int h, float scale){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < w && y < h) {
    float data = d_residual[x + y*w];
    float weight = ( (TDIST_DOF + 1.0f) / (TDIST_DOF + (data * data) / (scale * scale) ) );
    d_wSquareterm[x + y*w] = data * data * weight;
  }
}
__global__ void d_TdistUpdate(float* d_residual, int w, int h, float scale){
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < w && y < h) {
    float data = d_residual[x + y*w];
    float weight = ( (TDIST_DOF + 1.0f) / (TDIST_DOF + (data * data) / (scale * scale) ) );
    d_residual[x + y*w] = weight*data;
  }
}

void calcTDistWeighted_R(float * d_residual, int n, int w, int h, float &initScale)
{
  cublasStatus_t ret;  
  cublasHandle_t handle;
  cublasCreate(&handle);
  float cb_result = 0.0f;

  float scale = initScale;
  int iterations = 0;
  float *d_tdist_weight, *d_wSquareterm;
  cudaMalloc(&d_tdist_weight, w*h*sizeof(float)); CUDA_CHECK;
  cudaMalloc(&d_wSquareterm, w*h*sizeof(float)); CUDA_CHECK;
  dim3 blockDim(16, 16, 1);
  dim3 gridDim(ceilDivide(w, blockDim.x), ceilDivide(h, blockDim.y), 1);
  do{
    initScale = scale;   
    d_TdistSquareTerms<<<gridDim, blockDim>>>(d_wSquareterm, d_residual, w, h, initScale); CUDA_CHECK;
    //compute new scale:
    ret = cublasSasum(handle, n , d_wSquareterm, 1 , &cb_result);
    scale = sqrt(cb_result / n);
    iterations ++;
  }
  while(std::abs( 1/(scale * scale) -  1/(initScale * initScale) ) > 1e-3 && iterations < 5);
  //cout << "Tdist estimate scale in  " << iterations << " iterations" << endl;
  d_TdistUpdate<<<gridDim, blockDim>>>(d_residual, w, h, scale); CUDA_CHECK;
  initScale = scale;

  cublasDestroy(handle);
  cudaFree(d_tdist_weight); CUDA_CHECK;
  cudaFree(d_wSquareterm); CUDA_CHECK;
}

void calcResidualAndJacobian(
  float *d_jacobian, float *d_residual, int *d_n, float &tdistInitScale, float *d_visualResidual,
  float *d_grayPreImg, float *d_depthPreImg, float *d_grayCurImg,
  Vector6f xi, Matrix3f K, int w, int h, ResidualWeight wType
) {
  NumericJacobianStuff stuff;
  stuff.kMat = *(KMat3 *)K.data();

  float eps = 0.000001f;
  stuff.eps = eps;

  Matrix3f R; Vector3f t; convertSE3ToTf(xi, R, t);
  Matrix3f RKInv = R * K.inverse();
  stuff.tVec = *(KVec3 *)t.data();
  stuff.rkInvMat = *(KMat3 *)RKInv.data();

  for(int i = 0; i < 6; i++) {
    Vector6f epsVec = Vector6f::Zero(); epsVec(i) = eps;
    Vector6f xiPerm = SE3f::log(SE3f::exp(epsVec) * SE3f::exp(xi));

    Matrix3f RPerm; Vector3f tPerm; convertSE3ToTf(xiPerm, RPerm, tPerm);
    Matrix3f RKInvPerm = RPerm * K.inverse();
    stuff.tPermVecs[i] = *(KVec3 *)tPerm.data();
    stuff.rkInvPermMats[i] = *(KMat3 *)RKInvPerm.data();
  }

  dim3 blockDim(16, 16, 1);
  dim3 gridDim(ceilDivide(w, blockDim.x), ceilDivide(h, blockDim.y), 1);

  d_calcResidualAndJacobian<<<gridDim, blockDim>>>(
    d_jacobian, d_residual, d_n, d_visualResidual, d_grayPreImg, d_depthPreImg, 
    d_grayCurImg, stuff, w, h, wType
  ); CUDA_CHECK;

  if(wType == TDIST){
      int n;
      cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;
      calcTDistWeighted_R(d_residual, n, w, h, tdistInitScale);
      //cout << "Tdist update scale " << tdistInitScale  << endl;
  }
}

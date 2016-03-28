#include "aux.h"
#include "helpers.hpp"
//se3Exp
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include "kernel_math.cuh"
#include <math.h>       /* sqrt */

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

struct AnalyticJacobianStuff {
  KVec3 tVec;
  KMat3 rkInvMat, kMat;
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

inline __device__ float d_biinterpolate2(float q11, float q21, float q12, float q22, float x, float y){
  int x1 = (int) x, y1 = (int) y;
  int x2 = x1 + 1, y2 = y1 + 1;
  return q11 * (x2-x) * (y2-y ) + q21 * (x-x1) * (y2-y )
         + q12 * (x2-x) * (y -y1) + q22 * (x-x1)  *(y -y1);
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

inline __device__ void d_derivative(float *d, float *img, int x, int y, int w, int h){
  int right = x + 1 ;
  int left = x - 1 ;
  int up = y - 1;
  int down = y + 1;
  d[0] = (right < w && left >= 0 && y < h) ?  0.5f * (img[right + y * w] - img[left + y * w]) : 0;
  d[1]= (down < h && up >= 0 && x < w) ?  0.5f * (img[x + down *w ] - img[x + up * w]) : 0;

}


__global__ void d_calc_analytic_jacobian(float* jacobian, float *residual, int *n,
  float *visualResidual, float *grayPreImg, float *depthPreImg,
  float *grayCurImg, AnalyticJacobianStuff stuff, int w, int h, ResidualWeight wType) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < w && y < h) {
    float r = calcResidual(grayPreImg, depthPreImg, grayCurImg, stuff.kMat, stuff.rkInvMat, stuff.tVec, x, y, w, h, wType);
    float d = depthPreImg[x + y*w];
    KVec3 p = { x * d, y * d, d };
    KVec3 pTrans =  stuff.rkInvMat * p + stuff.tVec;
    KVec3 pTransproj = stuff.kMat * pTrans;
    if (pTrans.a2 != 0 && pTransproj.a2 > 0 && d > 0) {
        float xCur = pTransproj.a0 / pTransproj.a2;
        float yCur = pTransproj.a1 / pTransproj.a2;

      //calculate derivative dI/dx, dI/dy at (xCur,yCur) via bilinear interpolation
      int x1 = (int) xCur;
      int y1 = (int) yCur;
      int x2 = x1 + 1;
      int y2 = y1 + 1;

      float d11[2], d21[2], d12[2], d22[2];

      d_derivative(d11, grayCurImg, x1, y1, w, h);
      d_derivative(d21, grayCurImg, x2, y1, w, h);
      d_derivative(d12, grayCurImg, x1, y2, w, h);
      d_derivative(d22, grayCurImg, x2, y2, w, h);

      float dxInterp = stuff.kMat.a00 * d_biinterpolate2(d11[0], d21[0], d12[0], d22[0], xCur, yCur);// 4 points interpolation
      float dyInterp = stuff.kMat.a11 * d_biinterpolate2(d11[1], d21[1], d12[1], d22[1], xCur, yCur);// 4 points interpolation
      float xp = pTrans.a0;
      float yp = pTrans.a1;
      float zp = pTrans.a2;
      float tmp_jacobian[6];

      tmp_jacobian[0] = dxInterp / zp;
      tmp_jacobian[1] = dyInterp / zp;
      tmp_jacobian[2] =  -(dxInterp * xp + dyInterp * yp) / (zp * zp);
      tmp_jacobian[3] =  -(dxInterp * xp * yp) / (zp * zp) - dyInterp * (1 + (yp / zp) * (yp / zp) );
      tmp_jacobian[4] =  dxInterp * (1 + (xp / zp) * (xp / zp) ) + (dyInterp * xp * yp) / (zp * zp);
      tmp_jacobian[5] =  (-dxInterp * yp + dyInterp * xp) / zp;

      if (isValid(r) && isValid(tmp_jacobian[0]) && isValid(tmp_jacobian[1]) && isValid(tmp_jacobian[2]) &&
          isValid(tmp_jacobian[3]) && isValid(tmp_jacobian[4]) && isValid(tmp_jacobian[5])) {
        int pixelIdx = atomicAdd(n, 1);
          residual[pixelIdx] = r;
          for(int i = 0; i < 6; i++) {
            jacobian[w*h*i + pixelIdx] = -tmp_jacobian[i];
          }
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
    cublasSasum(handle, n , d_wSquareterm, 1 , &cb_result);
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

// TODO oskar: thought I should try and make some documentation using doxygen

/**
 * This is where some explanation text will go when I've read the whole function.
 * Should calculate the residual image and jacobian
 * @param d_jacobian is a float pointer argument.
 * @param d_residual is a float pointer argument.
 * @param d_n is a float pointer argument.
 * @param tdistInitScale is a float argument.
 * @param d_visualResidual is a float pointer argument.
 * @param d_grayPreImg is a float pointer argument.
 * @param d_depthPreImg is a float pointer argument.
 * @param d_grayCurImg is a float pointer argument.
 * @param xi is a Vector6f argument.
 * @param K is a Matrix3f representing camera prooperties
 * @param w is an int representing resolution height of the image in the current pyramid level.
 * @param h is an int representing resolution height of the image in the current pyramid level.
 * @param wType is an enum that can be either 'NONE', 'HUBER' or 'TDIST'.
 * @return The test results
 */
void calcResidualAndJacobian(float *d_jacobian, float *d_residual, int *d_n, float &tdistInitScale,float *d_visualResidual,
  float *d_grayPreImg, float *d_depthPreImg, float *d_grayCurImg,
  Vector6f xi, Matrix3f K, int w, int h, ResidualWeight wType) {

  AnalyticJacobianStuff stuff;
  Matrix3f R; Vector3f t; convertSE3ToTf(xi, R, t); // converts the data in xi and stores it in R and t, using Sophus::SE3f::exp() function
  Matrix3f RKInv = R * K.inverse();

  stuff.kMat = *(KMat3 *)K.data();
  stuff.tVec = *(KVec3 *)t.data();
  stuff.rkInvMat = *(KMat3 *)RKInv.data();

  dim3 blockDim(16, 16, 1);
  dim3 gridDim(ceilDivide(w, blockDim.x), ceilDivide(h, blockDim.y), 1);

  d_calc_analytic_jacobian<<<gridDim, blockDim>>>(
      d_jacobian, d_residual, d_n, d_visualResidual, d_grayPreImg, d_depthPreImg, d_grayCurImg, stuff, w, h, wType
  ); CUDA_CHECK;

  if(wType == TDIST){
      int n;
      cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;
      calcTDistWeighted_R(d_residual, n, w, h, tdistInitScale);
      //cout << "Tdist update scale " << tdistInitScale  << endl;
  }
}

#ifndef RESIDUAL_H
#define RESIDUAL_H

#ifndef WIN64
	#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Core>

// __device__ float3 mulKVec(float* k, float3 vec);
__device__ float3 mulKVec(float k0, float k6, float k4, float k7, float3 vec);
__device__ float3 mulRVec(float* R, float3 vec);
void setConstMemR(Eigen::Matrix3f kPy, Eigen::Matrix3f ikPy, int offset);
// __global__ void computeResidual(float *iRef, float *dRef, float *iCrr, float *dCrr, float *k, float *ik, float *res, int w, int h, float *d_R, float *d_t);
__global__ void computeResidual(float *iRef, float *dRef, float *iCrr, float *dCrr, int lvl, float *res, int w, int h, float *d_R, float *d_t);
// void computeResiduals(float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_kPy, float **d_ikPy, float *d_res, int lvl, int w, int h, float *d_R, float *d_t);
void computeResiduals(float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float *d_res, int lvl, int w, int h, float *d_R, float *d_t);

#endif
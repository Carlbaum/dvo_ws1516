#ifndef JACOBIAN_H
#define JACOBIAN_H

#ifndef WIN64
	#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Core>

// __device__ float3 mulKVecJ(float* k, float3 vec);
__device__ float3 mulKVecJ(float k0, float k6, float k4, float k7, float3 vec);
__device__ float3 mulRVecJ(float* R, float3 vec);
void setConstMemJ(Eigen::Matrix3f kPy, Eigen::Matrix3f ikPy, int offset);
// __global__ void computeJacobian(float *dRef, float *iCrr, float *k, float *ik, int w, int h, float *R, float *t, float *J);
__global__ void computeJacobian(float *dRef, float *iCrr, int lvl, int w, int h, float *R, float *t, float *J);
// void computeJacobians(float **d_dPyRef, float **d_iPyCrr, float **d_kPy, float **d_ikPy, int lvl, int w, int h, float *d_R, float *d_t, float *d_J, float *d_dX, float *d_dY);
void computeJacobians(float **d_dPyRef, float **d_iPyCrr, int lvl, int w, int h, float *d_R, float *d_t, float *d_J, float *d_dX, float *d_dY);


#endif
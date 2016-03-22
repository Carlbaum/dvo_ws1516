#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#ifndef WIN64
	#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

__global__ void downscaleIMap(float *iSrc, float *iDst, int n_w, int n_h, int w, int h);
__global__ void downscaleDMap(float *dSrc, float *dDst, int n_w, int n_h, int w, int h);
__global__ void computeDerivatives (float *iCrr, float *dX, float *dY, int w, int h);
void buildMapPyramids(float **img, float **depth, int lvl, int w, int h);
void buildDrvPyramids(float **img, float **d_dXPy, float **d_dYPy, int lvl, int w, int h);
void buildKPyramid (Eigen::Matrix3f *KPy, int lvl);
void buildIKPyramid (Eigen::Matrix3f *iKPy, Eigen::Matrix3f *KPy, int lvl);

#endif
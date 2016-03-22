#ifndef REDUCTION_H
#define REDUCTION_H

#ifndef WIN64
	#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

__global__ void multiplyAndReducePt1(float *J, float *res, float *redArr, int w, int h);
__global__ void reducePt2(float *d_redArr, float *d_redArr2, int n);
void multiplyAndReduce(float *d_J, float *d_res, float *d_redArr, float *d_redArr2, int lvl, int w, int h, Mat6f *A, Vec6f *b);

#endif
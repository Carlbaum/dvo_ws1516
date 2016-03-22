#ifndef MISC_H
#define MISC_H

#ifndef WIN64
	#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sophus/se3.hpp>

void convertSE3ToTf(const Eigen::VectorXf &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t);
void convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Eigen::VectorXf &xi);
cv::Mat loadIntensity(const std::string &filename);
// cv::Mat loadDepth(const std::string &filename);
// __device__ float3 mulKVec(float* k, float3 vec);
// __device__ float3 mulRVec(float* R, float3 vec);
// __global__ void computeDerivatives (float *iCrr, float *dX, float *dY, int w, int h);

#endif
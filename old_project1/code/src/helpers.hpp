#pragma once
#include "aux.h"
#include <sophus/se3.hpp>

void showGPUImage(string windowName, float *d_img, int w, int h, int winX = 0, int winY = 0) {
  cv::Mat mat(h, w, CV_32F);
  cudaMemcpy(mat.data, d_img, w*h*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
  showImage(windowName, mat, winX, winY);
}



void convertSE3ToTf(const Eigen::VectorXf &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t) {
	// rotation
	Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
	Eigen::Matrix4f mat = se3.matrix();
	rot = mat.topLeftCorner(3, 3);
	t = mat.topRightCorner(3, 1);
}


void convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Eigen::VectorXf &xi) {
    Sophus::SE3f se3(rot, t);
    xi = Sophus::SE3f::log(se3);
}

inline int ceilDivide(int length, int factor) {
  return (length + factor - 1) / factor;
}
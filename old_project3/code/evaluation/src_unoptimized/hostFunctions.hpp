#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include <sophus/se3.hpp>


void convertSE3ToTf(const Eigen::VectorXf &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t)
{
    // rotation
    Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
    Eigen::Matrix4f mat = se3.matrix();
    rot = mat.topLeftCorner(3, 3);
    t = mat.topRightCorner(3, 1);
}


void convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Eigen::VectorXf &xi)
{
    Sophus::SE3f se3(rot, t);
    xi = Sophus::SE3f::log(se3);
}


cv::Mat loadIntensity(const std::string &filename)
{
    cv::Mat imgGray = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    // convert gray to float
    cv::Mat gray;
    imgGray.convertTo(gray, CV_32FC1, 1.0f / 255.0f);
    return gray;
}


cv::Mat loadDepth(const std::string &filename)
{
    //fill/read 16 bit depth image
    cv::Mat imgDepthIn = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    cv::Mat imgDepth;
    imgDepthIn.convertTo(imgDepth, CV_32FC1, (1.0 / 5000.0));
    return imgDepth;
}


#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif

#include <cublas_v2.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sophus/se3.hpp>

#include "aux.h"
#include "tum_benchmark.hpp"
#include "dataset.hpp"
//#include "viewer.hpp"
#include "tracker.hpp"

using namespace Eigen;

int main(int argc, char *argv[]) {
  // Settings
  string datasetPath = "../data/freiburg1_xyz_first_10";
  int maxFrames = -1;

  // Declarations
  Dataset dataset(datasetPath);
  std::vector<Eigen::Matrix4f> poses;
  std::vector<double> timestamps;
  Eigen::Matrix3f K = dataset.K;

  // Load images for frame 0
  cv::Mat color0 = loadColor(dataset.frames[0].colorPath);
  cv::Mat gray0; cv::cvtColor(color0, gray0, CV_BGR2GRAY);
  cv::Mat depth0 = loadDepth(dataset.frames[0].depthPath);

  // Store pose for frame 0
  poses.push_back(Matrix4f::Identity());
  timestamps.push_back(dataset.frames[0].timestamp);

  Tracker tracker(gray0, depth0, K, GAUSS_NEWTON, HUBER, 0, 4); //TDIST

  for (int i = 1; i < dataset.frames.size() && (maxFrames < 0 || i < maxFrames); ++i) {
    // Load images for frame i
    cv::Mat color = loadColor(dataset.frames[i].colorPath);
    cv::Mat gray; cv::cvtColor(color, gray, CV_BGR2GRAY);
    cv::Mat depth = loadDepth(dataset.frames[i].depthPath);

    // Frame alignment
    Vector6f frameXi = tracker.align(gray, depth);
    std::cout << "Xi between frames " << (i-1) << " and " << i << " ("
      << tracker.frameComputationTime << "ms): " << frameXi.transpose() << std::endl;

    // Update and push absolute pose
    Matrix4f absPose = Sophus::SE3f::exp(tracker.xi).matrix();
    poses.push_back(absPose);
    timestamps.push_back(dataset.frames[i].timestamp);

    // Update viewer
    //updateViewer(createPointCloud(K, depth, color), absPose);
  }

  std::cout << "Average runtime: " << tracker.averageTime() << " ms" << std::endl;

  // Save poses to disk
  savePoses(datasetPath + "/traj.txt", poses, timestamps);

  // Clean up
  cv::destroyAllWindows();
  return 0;
}

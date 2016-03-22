#ifndef VIEWER_H
#define VIEWER_H

#include <fstream>
#include <vector>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

void runViewer();

bool initialized = false;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
boost::shared_ptr<boost::thread> visThread = boost::shared_ptr<boost::thread>(new boost::thread(runViewer));
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudInit(new pcl::PointCloud<pcl::PointXYZRGB>);
bool cloudUpdated = false;
bool cloudInitUpdated = false;
boost::mutex cloudMutex;
Eigen::Matrix4f cloudPose = Eigen::Matrix4f::Identity();
bool doStopViewer = false;
bool pausing = false;


pcl::PointCloud<pcl::PointXYZRGB> createPointCloud(
  const Eigen::Matrix3f K, const cv::Mat &depth, const cv::Mat &color
) {
  pcl::PointCloud<pcl::PointXYZRGB> pointCloud;

  int w = depth.cols;
  int h = depth.rows;

  float *pDepth = (float*)depth.data;
  float *pColor = (float*)color.data;

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      size_t off = (y*w + x);
      size_t off3 = off * 3;

      // Check depth value validity
      float d = pDepth[x + y*w];
      if (d == 0.0f || std::isnan(d)) { continue; }

      // RGBXYZ point
      pcl::PointXYZRGB point;

      // Color
      point.b = max(0, min(255, (int)(pColor[off3  ]*255)));
      point.g = max(0, min(255, (int)(pColor[off3+1]*255)));
      point.r = max(0, min(255, (int)(pColor[off3+2]*255)));

      // Position
      Eigen::Vector3f pos = K.inverse() * Eigen::Vector3f(x*d, y*d, d);
      point.x = pos[0];
      point.y = pos[1];
      point.z = pos[2];

      pointCloud.push_back(point);
    }
  }

  return pointCloud;
}

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event, void* viewer_void) {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
  if (event.getKeySym () == "n" && event.keyDown () && pausing)
  {
    std::cout << "n was pressed => next frame" << std::endl;
    pausing = false;
  }
}


void runViewer()
{
    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.3, 0.3, 0.3);

    viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbInit(cloudInit);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloudInit, rgbInit, "cloudInit");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloudInit");

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");

    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();

    viewer->setCameraPosition(0,0,-1,0,0,0,-1,0,0);
    viewer->setCameraPosition(0,0,-1,0,0,0,0,-1,0);

    initialized = true;

    while (!viewer->wasStopped() && !doStopViewer)
    {
        if (cloudUpdated) {
            cloudMutex.lock();
            viewer->updatePointCloud(cloud, "cloud");
            Eigen::Affine3f poseAff(cloudPose);
            viewer->updatePointCloudPose("cloud", poseAff);
            viewer->addCoordinateSystem(0.1, poseAff);
            cloudUpdated = false;
            cloudMutex.unlock();
        }
        if (cloudInitUpdated) {
            cloudMutex.lock();
            //viewer->updatePointCloud(cloudInit, "cloudInit");
            cloudInitUpdated = false;
            cloudMutex.unlock();
        }

        viewer->spinOnce(1);
        boost::this_thread::sleep(boost::posix_time::milliseconds(1));
    }

    //viewer->close();
}


bool updateViewer(const pcl::PointCloud<pcl::PointXYZRGB> &pointCloud, const Eigen::Matrix4f &pose)
{
    if (!initialized) { return true; }
    if (viewer->wasStopped()) { return false; }

    cloudMutex.lock();
    if (cloudInit->empty())
    {
        pcl::copyPointCloud(pointCloud, *cloudInit);
        cloudInitUpdated = true;
    }
    else
    {
        pcl::copyPointCloud(pointCloud, *cloud);
        cloudPose = pose;
        cloudUpdated = true;
    }
    cloudMutex.unlock();

    // while(pausing) { boost::this_thread::sleep(boost::posix_time::milliseconds(1)); }
    pausing = true;

    return !viewer->wasStopped();
}


void stopViewer()
{
    viewer->close();
    doStopViewer = true;
}

#endif

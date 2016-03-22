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

void runViewer()
{
    viewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0.3, 0.3, 0.3);

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbInit(cloudInit);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloudInit, rgbInit, "cloudInit");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloudInit");

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");

    viewer->addCoordinateSystem(0.3);
    viewer->initCameraParameters();
    initialized = true;

    while (!viewer->wasStopped() && !doStopViewer)
    {
        if (cloudUpdated)
        {
            cloudMutex.lock();
            viewer->updatePointCloud(cloud, "cloud");
            Eigen::Affine3f poseAff(cloudPose);
            viewer->updatePointCloudPose("cloud", poseAff);
            viewer->addCoordinateSystem(0.3, poseAff);
            cloudUpdated = false;
            cloudMutex.unlock();
        }
        if (cloudInitUpdated)
        {
            cloudMutex.lock();
            viewer->updatePointCloud(cloudInit, "cloudInit");
            cloudInitUpdated = false;
            cloudMutex.unlock();
        }

        viewer->spinOnce(5);
        boost::this_thread::sleep(boost::posix_time::milliseconds(5));
    }

    //viewer->close();
}


bool updateViewer(const cv::Mat &vertexMap, const cv::Mat &color, const Eigen::Matrix4f &pose)
{
    if (!initialized)
        return true;
   /* if (viewer->wasStopped())
        return false;*/

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scan(new pcl::PointCloud<pcl::PointXYZRGB>);
    int w = vertexMap.cols;
    int h = vertexMap.rows;
    float* ptrVert = (float*)vertexMap.data;
    unsigned char* ptrColor = (unsigned char*)color.data;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            size_t off = (y*w + x);
            size_t off3 = off * 3;
            float d = ptrVert[off3+2];
            if (d == 0.0f || std::isnan(d))
                continue;

            pcl::PointXYZRGB pt(ptrColor[off3+2], ptrColor[off3+1], ptrColor[off3]);
            if (cloudInit->empty())
            {
                float colorScale = 1.3;
                float r = pt.r * colorScale;
                if (r > 255.0) r = 255.0;
                float g = pt.g * colorScale;
                if (g > 255.0) g = 255.0;
                float b = pt.b * colorScale;
                if (b > 255.0) b = 255.0;
                pt.r = r;
                pt.g = g;
                pt.b = b;
            }
            pt.x = ptrVert[off3];
            pt.y = ptrVert[off3+1];
            pt.z = d;
            scan->push_back(pt);
        }
    }

    cloudMutex.lock();
    if (cloudInit->empty())
    {
        pcl::copyPointCloud(*scan, *cloudInit);
        cloudInitUpdated = true;
    }
    else
    {
        pcl::copyPointCloud(*scan, *cloud);
        cloudPose = pose;
        cloudUpdated = true;
    }
    cloudMutex.unlock();

    return !viewer->wasStopped();
}


void stopViewer()
{
    //viewer->close();
    doStopViewer = true;
}

#endif

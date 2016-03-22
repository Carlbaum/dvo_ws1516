#include <iostream>
#include <vector>

#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <sophus/se3.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
texture<float,2,cudaReadModeElementType> texRef;
texture<float,2,cudaReadModeElementType> texGradX;
texture<float,2,cudaReadModeElementType> texGradY;

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

#define ITER 10
#define LVL 4
 
#include "aux.h"

#include "hostFunctions.hpp"
#include "downsample.cuh"
#include "residual.cuh"
#include "pointCloud.cuh"
#include "deriveNumeric.cuh"
#include "tum_benchmark.hpp"
#include "save_ply.hpp"
#include "viewer.hpp"
#include "updateXi.cuh"

#define STR1(x)  #x
#define STR(x)  STR1(x)


bool useGpu = false;
int numIterations = 20;
int numPyramidLevels = 5;
int maxLevel = numPyramidLevels-1;
int minLevel = 1;
float errorImprovementStop = 0.995f; //0.99f

Eigen::Matrix3f K;
double runtimeAvg = 0.0;


void align(const cv::Mat &depthRefIn, const cv::Mat &grayRefIn, const cv::Mat &depthCurIn, const cv::Mat &grayCurIn, Vec6f& xi)
{
    // get image dimensions
    int w = grayRefIn.cols;         // width
    int h = grayRefIn.rows;         // height
    double tmr = 0;
    std::cout << "image: " << w << " x " << h << std::endl;

    // initialize intrinsic matrix
    Eigen::Matrix3f K;
    K <<    517.3, 0.0, 318.6,
            0.0, 516.5, 255.3,
            0.0, 0.0, 1.0;

    // initial pose
    Eigen::Matrix3f rot;
    Eigen::Vector3f t;
    //Vec6f xi = Vec6f::Zero();
    convertSE3ToTf(xi, rot, t);
    std::cout << "Initial pose: " << std::endl;
    std::cout << "t = " << t.transpose() << std::endl;
    std::cout << "R = " << rot << std::endl;

    //Saving the finest level of images
    std::vector<Eigen::Matrix3f> kPyramid;
    kPyramid.push_back(K);
    std::vector<cv::Mat> grayRefPyramid;
    grayRefPyramid.push_back(grayRefIn);
    std::vector<cv::Mat> depthRefPyramid;
    depthRefPyramid.push_back(depthRefIn);
    std::vector<cv::Mat> grayCurPyramid;
    grayCurPyramid.push_back(grayCurIn);

    // initialize cuda context
    cudaDeviceSynchronize();  CUDA_CHECK;

    //Downsampling
    downSample(LVL, kPyramid, grayRefPyramid, depthRefPyramid, grayCurPyramid);
	
    //allignment
    updateXi(kPyramid, grayRefPyramid, depthRefPyramid, grayCurPyramid,rot,t,xi);
    tmr = ((double)cv::getTickCount() - tmr)/cv::getTickFrequency();
    std::cout << "runtime: " << tmr * 1000.0 << " ms" << std::endl;
    runtimeAvg += tmr;
}


int main(int argc, char *argv[])
{
    std::string dataFolder = std::string(STR(DVO_CUDA_SOURCE_DIR)) + "/data/";

    // load file names
    std::string assocFile = dataFolder + "rgbd_assoc.txt";
    std::vector<std::string> filesColor;
    std::vector<std::string> filesDepth;
    std::vector<double> timestampsDepth;
    std::vector<double> timestampsColor;
    if (!loadAssoc(assocFile, filesDepth, filesColor, timestampsDepth, timestampsColor))
    {
        std::cout << "Assoc file could not be loaded!" << std::endl;
        return 1;
    }
    int numFrames = filesDepth.size();

    // initialize cuda context
    cudaDeviceSynchronize(); CUDA_CHECK;

    // initialize intrinsic matrix
    K <<    517.3, 0.0, 318.6,
            0.0, 516.5, 255.3,
            0.0, 0.0, 1.0;
    //std::cout << "Camera matrix: " << K << std::endl;

    int maxFrames = -1;
    //maxFrames = 20;
    bool canceled = false;

    // process frames
    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> poses;
    std::vector<double> timestamps;
    poses.push_back(absPose);
    timestamps.push_back(timestampsDepth[0]);

    int framesProcessed = 0;
    cv::Mat colorPrev = loadColor(dataFolder + filesColor[0]);
    cv::Mat grayPrev;
    cv::cvtColor(colorPrev, grayPrev, CV_BGR2GRAY);
    cv::Mat depthPrev = loadDepth(dataFolder + filesDepth[0]);

#if 1
    cv::Mat vertexMap0;
    depthToVertexMap(K.cast<double>(), depthPrev, vertexMap0);
    cv::Mat color0UC;
    colorPrev.convertTo(color0UC, CV_8UC3, 255.0f);
    canceled = !updateViewer(vertexMap0, color0UC, absPose);
#endif

    for (size_t i = 1; i < numFrames && (maxFrames < 0 || i < maxFrames) && !canceled; ++i)
    {
        // load input frame
        std::string fileColor1 = filesColor[i];
        std::string fileDepth1 = filesDepth[i];
        double timeDepth1 = timestampsDepth[i];
        //std::cout << "File " << i << ": " << fileColor1 << ", " << fileDepth1 << std::endl;
        cv::Mat color0 = colorPrev;
        cv::Mat depth0 = depthPrev;
        cv::Mat gray0 = grayPrev;

        cv::Mat color1 = loadColor(dataFolder + fileColor1);
        cv::Mat depth1 = loadDepth(dataFolder + fileDepth1);
        cv::Mat gray1;
        cv::cvtColor(color1, gray1, CV_BGR2GRAY);

        cv::Mat grayRef = gray0;
        cv::Mat depthRef = depth0;
        cv::Mat grayCur = gray1;
        cv::Mat depthCur = depth1;

        // frame alignment
        Vec6f xi = Vec6f::Zero();
        align(depthRef, grayRef, depthCur, grayCur, xi);

        Eigen::Matrix3f rot;
        Eigen::Vector3f t;
        convertSE3ToTf(xi, rot, t);
        std::cout << "pose (xi) between frames " << (i-1) << " and " << i  << ": " << xi.transpose() << std::endl;
        //std::cout << "t = " << t.transpose() << std::endl;
        //std::cout << "R = " << rot << std::endl;

        //Vec6f xiResult;
        //xiResult << -0.0021f, 0.0057f, 0.0374f, -0.0292f, -0.0183f, -0.0009f;
        //std::cout << "xi expected = " << xiResult.transpose() << std::endl;

        // concatenate poses
        Eigen::Matrix4f relPose = Eigen::Matrix4f::Identity();
        relPose.topLeftCorner(3,3) = rot;
        relPose.topRightCorner(3,1) = t;
        absPose = absPose * relPose.inverse();
        poses.push_back(absPose);
        timestamps.push_back(timeDepth1);

        rot = absPose.topLeftCorner(3,3);
        t = absPose.topRightCorner(3,1);
        cv::Mat vertexMap1;
        depthToVertexMap(K.cast<double>(), depth1, vertexMap1);
        cv::Mat vertexMapTf1 = vertexMap1.clone();
        transformVertexMap(rot.cast<double>(), t.cast<double>(), vertexMapTf1);
        cv::Mat color1UC;
        color1.convertTo(color1UC, CV_8UC3, 255.0f);

#if 0
        // save frames as point cloud
        std::stringstream ss;
        ss << dataFolder << "cloud_" << std::setw(4) << std::setfill('0') << i << ".ply";
        savePlyFile(ss.str(), color1UC, vertexMapTf1);
#endif

#if 1
        canceled = !updateViewer(vertexMap1, color1UC, absPose);
#endif

        colorPrev = color1;
        depthPrev = depth1;
        grayPrev = gray1;
        ++framesProcessed;
    }
    stopViewer();
    std::cout << "average runtime: " << (runtimeAvg / framesProcessed) * 1000.0 << " ms" << std::endl;
    
    // save poses
    savePoses(dataFolder + "traj.txt", poses, timestamps);

#if 0
    cv::waitKey(0);
#endif

    // clean up
    cv::destroyAllWindows();
    std::cout << "Direct Image Alignment finished." << std::endl;
    return 0;
}

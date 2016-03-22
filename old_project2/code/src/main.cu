#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>

#ifndef WIN64
	#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

#include <sophus/se3.hpp>

#include "aux.h"
#include "Reduction.h"
#include "GPUMem.h"
#include "Test.h"
#include "Preprocessing.h"
#include "Misc.cuh"
#include "Residual.h"
#include "Jacobian.h"
#include "tum_benchmark.hpp"
#include "save_ply.hpp"

#include "cublas_v2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define STR1(x)  #x
#define STR(x)  STR1(x)

typedef Eigen::Matrix<double, 6, 6> Mat6;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

float totTime = 0.0f;
float divisor = 0.0f;


// void align(const cv::Mat &depthCur, const cv::Mat &grayCur, Vec6f& xi, float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_kPy, float **d_ikPy, float **d_dXPy, float **d_dYPy, float *d_res, float *d_J, float *d_redArr, float *d_redArr2, float *d_R, float *d_t, int w, int h, size_t n, int lvlNum, float dth)
void align(const cv::Mat &depthCur, const cv::Mat &grayCur, Vec6f& xi, float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_dXPy, float **d_dYPy, float *d_res, float *d_J, float *d_redArr, float *d_redArr2, float *d_R, float *d_t, int w, int h, size_t n, int lvlNum, float dth)
{

    // initial pose
    Eigen::Matrix3f rot;
    Eigen::Vector3f t;
    convertSE3ToTf(xi, rot, t);
    Mat6f A;
    Vec6f b, delta;

    int n_w = w;
    int n_h = h;
    Timer timer; timer.start();
    for (int lvl=lvlNum-1; lvl>=0; lvl--) {
    
        for (int i=0; i<lvl; i++) {
            n_w = (n_w+1)/2;
            n_h = (n_h+1)/2;
        }

        for (int i=0; i<20; i++) {
            convertSE3ToTf(xi, rot, t);

            cudaMemcpy(d_R, rot.data(), 9*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;
            cudaMemcpy(d_t, t.data(), 3*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;

            cudaMemset(d_J, 0, 6*n*sizeof(float));  CUDA_CHECK;
            cudaMemset(d_res, 0, n*sizeof(float));  CUDA_CHECK;

            computeResiduals(d_iPyRef, d_dPyRef, d_iPyCrr, d_dPyCrr, d_res, lvl, n_w, n_h, d_R, d_t); CUDA_CHECK;

            // testPyramidLevels(n_w, n_h, d_iPyRef, d_iPyCrr, d_dPyRef, d_dPyCrr, d_res, lvl, showCur, showType);
            // testPyramidLevels(n_w, n_h, d_iPyRef, d_iPyCrr, d_dPyRef, d_dPyCrr, d_res, showLvl, showCur, showType);

            computeJacobians(d_dPyRef, d_iPyCrr, lvl, n_w, n_h, d_R, d_t, d_J, d_dXPy[lvl], d_dYPy[lvl]); CUDA_CHECK;

            // testJacobian(d_J, lvl, n_w, n_h);

            multiplyAndReduce(d_J, d_res, d_redArr, d_redArr2, lvl, n_w, n_h, &A, &b); CUDA_CHECK;


            delta = -(A.ldlt().solve(b));

            if (std::abs(delta[0]) < dth &&
                std::abs(delta[1]) < dth &&
                std::abs(delta[2]) < dth &&
                std::abs(delta[3]) < dth &&
                std::abs(delta[4]) < dth &&
                std::abs(delta[5]) < dth) {
                // std::cout << "early break of lvl " << lvl << " on iteration " << i << std::endl;
                break;
            }

            // update xi
            xi = Sophus::SE3f::log(Sophus::SE3f::exp(delta)*Sophus::SE3f::exp(xi));

        }

        n_w = w;
        n_h = h;
    }
    timer.end();  float timeElapsed = timer.get();  // elapsed time in seconds
    // std::cout << "aligning time: " << timeElapsed*1000 << " ms" << std::endl;
    totTime += (timeElapsed*1000); divisor ++;

}







// void align(const cv::Mat &depthCur, const cv::Mat &grayCur, Vec6f& xi, float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_kPy, float **d_ikPy, float **d_dXPy, float **d_dYPy, float *d_res, float *d_J, float *d_redArr, float *d_redArr2, float *d_R, float *d_t, int w, int h, size_t n, int lvlNum, float dth)
void alignCB(cublasHandle_t handle, const cv::Mat &depthCur, const cv::Mat &grayCur, Vec6f& xi, float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_dXPy, float **d_dYPy, float *d_res, float *d_J, float *d_A, float *d_b, float *d_R, float *d_t, int w, int h, size_t n, int lvlNum, float dth, float *d_redArr, float *d_redArr2)
{

    // initial pose
    Eigen::Matrix3f rot;
    Eigen::Vector3f t;
    convertSE3ToTf(xi, rot, t);
    Mat6f A;
    Vec6f b, delta;

    float one = 1.0f;

    int n_w = w;
    int n_h = h;
    Timer timer; timer.start();
    for (int lvl=lvlNum-1; lvl>=0; lvl--) {
    
        for (int i=0; i<lvl; i++) {
            n_w = (n_w+1)/2;
            n_h = (n_h+1)/2;
        }

        for (int i=0; i<20; i++) {
            convertSE3ToTf(xi, rot, t);

            cudaMemcpy(d_R, rot.data(), 9*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;
            cudaMemcpy(d_t, t.data(), 3*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;

            cudaMemset(d_J, 0, 6*n*sizeof(float));  CUDA_CHECK;
            cudaMemset(d_res, 0, n*sizeof(float));  CUDA_CHECK;

            computeResiduals(d_iPyRef, d_dPyRef, d_iPyCrr, d_dPyCrr, d_res, lvl, n_w, n_h, d_R, d_t); CUDA_CHECK;

            // testPyramidLevels(n_w, n_h, d_iPyRef, d_iPyCrr, d_dPyRef, d_dPyCrr, d_res, lvl, showCur, showType);
            // testPyramidLevels(n_w, n_h, d_iPyRef, d_iPyCrr, d_dPyRef, d_dPyCrr, d_res, showLvl, showCur, showType);

            computeJacobians(d_dPyRef, d_iPyCrr, lvl, n_w, n_h, d_R, d_t, d_J, d_dXPy[lvl], d_dYPy[lvl]); CUDA_CHECK;

            // testJacobian(d_J, lvl, n_w, n_h);


            // multiplyAndReduce(d_J, d_res, d_redArr, d_redArr2, lvl, n_w, n_h, &A, &b); CUDA_CHECK;

            cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T, 6, 6, n_w*n_h, &one, d_J, 6, d_J, 6, &one, d_A, 6 );
            cublasSgemv( handle, CUBLAS_OP_N, 6, n_w*n_h, &one, d_J, 6, d_res, 1, &one, d_b, 1 );

            cudaMemcpy( A.data(), d_A, 36*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;
            cudaMemcpy( b.data(), d_b, 6*sizeof(float), cudaMemcpyDeviceToHost ); CUDA_CHECK;


            delta = -(A.ldlt().solve(b));

            if (std::abs(delta[0]) < dth &&
                std::abs(delta[1]) < dth &&
                std::abs(delta[2]) < dth &&
                std::abs(delta[3]) < dth &&
                std::abs(delta[4]) < dth &&
                std::abs(delta[5]) < dth) {
                // std::cout << "early break of lvl " << lvl << " on iteration " << i << std::endl;
                break;
            }

            // update xi
            xi = Sophus::SE3f::log(Sophus::SE3f::exp(delta)*Sophus::SE3f::exp(xi));

        }

        n_w = w;
        n_h = h;
    }
    timer.end();  float timeElapsed = timer.get();  // elapsed time in seconds
    // std::cout << "aligning time: " << timeElapsed*1000 << " ms" << std::endl;
    totTime += (timeElapsed*1000); divisor ++;

}







int main(int argc, char *argv[])
{
	// parameter definitions

    // returns the parameters
    bool help = false;
    getParam("help", help, argc, argv);
    if(help) {
        std::cout << "[-lvlnum <Number of levels>]" << std::endl
        << "[-slvl <level to show>]" << std::endl
        << "[-scur]" << std::endl
        << "[-stype <0:intensity|1:depth|2:residuals>]" << std::endl
        << "[-datfol <Name of data folder>]" << std::endl
        << "[-maxf <Maximum number of frames>]" << std::endl
        << "[-dth <Early break threshold for delta>]" << std::endl
        << "[-cublas]" << std::endl;
        return 0;
    }

	// gives the number of levels of the pyramids
	int lvlNum = 4;
	getParam("lvlnum", lvlNum, argc, argv);
	lvlNum = std::max(1, lvlNum);
	std::cout << "number of levels in pyramids: " << lvlNum << std::endl;
	
	// indicates which Lvl to show
	int showLvl = 0;
	getParam("slvl", showLvl, argc, argv);
	showLvl = std::max(0, showLvl);
	if (showLvl >= lvlNum) showLvl = lvlNum - 1;
	std::cout << "showLvl: " << showLvl << std::endl;

	// indicates if current or reference is shown
	bool showCur = false;
	getParam("scur", showCur, argc, argv);
	std::cout << "showCurrent: " << showCur << std::endl;

    // indicates which image type is shown (0:intensity|1:depth|2:residuals)
    int showType = 0;
    getParam("stype", showType, argc, argv);
    std::cout << "show image type: " << showType << std::endl;

    // name of data folder
    std::string datFolder = "freiburg";
    getParam("datfol", datFolder, argc, argv);
    std::cout << "data folder name: " << datFolder << std::endl;

    // maximum number of frames (-1 for all)
    int maxFrames = -1;
    getParam("maxf", maxFrames, argc, argv);
    std::cout << "max number of frames: " << maxFrames << std::endl;

    // early break threshold for delta
    float dth = 0.00005f;
    getParam("dth", dth, argc, argv);
    std::cout << "early break threshold for delta: " << dth << std::endl;

    // use cuBLAS for multiplication
    bool cublas = false;
    getParam("cublas", cublas, argc, argv);
    std::cout << "use cuBLAS for multiplication: " << cublas << std::endl;

    std::string dataFolder = std::string(STR(DVO_CUDA_SOURCE_DIR)) + "/" + datFolder +"/";
    std::cout << "data folder: " << dataFolder << std::endl;


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

    // process frames
    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> poses;
    std::vector<double> timestamps;
    poses.push_back(absPose);
    timestamps.push_back(timestampsDepth[0]);

    cudaDeviceSynchronize();  CUDA_CHECK;

    Vec6f xi = Vec6f::Zero();

    cv::Mat grayRef = loadIntensity(dataFolder + filesColor[0]);
    cv::Mat depthRef = loadDepth(dataFolder + filesDepth[0]);

    // get image dimensions
    int w = grayRef.cols;        // width
    int h = grayRef.rows;        // height
    std::cout << "image: " << w << " x " << h << std::endl;

    //#########################################################################
    //###### Begin ############################################################
    //#########################################################################

    size_t n = (size_t)w*h;

    // define pointer arrays for pyramids and pointers for R, t, A and b
    float **d_iPyRef = new float*[lvlNum];  // intensity map pyramid of reference image
    float **d_dPyRef = new float*[lvlNum];  // depth     map pyramid of reference image
    float **d_iPyCrr = new float*[lvlNum];  // intensity map pyramid of current   image
    float **d_dPyCrr = new float*[lvlNum];  // depth     map pyramid of current   image
    float **d_kPy = new float*[lvlNum];     // K         matrix pyramid
    float **d_ikPy = new float*[lvlNum];    // K inverse matrix pyramid
    float **d_dXPy = new float*[lvlNum];    // horizontal derivative map pyramid
    float **d_dYPy = new float*[lvlNum];    // vertical derivative map pyramid
    float *d_res = 0;                       // residual map
    float *d_J = 0;                         // jacobian matrix
    float *d_redArr = 0;                    // parallel reduction array in global memory
    float *d_redArr2 = 0;                   // parallel reduction array in global memory
    float *d_A = 0;                         // A-matrix in global memory (CuBLAS)
    float *d_b = 0;                         // b-vector in global memory (CuBLAS)
    float *d_R = 0;                         // rotation matrix
    float *d_t = 0;                         // translation vector

    // allocate and initialize all the pyramids
    allocGPUMem(d_iPyRef, d_dPyRef, d_iPyCrr, d_dPyCrr, d_kPy, d_ikPy, d_dXPy, d_dYPy, &d_res, &d_J, lvlNum, (size_t)w*h, &d_R, &d_t, &d_redArr, &d_redArr2, &d_A, &d_b);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // array for K pyramid
    Eigen::Matrix3f *kPy = new Eigen::Matrix3f[lvlNum];
    kPy[0] <<   517.3f, 0.0f, 318.6f,
                0.0f, 516.5f, 255.3f,
                0.0f, 0.0f, 1.0f;
    buildKPyramid(kPy, lvlNum);
    Eigen::Matrix3f *ikPy = new Eigen::Matrix3f[lvlNum];
    buildIKPyramid(ikPy, kPy, lvlNum);

    // upload K pyramids to GPU
    int offset = 0;
    for (int i=0; i<lvlNum; i++) {
        // cudaMemcpy(d_kPy[i], kPy[i].data(), 9*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;
        // cudaMemcpy(d_ikPy[i], ikPy[i].data(), 9*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;
        offset = i*9*sizeof(float);
        setConstMemR(kPy[i], ikPy[i], offset); CUDA_CHECK;
        setConstMemJ(kPy[i], ikPy[i], offset); CUDA_CHECK;
    }

    size_t numFrames = filesDepth.size();
    bool canceled = false;

    cv::Mat grayCur = loadIntensity(dataFolder + filesColor[0]);
    cv::Mat depthCur = loadDepth(dataFolder + filesDepth[0]);
        
    // initially copy current input images to level 0 for later use as reference
    cudaMemcpy(d_iPyRef[0], (void*)grayCur.data, n*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;
    cudaMemcpy(d_dPyRef[0], (void*)depthCur.data, n*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;
    buildMapPyramids(d_iPyRef, d_dPyRef, lvlNum, w, h);

    float ** switchTemp; // temporary pointer for switching pointers
    //##### LOOP BEGIN #####
    Timer timer2; timer2.start();
    for (size_t frame = 1; frame < numFrames && (maxFrames < 0 || (int)frame < maxFrames) && !canceled; ++frame)
    {

        // load timestamp
        double timeDepth1 = timestampsDepth[frame];
        // get reference frame
        grayRef = loadIntensity(dataFolder + filesColor[frame]);
        depthRef = loadDepth(dataFolder + filesDepth[frame]);
        
        // initially copy current input images to level 0 for later use as reference
        cudaMemcpy(d_iPyCrr[0], (void*)grayRef.data, n*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;
        cudaMemcpy(d_dPyCrr[0], (void*)depthRef.data, n*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK;
        buildMapPyramids(d_iPyCrr, d_dPyCrr, lvlNum, w, h);
        
        buildDrvPyramids(d_iPyRef, d_dXPy, d_dYPy, lvlNum, w, h);

        // Pointer Switch reference <-> current
        switchTemp = d_iPyRef; d_iPyRef = d_iPyCrr; d_iPyCrr = switchTemp;
        switchTemp = d_dPyRef; d_dPyRef = d_dPyCrr; d_dPyCrr = switchTemp;

        if (!cublas) {
    		// align(depthCur, grayCur, xi, d_iPyRef, d_dPyRef, d_iPyCrr, d_dPyCrr, d_kPy, d_ikPy, d_dXPy, d_dYPy, d_res, d_J, d_redArr, d_redArr2, d_R, d_t, w, h, n, lvlNum, dth);
            align(depthCur, grayCur, xi, d_iPyRef, d_dPyRef, d_iPyCrr, d_dPyCrr, d_dXPy, d_dYPy, d_res, d_J, d_redArr, d_redArr2, d_R, d_t, w, h, n, lvlNum, dth);
        } else {
            alignCB(handle, depthCur, grayCur, xi, d_iPyRef, d_dPyRef, d_iPyCrr, d_dPyCrr, d_dXPy, d_dYPy, d_res, d_J, d_A, d_b, d_R, d_t, w, h, n, lvlNum, dth, d_redArr, d_redArr2);
        }
        
        Eigen::Matrix3f rot;
        Eigen::Vector3f t;
        convertSE3ToTf(xi, rot, t);
        // std::cout << "pose (xi) between frames " << (frame-1) << " and " << frame  << ": " << xi.transpose() << std::endl;
        // std::cout << "processed frame " << frame  << std::endl;

        // concatenate poses
        Eigen::Matrix4f relPose = Eigen::Matrix4f::Identity();
        relPose.topLeftCorner(3,3) = rot.cast<float>();
        relPose.topRightCorner(3,1) = t.cast<float>();
        // Eigen::Matrix4f relPoseInv = relPose.inverse();
        absPose = absPose * relPose;
        poses.push_back(absPose);
        timestamps.push_back(timeDepth1);

        // store current frame as next reference frame
        grayCur = grayRef;
        depthCur = depthRef;

    }
    timer2.end();  float timeElapsed2 = timer2.get();
    //#####  LOOP END  #####

    std::cout << "total aligning time of " << divisor << " frames: " << totTime << "ms" << std::endl
              << "avarage frame aligning time: " << totTime/divisor << "ms" << std::endl;
    std::cout << "total processing time of " << divisor << " frames: " << timeElapsed2*1000 << "ms" << std::endl
              << "avarage frame processing time: " << timeElapsed2*1000/divisor << "ms" << std::endl;

    cublasDestroy(handle);

    // free all the pyramids
    freeGPUMem(d_iPyRef, d_dPyRef, d_iPyCrr, d_dPyCrr, d_kPy, d_ikPy, d_dXPy, d_dYPy, d_res, d_J, lvlNum, w, h, d_R, d_t, d_redArr, d_redArr2, d_A, d_b);

    // delete the arrays
    delete[] d_iPyRef;
    delete[] d_dPyRef;
    delete[] d_iPyCrr;
    delete[] d_dPyCrr;
    delete[] d_kPy;
    delete[] d_ikPy;
    delete[] d_dXPy;
    delete[] d_dYPy;

    //#########################################################################
    //###### End ##############################################################
    //#########################################################################

    // save poses
    savePoses(dataFolder + "traj.txt", poses, timestamps);

	// close all opencv windows
	cvDestroyAllWindows();

	return 0;
}

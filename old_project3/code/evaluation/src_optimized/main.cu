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

#include "aux.h"
texture<float,2,cudaReadModeElementType> texRef;
texture<float,2,cudaReadModeElementType> texGradX;
texture<float,2,cudaReadModeElementType> texGradY;
#define LVL 4
#define ITER 10
#include "hostFunctions.hpp"
#include "downsample.cuh"
#include "deriveNumeric.cuh"
#include "pointCloud.cuh"
#include "residual.cuh"
#include "tum_benchmark.hpp"
#include "save_ply.hpp"

#define STR1(x)  #x
#define STR(x)  STR1(x)

typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;


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

    // initialize cuda context
    cudaDeviceSynchronize(); CUDA_CHECK;

    int maxFrames = -1;
    //maxFrames = 50;

    // process frames
    Eigen::Matrix4f absPose = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> poses;
    std::vector<double> timestamps;
    poses.push_back(absPose);
    timestamps.push_back(timestampsDepth[0]);

    cv::Mat colorPrev = loadColor(dataFolder + filesColor[0]);
    cv::Mat grayPrev;
    cv::cvtColor(colorPrev, grayPrev, CV_BGR2GRAY);
    cv::Mat depthPrev = loadDepth(dataFolder + filesDepth[0]);
    for (size_t i = 1; i < filesDepth.size() && (maxFrames < 0 || i < maxFrames); ++i)
    {
        // load input frame
        std::string fileColor1 = filesColor[i];
        std::string fileDepth1 = filesDepth[i];
        double timeDepth1 = timestampsDepth[i];
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

	//test

	{

	    // get image dimensions
	int w = grayRef.cols;         // width
	int h = grayRef.rows;         // height

	// initialize intrinsic matrix
	Eigen::Matrix3f K;
	K <<    517.3, 0.0, 318.6,
	    0.0, 516.5, 255.3,
	    0.0, 0.0, 1.0;


	// initial pose
	Eigen::Matrix3f rot;
	Eigen::Vector3f t;
	// Vec6f xi = Vec6f::Zero();
	convertSE3ToTf(xi, rot, t);

	//Saving the finest level of images
	std::vector<Eigen::Matrix3f> kPyramid;
	kPyramid.push_back(K);

	//Downsampling K
	for (int i = LVL-1; i > 0; i--)
	{
		K = scaleIntrinsic(K);
		kPyramid.push_back(K);
	}

	int nbytes = w*h*sizeof(float);
	float *d_curImg0, *d_curImg1, *d_curImg2, *d_curImg3;
	float *d_refImg0, *d_refImg1, *d_refImg2, *d_refImg3;
	float *d_depthcur0, *d_depthcur1, *d_depthcur2, *d_depthcur3;
	float *d_depthref0, *d_depthref1, *d_depthref2, *d_depthref3;
	float *d_resImg0, *d_resImg1, *d_resImg2, *d_resImg3;
	float *d_jacobif0, *d_jacobif1, *d_jacobif2, *d_jacobif3;	
	float *JtJ_final0, *JtJ_final1, *JtJ_final2, *JtJ_final3;
	float *Jtb_final0, *Jtb_final1, *Jtb_final2, *Jtb_final3;
	float *d_vx0, *d_vx1, *d_vx2, *d_vx3;
	float *d_vy0, *d_vy1, *d_vy2, *d_vy3;

	cudaMalloc(&d_jacobif0, w*h*6*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_jacobif1, w/2*h/2*6*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_jacobif2, w/4*h/4*6*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_jacobif3, w/8*h/8*6*sizeof(float));  CUDA_CHECK;


	cudaMalloc(&JtJ_final0, w*h*21*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&JtJ_final1, w/2*h/2*21*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&JtJ_final2, w/4*h/4*21*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&JtJ_final3, w/8*h/8*21*sizeof(float));  CUDA_CHECK;

	cudaMalloc(&Jtb_final0, w*h*6*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&Jtb_final1, w/2*h/2*6*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&Jtb_final2, w/4*h/4*6*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&Jtb_final3, w/8*h/8*6*sizeof(float));  CUDA_CHECK;

	cudaMalloc(&d_vx0,  w*h*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_vx1,  w/2*h/2*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_vx2,  w/4*h/4*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_vx3,  w/8*w/8*sizeof(float));  CUDA_CHECK;

	cudaMalloc(&d_vy0,  w*h*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_vy1,  w/2*h/2*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_vy2,  w/4*h/4*sizeof(float));  CUDA_CHECK;
	cudaMalloc(&d_vy3,  w/8*h/8*sizeof(float));  CUDA_CHECK;

	cudaMalloc(&d_resImg0,nbytes);  CUDA_CHECK; 
	cudaMemset(d_resImg0, 0, nbytes);  CUDA_CHECK;
	cudaMalloc(&d_resImg1,w/2*h/2*sizeof(float));   CUDA_CHECK; 
	cudaMemset(d_resImg1,0,w/2*h/2*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_resImg2,w/4*h/4*sizeof(float));    CUDA_CHECK; 
	cudaMemset(d_resImg2,0,w/4*h/4*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_resImg3,w/8*h/8*sizeof(float));   CUDA_CHECK; 
	cudaMemset(d_resImg3,0,w/8*h/8*sizeof(float)); CUDA_CHECK;

	cudaMalloc(&d_curImg0,nbytes);  CUDA_CHECK; 
	cudaMemcpy(d_curImg0, (void*)grayCur.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
	cudaMalloc(&d_curImg1,w/2*h/2*sizeof(float));   CUDA_CHECK; 
	cudaMemset(d_curImg1,0,w/2*h/2*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_curImg2,w/4*h/4*sizeof(float));    CUDA_CHECK; 
	cudaMemset(d_curImg2,0,w/4*h/4*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_curImg3,w/8*h/8*sizeof(float));   CUDA_CHECK; 
	cudaMemset(d_curImg3,0,w/8*h/8*sizeof(float)); CUDA_CHECK;

	cudaMalloc(&d_refImg0,nbytes);    CUDA_CHECK; 
	cudaMemcpy(d_refImg0, (void*)grayRef.data,nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
	cudaMalloc(&d_refImg1,w/2*h/2*sizeof(float));   CUDA_CHECK; 
	cudaMemset(d_refImg1,0,w/2*h/2*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_refImg2,w/4*h/4*sizeof(float));    CUDA_CHECK; 
	cudaMemset(d_refImg2,0,w/4*h/4*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_refImg3,w/8*h/8*sizeof(float));   CUDA_CHECK; 
	cudaMemset(d_refImg3,0,w/8*h/8*sizeof(float)); CUDA_CHECK;

	cudaMalloc(&d_depthcur0,nbytes);   CUDA_CHECK; 
	cudaMemcpy(d_depthcur0, (void*)depthCur.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK; 
	cudaMalloc(&d_depthcur1,w/2*h/2*sizeof(float));   CUDA_CHECK; 
	cudaMemset(d_depthcur1,0,w/2*h/2*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_depthcur2,w/4*h/4*sizeof(float));   CUDA_CHECK;  
	cudaMemset(d_depthcur2,0,w/4*h/4*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_depthcur3,w/8*h/8*sizeof(float));   CUDA_CHECK; 
	cudaMemset(d_depthcur3,0,w/8*h/8*sizeof(float)); CUDA_CHECK;

	cudaMalloc(&d_depthref0,nbytes);    CUDA_CHECK; 
	cudaMemcpy(d_depthref0, (void*)depthRef.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
	cudaMalloc(&d_depthref1,w/2*h/2*sizeof(float));   CUDA_CHECK;
	cudaMemset(d_depthref1,0,w/2*h/2*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_depthref2,w/4*h/4*sizeof(float));    CUDA_CHECK; 
	cudaMemset(d_depthref2,0,w/4*h/4*sizeof(float)); CUDA_CHECK;
	cudaMalloc(&d_depthref3,w/8*h/8*sizeof(float));  CUDA_CHECK; 
	cudaMemset(d_depthref3,0,w/8*h/8*sizeof(float)); CUDA_CHECK;

	std::vector<float *> JtJPyramid;
	JtJPyramid.push_back(JtJ_final0);
	JtJPyramid.push_back(JtJ_final1);
	JtJPyramid.push_back(JtJ_final2);
	JtJPyramid.push_back(JtJ_final3);

	std::vector<float *> jacobifPyramid;
	jacobifPyramid.push_back(d_jacobif0);
	jacobifPyramid.push_back(d_jacobif1);
	jacobifPyramid.push_back(d_jacobif2);
	jacobifPyramid.push_back(d_jacobif3);

	std::vector<float *> JtbPyramid;
	JtbPyramid.push_back(Jtb_final0);
	JtbPyramid.push_back(Jtb_final1);
	JtbPyramid.push_back(Jtb_final2);
	JtbPyramid.push_back(Jtb_final3);

	std::vector<float *> dvxPyramid;
	dvxPyramid.push_back(d_vx0);
	dvxPyramid.push_back(d_vx1);
	dvxPyramid.push_back(d_vx2);
	dvxPyramid.push_back(d_vx3);

	std::vector<float *> dvyPyramid;
	dvyPyramid.push_back(d_vy0);
	dvyPyramid.push_back(d_vy1);
	dvyPyramid.push_back(d_vy2);
	dvyPyramid.push_back(d_vy3);

	std::vector<float *> grayRefPyramid;
	grayRefPyramid.push_back(d_refImg0);
	grayRefPyramid.push_back(d_refImg1);
	grayRefPyramid.push_back(d_refImg2);
	grayRefPyramid.push_back(d_refImg3);

	std::vector<float *> depthRefPyramid;
	depthRefPyramid.push_back(d_depthref0);
	depthRefPyramid.push_back(d_depthref1);
	depthRefPyramid.push_back(d_depthref2);
	depthRefPyramid.push_back(d_depthref3);

	std::vector<float *> grayCurPyramid;
	grayCurPyramid.push_back(d_curImg0);
	grayCurPyramid.push_back(d_curImg1);
	grayCurPyramid.push_back(d_curImg2);
	grayCurPyramid.push_back(d_curImg3);

	std::vector<float *> depthCurPyramid;
	depthCurPyramid.push_back(d_depthcur0);
	depthCurPyramid.push_back(d_depthcur1);
	depthCurPyramid.push_back(d_depthcur2);
	depthCurPyramid.push_back(d_depthcur3);

	std::vector<float *> residualPyramid;
	residualPyramid.push_back(d_resImg0);
	residualPyramid.push_back(d_resImg1);
	residualPyramid.push_back(d_resImg2);
	residualPyramid.push_back(d_resImg3);

	// initialize cuda context
	cudaDeviceSynchronize();  CUDA_CHECK;
	downSample(grayRefPyramid, depthRefPyramid, grayCurPyramid, depthCurPyramid, w, h);

	float *result = new float[(size_t)1];
	float *A_interim = new float[(size_t)21];
	float *b_interim = new float[(size_t)6];
	Mat6f A = Mat6f::Zero();
	Vec6f b = Vec6f::Zero();

	Timer timer; timer.start();
	//updatexi(kPyramid, grayRefPyramid, depthRefPyramid, grayCurPyramid, rot, t, xi,w,h);
	for (int level = LVL; level>0 ; level--)
	{
		int width = w/pow(2,level-1);
		int height = h/pow(2,level-1);
		// initialize intrinsic matrix
		Eigen::Matrix3f kLevel = kPyramid[level-1];

		// initialize cuda context
		cudaDeviceSynchronize();  CUDA_CHECK;

		// copy data to device
		// Current grayscale image
		float *d_currImg = grayCurPyramid[level - 1] ;

		// Reference grayscale image
		float *d_refimgIn = grayRefPyramid[level -1];

		//Current depth image
		float *d_depthImgIn = depthCurPyramid[level-1];

		//Reference depth image
		float *d_refdepthImgIn = depthRefPyramid[level-1];

		//Residual Image
		float *d_resImg = residualPyramid[level-1];

		float *d_rot, *d_t;
		float fx = kLevel(0,0);
		float fy = kLevel(1,1);
		float cx = kLevel(0,2);
		float cy = kLevel(1,2); 

		size_t n_d_vx = (size_t)width*height;
		size_t n_d_vy = (size_t)width*height;
		int N = width*height;

		float *d_jacobif = jacobifPyramid[level-1];
		float *d_vx = dvxPyramid[level-1];
		float *d_vy = dvyPyramid[level-1];
		float *JtJ_final = JtJPyramid[level-1];
		float *Jtb_final = JtbPyramid[level-1];
		cudaMalloc(&d_rot,  9*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_t,  3*sizeof(float));  CUDA_CHECK;


		float errLast = std::numeric_limits<float>::max();
		for(int i = 0; i < ITER ; i++)
		{

			float *rot_data = rot.data();
			float *t_data = t.data();
	
			cudaMemset(d_vy, 0, n_d_vy*sizeof(float));  CUDA_CHECK;
			cudaMemset(d_vx, 0, n_d_vx*sizeof(float));  CUDA_CHECK;
			cudaMemset(Jtb_final, 0, N*6*sizeof(float));  CUDA_CHECK;
			cudaMemset(JtJ_final, 0, N*21*sizeof(float));  CUDA_CHECK;
			cudaMemset(d_jacobif, 0, N*6*sizeof(float));  CUDA_CHECK;
			cudaMemcpy(d_rot,rot_data,9*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
			cudaMemcpy(d_t,t_data,3*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;

			dim3 block = dim3(32, 8, 1);
			dim3 grid = dim3((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);

			//Texture Memory
			texRef.addressMode[0] = cudaAddressModeClamp;
			texRef.addressMode[1] = cudaAddressModeClamp;
			texRef.filterMode = cudaFilterModeLinear;
			texRef.normalized = false;
			cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
			cudaBindTexture2D(NULL, &texRef, d_currImg, &desc, width, height, width*sizeof(d_currImg[0]));
			 
			texGradX.addressMode[0] = cudaAddressModeClamp;
			texGradX.addressMode[1] = cudaAddressModeClamp;
			texGradX.filterMode = cudaFilterModeLinear;
			texGradX.normalized = false;
			cudaChannelFormatDesc descX = cudaCreateChannelDesc<float>();
			cudaBindTexture2D(NULL, &texGradX, d_vx, &descX, width, height, width*sizeof(d_vx[0]));

			texGradY.addressMode[0] = cudaAddressModeClamp;
			texGradY.addressMode[1] = cudaAddressModeClamp;
			texGradY.filterMode = cudaFilterModeLinear;
			texGradY.normalized = false;
			cudaChannelFormatDesc descY = cudaCreateChannelDesc<float>();
			cudaBindTexture2D(NULL, &texGradY, d_vy, &descY, width, height, width*sizeof(d_vy[0]));
	
			calcErr <<<grid,block>>> (d_refimgIn,d_currImg,d_refdepthImgIn,d_resImg,d_rot,d_t,fx,fy,cx,cy,width,height);
			CUDA_CHECK;
			gradCompute <<<grid,block>>> (d_currImg,d_vx,d_vy,width,height); CUDA_CHECK;
	
			deriveNumeric <<<grid,block>>>(d_vx,d_vy,d_refdepthImgIn,d_resImg,d_jacobif,width,height,fx,fy,cx,cy,d_rot,d_t,JtJ_final,Jtb_final);
			CUDA_CHECK;
	

			cv::Mat residualGPU(height,width,grayRef.type());
			cudaMemcpy((void *)residualGPU.data,d_resImg,N*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
			Eigen::VectorXf residual(N);
		    	int idx = 0;
		    	for(int i =0 ;i<width;i++)
			{
				for(int j =0 ;j<height;j++)
				{
					residual[idx] = residualGPU.at<float>(i,j);
					idx++;
				}
			}

			dim3 block1 = dim3(128, 1, 1);
			dim3 grid1 = dim3((N + block1.x -1)/block1.x,1,1);
			size_t smBytes = block1.x * block1.y * block1.z * sizeof(float);
			//Reduction for JtJ
			float* ptrJtJ = JtJ_final;		
			for(int j = 0; j<21; j++)
			{
				block_sum <<<grid1,block1,smBytes>>> (ptrJtJ,ptrJtJ,N);
				for(int offset = block1.x / 2;offset > 0;offset >>= 1)
					block_sum <<<grid1,block1,smBytes>>> (ptrJtJ,ptrJtJ,N);
				cudaMemcpy(result,ptrJtJ,sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
				A_interim[j]= result[0];
				ptrJtJ = ptrJtJ + N;	
			}
	
			int k = 0;	
			for(int i = 0; i<6; i++)
			{
				for(int j = i; j<6; j++)
				{
					A(i,j) = A(j,i) =A_interim[k];	
					k++;
				}
			}

			float *ptrJtb = Jtb_final;
			for(int j = 0; j<6; j++)
			{
				block_sum <<<grid1,block1,smBytes>>> (ptrJtb,ptrJtb,N);
				for(int offset = block1.x / 2;offset > 0;offset >>= 1)
					block_sum <<<grid1,block1,smBytes>>> (ptrJtb,ptrJtb,N);
				cudaMemcpy(result,ptrJtb,sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
				b_interim[j]= result[0];
				ptrJtb = ptrJtb + N;
			}

			for(int i = 0; i<6; i++)
				b(i) = b_interim[i];
	
			// solve using Cholesky LDLT decomposition
			Vec6f delta = -(A.ldlt().solve(b));

			// update xi
			xi = Sophus::SE3f::log(Sophus::SE3f::exp(delta)*Sophus::SE3f::exp(xi));

			cudaUnbindTexture(texRef);
			cudaUnbindTexture(texGradX);
			cudaUnbindTexture(texGradY);
	
			// print out final pose
			convertSE3ToTf(xi, rot, t);
			/*std::cout << "xi = " << xi.transpose() << std::endl;
			Eigen::VectorXd xiResult(6);
			xiResult << -0.0021f, 0.0057f, 0.0374f, -0.0292f, -0.0183f, -0.0009f;
			std::cout << "xi expected = " << xiResult.transpose() << std::endl;*/
	
			float error = (residual.cwiseProduct(residual)).mean();	
			if((error/errLast) > 0.995f)
				break;
			errLast = error;
	
	
		}

			cudaFree(d_rot); CUDA_CHECK;
			cudaFree(d_t); CUDA_CHECK;

			width = width * 2;
			height = height * 2;
		}

		timer.end(); float timeElapsed = timer.get();
		std::cout << "Time: " << timeElapsed * 1000 << std::endl;

		delete[] result;
		delete[] A_interim;
		delete[] b_interim;

		cudaFree(d_curImg0); CUDA_CHECK;
		cudaFree(d_curImg1); CUDA_CHECK;
		cudaFree(d_curImg2); CUDA_CHECK;
		cudaFree(d_curImg3); CUDA_CHECK;

		cudaFree(d_depthref0); CUDA_CHECK;
		cudaFree(d_depthref1); CUDA_CHECK;
		cudaFree(d_depthref2); CUDA_CHECK;
		cudaFree(d_depthref3); CUDA_CHECK;

		cudaFree(d_refImg0); CUDA_CHECK;
		cudaFree(d_refImg1); CUDA_CHECK;
		cudaFree(d_refImg2); CUDA_CHECK;
		cudaFree(d_refImg3); CUDA_CHECK;

		cudaFree(d_depthcur0); CUDA_CHECK;
		cudaFree(d_depthcur1); CUDA_CHECK;
		cudaFree(d_depthcur2); CUDA_CHECK;
		cudaFree(d_depthcur3); CUDA_CHECK;

		cudaFree(d_resImg0); CUDA_CHECK;
		cudaFree(d_resImg1); CUDA_CHECK;
		cudaFree(d_resImg2); CUDA_CHECK;
		cudaFree(d_resImg3); CUDA_CHECK;

		cudaFree(d_vx0); CUDA_CHECK;
		cudaFree(d_vx1); CUDA_CHECK;
		cudaFree(d_vx2); CUDA_CHECK;
		cudaFree(d_vx3); CUDA_CHECK;

		cudaFree(d_vy0); CUDA_CHECK;
		cudaFree(d_vy1); CUDA_CHECK;
		cudaFree(d_vy2); CUDA_CHECK;
		cudaFree(d_vy3); CUDA_CHECK;

		cudaFree(d_jacobif0); CUDA_CHECK;
		cudaFree(d_jacobif1); CUDA_CHECK;
		cudaFree(d_jacobif2); CUDA_CHECK;
		cudaFree(d_jacobif3); CUDA_CHECK;

		cudaFree(JtJ_final0); CUDA_CHECK;
		cudaFree(JtJ_final1); CUDA_CHECK;
		cudaFree(JtJ_final2); CUDA_CHECK;
		cudaFree(JtJ_final3); CUDA_CHECK;

		cudaFree(Jtb_final0); CUDA_CHECK;
		cudaFree(Jtb_final1); CUDA_CHECK;
		cudaFree(Jtb_final2); CUDA_CHECK;
		cudaFree(Jtb_final3); CUDA_CHECK;

	}

        Eigen::Matrix3f rot;
        Eigen::Vector3f t;
        convertSE3ToTf(xi, rot, t);
        std::cout << "pose (xi) between frames " << (i-1) << " and " << i  << ": " << xi.transpose() << std::endl;

        // concatenate poses
        Eigen::Matrix4f relPose = Eigen::Matrix4f::Identity();
        relPose.topLeftCorner(3,3) = rot;
        relPose.topRightCorner(3,1) = t;
        absPose = absPose * relPose.inverse();
        poses.push_back(absPose);
        timestamps.push_back(timeDepth1);

#if 0
        // save frames as point cloud
        rot = absPose.topLeftCorner(3,3);
        t = absPose.topRightCorner(3,1);
        cv::Mat vertexMap1;
        depthToVertexMap(K.cast<double>(), depth1, vertexMap1);
        transformVertexMap(rot.cast<double>(), t.cast<double>(), vertexMap1);
        cv::Mat color1UC;
        color1.convertTo(color1UC, CV_8UC3, 255.0f);
        std::stringstream ss;
        ss << dataFolder << "cloud_" << std::setw(4) << std::setfill('0') << i << ".ply";
        savePlyFile(ss.str(), color1UC, vertexMap1);
#endif

        colorPrev = color1;
        depthPrev = depth1;
        grayPrev = gray1;
    }
    
    // save poses
    savePoses(dataFolder + "traj.txt", poses, timestamps);

    // clean up
    cv::destroyAllWindows();
    std::cout << "Direct Image Alignment finished." << std::endl;
    return 0;
}


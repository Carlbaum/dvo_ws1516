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
#define LVL 4
#define ITER 10
#include "aux.h"

#include "downsample.cuh"
#include "hostFunctions.hpp"
#include "pointCloud.cuh"
#include "deriveNumeric.cuh"
#include "residual.cuh"
#include "tum_benchmark.hpp"
//#include "save_ply.hpp"

#define STR1(x)  #x
#define STR(x)  STR1(x)


typedef Eigen::Matrix<float, 6, 6> Mat6f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

void align(const cv::Mat &depthRefIn, const cv::Mat &grayRefIn, const cv::Mat &depthCurIn, const cv::Mat &grayCurIn, Vec6f& xi)
{
    // get image dimensions
    int w = grayRefIn.cols;         // width
    int h = grayRefIn.rows;         // height

    // initialize intrinsic matrix
    Eigen::Matrix3f K;
    K <<    517.3, 0.0, 318.6,
            0.0, 516.5, 255.3,
            0.0, 0.0, 1.0;

    // initial pose
    Eigen::Matrix3f rot;
    Eigen::Vector3f t;
    convertSE3ToTf(xi, rot, t);

    //Saving the finest level of images
    std::vector<Eigen::Matrix3f> kPyramid;
    kPyramid.push_back(K);
    std::vector<cv::Mat> grayRefPyramid;
    grayRefPyramid.push_back(grayRefIn);
    std::vector<cv::Mat> depthRefPyramid;
    depthRefPyramid.push_back(depthRefIn);
    std::vector<cv::Mat> grayCurPyramid;
    grayCurPyramid.push_back(grayCurIn);
    std::vector<cv::Mat> depthCurPyramid;
    depthCurPyramid.push_back(depthCurIn);
    // initialize cuda context
    cudaDeviceSynchronize();  CUDA_CHECK;

    // copy data to device
    float nbytes = w*h*sizeof(float);
    float *d_imgIn;
    cudaMalloc(&d_imgIn, nbytes);  CUDA_CHECK;
    cudaMemcpy(d_imgIn, (void*)grayCurIn.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
    float *d_refimgIn;
    cudaMalloc(&d_refimgIn, nbytes);  CUDA_CHECK;
    cudaMemcpy(d_refimgIn, (void*)grayRefIn.data,nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
    float *d_depthImgIn;
    cudaMalloc(&d_depthImgIn, nbytes);  CUDA_CHECK;
    cudaMemcpy(d_depthImgIn, (void*)depthCurIn.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
    float *d_refdepthImgIn;
    cudaMalloc(&d_refdepthImgIn, nbytes);  CUDA_CHECK;
    cudaMemcpy(d_refdepthImgIn, (void*)depthRefIn.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;

    for(int i = 1; i <LVL ; i++)
    {
        
       
    	w = w/2, h = h/2;
    	// Output graycurr image
        float nbytes_scaled = w*h*sizeof(float);
    	float *d_imgOut;
    	cudaMalloc(&d_imgOut, nbytes_scaled);  CUDA_CHECK;
    	cudaMemset(d_imgOut, 0, nbytes_scaled);  CUDA_CHECK;
	
	// Output gray reference image
    	float *d_refimgOut;
    	cudaMalloc(&d_refimgOut, nbytes_scaled);  CUDA_CHECK;
    	cudaMemset(d_refimgOut, 0, nbytes_scaled);  CUDA_CHECK;
    
 	//Output depth current image
    	float *d_depthImgOut;
    	cudaMalloc(&d_depthImgOut, nbytes_scaled);  CUDA_CHECK;
    	cudaMemset(d_depthImgOut, 0, nbytes_scaled);  CUDA_CHECK;
	
	//Output refernce depth image
    	float *d_refdepthImgOut;
    	cudaMalloc(&d_refdepthImgOut, nbytes_scaled);  CUDA_CHECK;
    	cudaMemset(d_refdepthImgOut, 0, nbytes_scaled);  CUDA_CHECK;

    	// execute kernel
    	dim3 block = dim3(32, 8, 1);
    	dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);
         
 	downSampleGray <<<grid,block>>> (d_imgOut,d_imgIn,h,w); CUDA_CHECK;  	//Gray current image
	downSampleGray <<<grid,block>>> (d_refimgOut,d_refimgIn,h,w); CUDA_CHECK;  	//Reference image
	downSampleDepth <<<grid,block>>> (d_depthImgOut,d_depthImgIn,h,w);	//Current depth image
	downSampleDepth <<<grid,block>>> (d_refdepthImgOut,d_refdepthImgIn,h,w);	//Reference depth image
        K = scaleIntrinsic(K);	
	kPyramid.push_back(K);

    	cudaDeviceSynchronize();
    	

	cv::Mat mOut(h,w,grayCurIn.type()); 
	cv::Mat refmOut(h,w,grayCurIn.type());
	cv::Mat depth_mOut(h,w,grayCurIn.type());
	cv::Mat refdepth_mOut(h,w,grayCurIn.type()); 
  
    	// copy data back to host
    	cudaMemcpy((void *)mOut.data, d_imgOut, nbytes_scaled , cudaMemcpyDeviceToHost);  CUDA_CHECK;
	grayCurPyramid.push_back(mOut);
    	cudaFree(d_imgOut);  CUDA_CHECK;
	
	cudaMemcpy((void *)refmOut.data, d_refimgOut,nbytes_scaled , cudaMemcpyDeviceToHost);  CUDA_CHECK;
	grayRefPyramid.push_back(refmOut);
    	cudaFree(d_refimgOut);  CUDA_CHECK;
	
	cudaMemcpy((void *)depth_mOut.data, d_depthImgOut, nbytes_scaled , cudaMemcpyDeviceToHost);  CUDA_CHECK;
	depthCurPyramid.push_back(depth_mOut);
    	cudaFree(d_depthImgOut);  CUDA_CHECK;
	
	cudaMemcpy((void *)refdepth_mOut.data, d_refdepthImgOut, nbytes_scaled , cudaMemcpyDeviceToHost);  CUDA_CHECK;
	depthRefPyramid.push_back(refdepth_mOut);
    	cudaFree(d_refdepthImgOut);  CUDA_CHECK;
      
        cudaFree(d_imgIn); CUDA_CHECK;
	cudaFree(d_refimgIn); CUDA_CHECK;
	cudaFree(d_depthImgIn); CUDA_CHECK;
	cudaFree(d_refdepthImgIn); CUDA_CHECK;

	// copy current output to next input
        cudaMalloc(&d_depthImgIn,  nbytes_scaled );  CUDA_CHECK;
        cudaMemcpy(d_depthImgIn, (void *)depth_mOut.data, nbytes_scaled , cudaMemcpyHostToDevice);  CUDA_CHECK;

	cudaMalloc(&d_refdepthImgIn, nbytes_scaled );  CUDA_CHECK;
        cudaMemcpy(d_refdepthImgIn, (void *)refdepth_mOut.data,nbytes_scaled , cudaMemcpyHostToDevice);  CUDA_CHECK;

	cudaMalloc(&d_refimgIn,  nbytes_scaled );  CUDA_CHECK;
        cudaMemcpy(d_refimgIn, (void *)refmOut.data, nbytes_scaled , cudaMemcpyHostToDevice);  CUDA_CHECK;

	cudaMalloc(&d_imgIn,  nbytes_scaled );  CUDA_CHECK;
        cudaMemcpy(d_imgIn, (void *)mOut.data, nbytes_scaled , cudaMemcpyHostToDevice);  CUDA_CHECK;
    }
    cudaFree(d_imgIn);  CUDA_CHECK;
    cudaFree(d_refimgIn);  CUDA_CHECK;
    cudaFree(d_depthImgIn);  CUDA_CHECK;
    cudaFree(d_refdepthImgIn);  CUDA_CHECK;

    for (int level = LVL; level>0 ; level--)
    {
	cv::Mat grayRef = grayRefPyramid[level-1];
	cv::Mat depthRef = depthRefPyramid[level-1];
	cv::Mat grayCur = grayCurPyramid[level-1];
	cv::Mat depthCur = depthCurPyramid[level-1];
	Eigen::Matrix3f kLevel = kPyramid[level-1];


	// get image dimensions
	int w = grayRef.cols;         // width
	int h = grayRef.rows;         // height

	// initialize cuda context
	cudaDeviceSynchronize();  CUDA_CHECK;

	// copy data to device
	// Current grayscale image
	float nbytes = w*h*sizeof(float);
	float *d_currImg;
	cudaMalloc(&d_currImg, nbytes);  CUDA_CHECK;
	cudaMemcpy(d_currImg, (void*)grayCur.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;

	// Reference grayscale image
	float *d_refimgIn;
	cudaMalloc(&d_refimgIn, nbytes);  CUDA_CHECK;
	cudaMemcpy(d_refimgIn, (void*)grayRef.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;

	//Current depth image
	float *d_depthImgIn;
	cudaMalloc(&d_depthImgIn,  nbytes);  CUDA_CHECK;
	cudaMemcpy(d_depthImgIn, (void*)depthCur.data,nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;

	//Reference depth image
	float *d_refdepthImgIn;
	cudaMalloc(&d_refdepthImgIn, nbytes);  CUDA_CHECK;
	cudaMemcpy(d_refdepthImgIn, (void*)depthRef.data,nbytes,cudaMemcpyHostToDevice);  CUDA_CHECK;

	//Residual Image
	float *d_resImg;
	cudaMalloc(&d_resImg,  nbytes);  CUDA_CHECK;
	cudaMemset(d_resImg, 0, nbytes);  CUDA_CHECK;
	
	float *d_rot, *d_t;
	float fx = kLevel(0,0);
	float fy = kLevel(1,1);
	float cx = kLevel(0,2);
	float cy = kLevel(1,2); 
	int N = w*h;
		float *JTJ  = new float[(size_t)N*21];
		float *JTB  = new float[(size_t)N*6];
		float *temp = new float[(size_t)N];
		float *result = new float[(size_t)1];
		float *A_interim = new float[(size_t)21];
		float *b_interim = new float[(size_t)6];
		Mat6f A = Mat6f::Zero();
		Vec6f b = Vec6f::Zero();
	float errLast = std::numeric_limits<float>::max();
	for(int i = 0; i < ITER ; i++)
	{
		float *rot_data = rot.data();
		float *t_data = t.data();
		
		float *d_vx, *d_vy, *d_jacobif,*JtJ_final, *Jtb_final, *d_temp, *d_result;
		size_t n_d_vx = (size_t)w*h;
		size_t n_d_vy = (size_t)w*h;
	

		cudaMalloc(&d_jacobif, N*6*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_jacobif, 0, N*6*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_temp, N*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_temp, 0, N*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_result, N*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_result, 0, N*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&JtJ_final, N*21*sizeof(float));  CUDA_CHECK;
		cudaMemset(JtJ_final, 0, N*21*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&Jtb_final, N*6*sizeof(float));  CUDA_CHECK;
		cudaMemset(Jtb_final, 0, N*6*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_vx,  n_d_vx*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_vx, 0, n_d_vx*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_vy,  n_d_vy*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_vy, 0, n_d_vy*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_rot,  9*sizeof(float));  CUDA_CHECK;
		cudaMemcpy(d_rot,rot_data,9*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;
		cudaMalloc(&d_t,  3*sizeof(float));  CUDA_CHECK;
		cudaMemcpy(d_t,t_data,3*sizeof(float),cudaMemcpyHostToDevice); CUDA_CHECK;

		dim3 block = dim3(32, 8, 1);
		dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);

		//Texture Memory
		texRef.addressMode[0] = cudaAddressModeClamp;
		texRef.addressMode[1] = cudaAddressModeClamp;
		texRef.filterMode = cudaFilterModeLinear;
		texRef.normalized = false;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(NULL, &texRef, d_currImg, &desc, w, h, w*sizeof(d_currImg[0]));
		 
		texGradX.addressMode[0] = cudaAddressModeClamp;
		texGradX.addressMode[1] = cudaAddressModeClamp;
		texGradX.filterMode = cudaFilterModeLinear;
		texGradX.normalized = false;
		cudaChannelFormatDesc descX = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(NULL, &texGradX, d_vx, &descX, w, h, w*sizeof(d_vx[0]));

		texGradY.addressMode[0] = cudaAddressModeClamp;
		texGradY.addressMode[1] = cudaAddressModeClamp;
		texGradY.filterMode = cudaFilterModeLinear;
		texGradY.normalized = false;
		cudaChannelFormatDesc descY = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(NULL, &texGradY, d_vy, &descY, w, h, w*sizeof(d_vy[0]));

		//cudaMemset(d_imgOut, 0, nbytes);  CUDA_CHECK;
		cudaMemset(d_resImg, 0, nbytes);  CUDA_CHECK;

		
		calcErr <<<grid,block>>> (d_refimgIn,d_currImg,d_refdepthImgIn,d_resImg,d_rot,d_t,fx,fy,cx,cy,w,h);
		CUDA_CHECK;
		gradCompute <<<grid,block>>> (d_currImg,d_vx,d_vy,w,h); CUDA_CHECK;
		
		deriveNumeric <<<grid,block>>>(d_vx,d_vy,d_refdepthImgIn,d_resImg,d_jacobif,w,h,fx,fy,cx,cy,d_rot,d_t,JtJ_final,Jtb_final); 
		CUDA_CHECK;
		
		cv::Mat residualGPU(h,w,grayRef.type());
		cudaMemcpy((void *)residualGPU.data,d_resImg,N*sizeof(float),cudaMemcpyDeviceToHost); CUDA_CHECK;
		Eigen::VectorXf residual(N);
	    	
	    	int idx = 0;
	    	for(int i =0 ;i<w;i++)
    		{
			for(int j =0 ;j<h;j++)
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
		
		
		
		//Reduction for Jtb
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
		cudaFree(d_rot); CUDA_CHECK;
		cudaFree(d_t); CUDA_CHECK;
		cudaFree(d_vx); CUDA_CHECK;
		cudaFree(d_vy); CUDA_CHECK;
		cudaFree(d_jacobif); CUDA_CHECK;
		cudaFree(d_result); CUDA_CHECK;
		cudaFree(d_temp); CUDA_CHECK;
		cudaFree(JtJ_final); CUDA_CHECK;
		cudaFree(Jtb_final); CUDA_CHECK;
	
	
		// print out final pose
		convertSE3ToTf(xi, rot, t);
		
		float error = (residual.cwiseProduct(residual)).mean();	
		if((error/errLast) > 0.995f)
			break;
		errLast = error;

		
		
	}
        delete[] JTJ;
	delete[] JTB;
	delete[] temp;
	delete[] result;
	delete[] A_interim;
	delete[] b_interim;

	cudaFree(d_currImg); CUDA_CHECK;
	cudaFree(d_refimgIn); CUDA_CHECK;
	cudaFree(d_depthImgIn); CUDA_CHECK;
	cudaFree(d_refdepthImgIn); CUDA_CHECK;
	cudaFree(d_resImg); CUDA_CHECK;

}
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
        Timer timer; timer.start();
        align(depthRef, grayRef, depthCur, grayCur, xi);
	timer.end(); float time = timer.get();
	std::cout <<"Time : " << time*1000 << std::endl;

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


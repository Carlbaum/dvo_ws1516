#ifndef DOWNSAMPLE_H
#define DOWNSAMPLE_H

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/core.hpp>


Eigen::Matrix3f scaleIntrinsic(Eigen::Matrix3f K)
{
	K(0,0) /= 2.0f;
	K(1,1) /= 2.0f;
	K(0,2) = ((K(0,2) + 0.5f)/2.0f) - 0.5f;
	K(1,2) = ((K(1,2) + 0.5f)/2.0f) - 0.5f;
	return K;
}
__global__ void downSampleGray(float *imgOut, float *imgIn,int h, int w)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    //check if within bounds
    if (x < w && y < h)
    {
	size_t idx = x + (size_t)w*y;
        size_t id = 2*x + 2*y*2*w;
	
	imgOut[idx] += 0.25f * (imgIn[id] + imgIn[id + 1] + imgIn[id + 2*w] + imgIn[id + 1 + 2*w]); 
    }
}


__global__ void downSampleDepth(float *imgOut, float *imgIn,int h, int w)
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    //check if within bounds
    if (x < w && y < h)
    {
	size_t idx = x + (size_t)w*y;
	if(2*x < 2*w && 2*y < 2*h)
	{
        	size_t id = 2*x + 2*y*2*w;
        	float N = 0;
       	
		if (imgIn[id] == 0 && imgIn[id + 1] == 0 && imgIn[id + 2*w] == 0 && imgIn[id + 1 + 2*w] == 0)
		{      
			imgOut[idx] = 0; 
		}
		else
		{ 
		if ( imgIn[id] != 0) N+= 1.0f;
		if ( imgIn[id] != 0) N+= 1.0f;
		if ( imgIn[id] != 0) N+= 1.0f;
		if ( imgIn[id] != 0) N+= 1.0f;
	
		imgOut[idx] += 1.0f/N * (imgIn[id] + imgIn[id + 1] + imgIn[id + 2*w] + imgIn[id + 1 +2*w]); 
		N = 0.0f;
		}
	}		
    }
}

void downSample(int levels,std::vector<Eigen::Matrix3f> &kPyramid,std::vector<cv::Mat> &grayRefPyramid,
		std::vector<cv::Mat> &depthRefPyramid, std::vector<cv::Mat> &grayCurPyramid)
{
	cv::Mat grayRef = grayRefPyramid[0];
	cv::Mat depthRef = depthRefPyramid[0];
	cv::Mat grayCur = grayCurPyramid[0];
	Eigen::Matrix3f K = kPyramid[0];
	
	int w = grayRef.cols;
	int h = grayRef.rows;
	// copy data to device
	float nbytes = w*h*sizeof(float);
	float *d_imgIn;
	cudaMalloc(&d_imgIn, nbytes);  CUDA_CHECK;
	cudaMemcpy(d_imgIn, (void*)grayCur.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
	float *d_refimgIn;
	cudaMalloc(&d_refimgIn, nbytes);  CUDA_CHECK;
	cudaMemcpy(d_refimgIn, (void*)grayRef.data,nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
	float *d_refdepthImgIn;
	cudaMalloc(&d_refdepthImgIn, nbytes);  CUDA_CHECK;
	cudaMemcpy(d_refdepthImgIn, (void*)depthRef.data, nbytes, cudaMemcpyHostToDevice);  CUDA_CHECK;
	
	for(int i = 1; i < levels ; i++)
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

		//Output refernce depth image
		float *d_refdepthImgOut;
		cudaMalloc(&d_refdepthImgOut, nbytes_scaled);  CUDA_CHECK;
		cudaMemset(d_refdepthImgOut, 0, nbytes_scaled);  CUDA_CHECK;

		// execute kernel
		dim3 block = dim3(32, 8, 1);
		dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);
		Timer timer; timer.start();

		downSampleGray <<<grid,block>>> (d_imgOut,d_imgIn,h,w); CUDA_CHECK;  		//Gray current image
		downSampleGray <<<grid,block>>> (d_refimgOut,d_refimgIn,h,w); CUDA_CHECK;  	//Reference image
		downSampleDepth <<<grid,block>>> (d_refdepthImgOut,d_refdepthImgIn,h,w);	//Reference depth image
		K = scaleIntrinsic(K);	
		std::cout<< K<<std::endl;
		kPyramid.push_back(K);

		cudaDeviceSynchronize();
		timer.end();  float timeElapsed = timer.get();  // elapsed time in seconds
		std::cout << "time: " << timeElapsed*1000 << " ms" << std::endl;

		cv::Mat mOut(h,w,grayRef.type()); 
		cv::Mat refmOut(h,w,grayRef.type());
		cv::Mat refdepth_mOut(h,w,grayRef.type()); 

		// copy data back to host
		cudaMemcpy((void *)mOut.data, d_imgOut, nbytes_scaled , cudaMemcpyDeviceToHost);  CUDA_CHECK;
		grayCurPyramid.push_back(mOut);
		cudaFree(d_imgOut);  CUDA_CHECK;

		cudaMemcpy((void *)refmOut.data, d_refimgOut,nbytes_scaled , cudaMemcpyDeviceToHost);  CUDA_CHECK;
		grayRefPyramid.push_back(refmOut);
		cudaFree(d_refimgOut);  CUDA_CHECK;

		cudaMemcpy((void *)refdepth_mOut.data, d_refdepthImgOut, nbytes_scaled , cudaMemcpyDeviceToHost);  CUDA_CHECK;
		depthRefPyramid.push_back(refdepth_mOut);
		cudaFree(d_refdepthImgOut);  CUDA_CHECK;

		cudaFree(d_imgIn); CUDA_CHECK;
		cudaFree(d_refimgIn); CUDA_CHECK;
		cudaFree(d_refdepthImgIn); CUDA_CHECK;

		cudaMalloc(&d_refdepthImgIn, nbytes_scaled );  CUDA_CHECK;
		cudaMemcpy(d_refdepthImgIn, (void *)refdepth_mOut.data,nbytes_scaled , cudaMemcpyHostToDevice);  CUDA_CHECK;

		cudaMalloc(&d_refimgIn,  nbytes_scaled );  CUDA_CHECK;
		cudaMemcpy(d_refimgIn, (void *)refmOut.data, nbytes_scaled , cudaMemcpyHostToDevice);  CUDA_CHECK;

		cudaMalloc(&d_imgIn,  nbytes_scaled );  CUDA_CHECK;
		cudaMemcpy(d_imgIn, (void *)mOut.data, nbytes_scaled , cudaMemcpyHostToDevice);  CUDA_CHECK;
	}
	cudaFree(d_imgIn);  CUDA_CHECK;
	cudaFree(d_refimgIn);  CUDA_CHECK;
	cudaFree(d_refdepthImgIn);  CUDA_CHECK;
}

#endif //DOWNSAMPLE_H


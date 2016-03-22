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

void downSample(std::vector<float *> &grayRefPyramid, std::vector<float *> &depthRefPyramid,std::vector<float *> &grayCurPyramid,
                std::vector<float *> &depthCurPyramid, int w, int h)
{
   int width = w, height = h;
    for(int i = 1; i <LVL ; i++)
    {
        
    	width = width/2, height = height/2;
    	// Output graycurr image
    	float *d_curOut = grayCurPyramid[i];
	
	// Output gray reference image
    	float *d_refimgOut = grayRefPyramid[i];
    
 	//Output depth current image
    	float *d_depthImgOut = depthCurPyramid[i];
	
	//Output refernce depth image
    	float *d_refdepthImgOut = depthRefPyramid[i];

    	// execute kernel
    	dim3 block = dim3(32, 8, 1);
    	dim3 grid = dim3((width+block.x-1)/block.x, (height+block.y-1)/block.y, 1);
        Timer timer; timer.start();
        
 	downSampleGray <<<grid,block>>> (d_curOut,grayCurPyramid[i-1],height,width); CUDA_CHECK;  	//Gray current image
	downSampleGray <<<grid,block>>> (d_refimgOut,grayRefPyramid[i-1],height,width); CUDA_CHECK;  	//Reference image
	downSampleDepth <<<grid,block>>> (d_depthImgOut,depthCurPyramid[i-1],height,width);	//Current depth image
	downSampleDepth <<<grid,block>>> (d_refdepthImgOut,depthRefPyramid[i-1],height,width);	//Reference depth image

    	cudaDeviceSynchronize();
    	timer.end();  float timeElapsed = timer.get();  // elapsed time in seconds

    }
}

#endif //DOWNSAMPLE_H


#ifndef POINTCLOUD_H
#define POINTCLOUD_H


#include <cuda_runtime.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <sstream>

__device__ void calPointCloud(float d_refdepthImg, float fx, float fy, float cx, float cy, float *rot, float *t, float (&pTrans)[3],int w, int h)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x; 
        int y = threadIdx.y + blockDim.y*blockIdx.y;
	if(x<w && y<h)
	{
		if(d_refdepthImg == 0) return;
		float p[3];
		p[0] = ((float)x -cx)/fx * d_refdepthImg;
		p[1] = ((float)y -cy)/fy * d_refdepthImg;
		p[2] = d_refdepthImg;
			
	     	// Rot * p
		//float p_new[3];
		for (int i = 0; i<3; i++)
		{
			pTrans[i] = 0.0f;
			for(int j = 0; j<3; j++)
			{
				pTrans[i] += rot[i+j*3]*p[j];
			}
	
		}

		// rOT * P + T
		for (int i = 0; i<3; i++)
		{
			pTrans[i]+=t[i];
		}

		return;
	}
}

#endif //POINTCLOUD_H

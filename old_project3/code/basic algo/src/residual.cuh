
#include <cuda_runtime.h>
#include "pointCloud.cuh"

__global__ void calcErr(float *d_refImg, float *d_currImg, float *d_refdepthImg, float *resImg,float *rot,float *t,float fx, float fy, float cx, float cy, int w, int h)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	if(x<w && y<h)
	{
		int idx = x + y * w;
		float ref_depth = d_refdepthImg[idx];
		if(ref_depth == 0) return;
		float pTrans[3] = {0,0,0};
		calPointCloud(ref_depth,fx,fy,cx,cy,rot,t,pTrans,w,h);
		if(pTrans[2] == 0.0f) return;
		
		float u = (pTrans[0]*fx / pTrans[2])+cx + 0.5f;
		float v = (pTrans[1]*fy / pTrans[2])+cy + 0.5f;		
		
		if(u >= 0.0f && u < (float)w && v < (float)h && v >=0.0f)
		{
			resImg[idx] = d_refImg[idx] - tex2D(texRef,u,v);
		}
	}
}


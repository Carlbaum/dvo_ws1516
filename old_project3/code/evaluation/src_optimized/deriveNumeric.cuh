#include <cuda_runtime.h>
#include "pointCloud.cuh"
__global__ void block_sum(float *input, float *results, size_t n)
{
	extern __shared__ float sdata[];
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int tx = threadIdx.x;
	
	//loading data into shared memory
	
	float x = 0;
	if ( i<n)
		x = input[i];
	sdata[tx] = x;
        
	__syncthreads();

        //results[i] = sdata[tx];
	// block-wide reduction in __shared__ mem
	for(int offset = blockDim.x / 2;offset > 0;offset >>= 1)
	{
		if(tx < offset)
		{
		// add a partial sum upstream to our own
		sdata[tx] += sdata[tx + offset];
		}
		__syncthreads();
	}
	if(threadIdx.x == 0)
	{
		// note that the result is per-block
		// not per-thread
		results[blockIdx.x] = sdata[0];
	}
}

__device__ void calJacobian(float (&d_jacobian)[6], float *d_JtJ, float *Jtb, float b,float dX, float dY, float (&pt3)[3],int w,int h)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	if(x<w && y<h)
	{
		d_jacobian[0] = dX / pt3[2];
		d_jacobian[1] = dY / pt3[2];
		d_jacobian[2] = - (dX * pt3[0] + dY * pt3[1]) / (pt3[2] * pt3[2]);
		d_jacobian[3] = - (dX * pt3[0] * pt3[1]) / (pt3[2] * pt3[2]) - dY * (1 + (pt3[1] / pt3[2]) * (pt3[1] / pt3[2]));
		d_jacobian[4] = + dX * (1.0 + (pt3[0] / pt3[2]) * (pt3[0] / pt3[2])) + (dY * pt3[0] * pt3[1]) / (pt3[2] * pt3[2]);
		d_jacobian[5] = (- dX * pt3[1] + dY * pt3[0]) / pt3[2];

            //Update the n*6 matrix
            for(int i = 0; i < 6; ++i)
                d_jacobian[i] = -d_jacobian[i];
	
		//Calculating Jt*J
		int k = 0;
		for(int i =0; i<6; i++)
		{
			for(int j = i; j<6; j++)
			{
				d_JtJ[k*w*h+y*w+x] = d_jacobian[j]*d_jacobian[i];
				k++;
			}
		}
		
		//Calculating Jt*b
		for(int i = 0; i<6; i++)
			Jtb[i*w*h+y*w+x] = d_jacobian[i]*b;
	}
}



__global__ void deriveNumeric(float *d_vx, float *d_vy, float *d_refdepthImg,float *d_residual,float *d_jacobif,int w, int h,  float fx, float fy, float cx, float cy, float *rot, float *t, float *JtJ_final,float *Jtb_final)
{	
        //int j = 0;
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	if(x<w && y<h)
	{
		int idx = x + y*w;
		float ref_depth = d_refdepthImg[idx];
		if(ref_depth == 0) return;
		float pt3[3] = {0.0f,0.0f,0.0f};
		calPointCloud(ref_depth,fx,fy,cx,cy,rot,t,pt3,w,h);
		
		if(pt3[2] == 0.0f) return;
		float u = (pt3[0]*fx / pt3[2])+cx + 0.5f;
		float v = (pt3[1]*fy / pt3[2])+cy + 0.5f;
		
		
		if(u >= 0.0f && u < (float)w && v < (float)h && v >=0.0f)
		{	
			float d_Jacobi[6];
			float dX = tex2D(texGradX,u,v) * fx; 
			float dY = tex2D(texGradY,u,v) * fy; 
			float residual = d_residual[idx];
			calJacobian(d_Jacobi,JtJ_final,Jtb_final,residual,dX,dY,pt3,w,h);
		
			//Update the n*6 matrix
			for(int j = 0; j<6; j++)
				d_jacobif[idx*6+j] = d_Jacobi[j];			
		
		}
	}	
}

__global__ void gradCompute(float *inImg, float *v1, float *v2, int w, int h)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if(x < w && y <h) 
	{
		int idx = x + y*w;
		if(x+1<w && x-1 >=0) v1[idx] = 0.5f * (inImg[idx+1] - inImg[idx-1]);
		else v1[idx] = 0;
		if(y+1<h && y-1 >=0) v2[idx] = 0.5f * (inImg[idx+w] - inImg[idx-w]);
		else v2[idx] = 0;
	}
}


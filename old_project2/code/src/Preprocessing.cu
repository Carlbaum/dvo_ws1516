#include "Preprocessing.h"

__global__ void downscaleIMap(float *iSrc, float *iDst, int n_w, int n_h, int w, int h) {
	size_t x = threadIdx.x + blockDim.x*blockIdx.x;
	size_t y = threadIdx.y + blockDim.y*blockIdx.y;
	
	if (x<n_w && y<n_h)
	{
		size_t idx = x + (size_t)n_w*y;
		float iCrr = iSrc[2*x+2*y*w];
		size_t fICrr = 4;
		bool lr = true;
		
		if (2*x+1<w) iCrr += iSrc[2*x+1+2*y*w]; else {fICrr-=2; lr=false;}
		if ((2*y+1)*w<h) iCrr += iSrc[2*x+(2*y+1)*w]; else { if (lr) {fICrr-=2; lr=false;} else fICrr--;}
		if (lr) iCrr += iSrc[2*x+1+(2*y+1)*w];
		iDst[idx] = iCrr / (float)fICrr;
	}
}


__global__ void downscaleDMap(float *dSrc, float *dDst, int n_w, int n_h, int w, int h) {
	size_t x = threadIdx.x + blockDim.x*blockIdx.x;
	size_t y = threadIdx.y + blockDim.y*blockIdx.y;
	
	if (x<n_w && y<n_h)
	{
		size_t idx = x + (size_t)n_w*y;
		float dCrr = dSrc[x*2+y*2*w];
		size_t fDCrr;
		if (dCrr == 0) fDCrr = 3; else fDCrr = 4;
		bool lri = true;
		float val; 
		
		if (2*x+1<w) {
			val = dSrc[2*x+1+2*y*w];
			if (val>0.0f) dCrr += 1.0f/val; else fDCrr--;
		} else { fDCrr-=2; lri=false; }
		if ((y+1)*w<h) {
			val = dSrc[2*x+(2*y+1)*w];
			if (val>0.0f) dCrr += 1.0f/val; else fDCrr--;
		} else { if (lri) {fDCrr-=2; lri=false;} else fDCrr--;}
		if (lri) {
			val = dSrc[2*x+1+(2*y+1)*w];
			if (val>0.0f) dCrr += 1.0f/val; else fDCrr--;
		}
		if (dCrr!=0.0f) dDst[idx] = (float)fDCrr / dCrr;
		// no else as dDst is initialised to 0
	}
}

__global__ void computeDerivatives (float *iCrr, float *dX, float *dY, int w, int h)
{
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t ind = x + y * w;
    
    if (x<w && y<h) {
        dX[ind] = (iCrr[min((int)(x+1), w-1) + w*(int)y]-iCrr[max((int)(x-1), 0) + w*(int)y])*0.5f;
        dY[ind] = (iCrr[(int)x + w*min((int)(y+1), h-1)]-iCrr[(int)x + w*max((int)(y-1), 0)])*0.5f;
    }
    
}


void  buildMapPyramids(float **img, float **depth, int lvl, int w, int h) {
	
	dim3 block = dim3(32, 8, 1);
	dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);
	int n_w, n_h;
	
	for (int i=0; i<lvl-1; i++) {
		n_w = (w+1)/2;
		n_h = (h+1)/2;
		grid = dim3((n_w+block.x-1)/block.x, (n_h+block.y-1)/block.y, 1);
		downscaleIMap <<<grid, block>>> (img[i], img[i+1], n_w, n_h, w, h);
		downscaleDMap <<<grid, block>>> (depth[i], depth[i+1], n_w, n_h, w, h);
		w = n_w;
		h = n_h;
	}
}

void  buildDrvPyramids(float **img, float **d_dXPy, float **d_dYPy, int lvl, int w, int h) {
	
	dim3 block = dim3(32, 8, 1);
	dim3 grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);
	
	for (int i=0; i<lvl; i++) {
		grid = dim3((w+block.x-1)/block.x, (h+block.y-1)/block.y, 1);
    	computeDerivatives <<<grid, block>>> (img[i], d_dXPy[i], d_dYPy[i], w, h);
		w = (w+1)/2;
		h = (h+1)/2;
	}
}

void buildKPyramid (Eigen::Matrix3f *KPy, int lvl) {
	for (int i=0; i<lvl-1; i++) {
		KPy[i+1] << KPy[i](0,0)/2.0f, 0.0, (KPy[i](0,2)+0.5f)/2.0f - 0.5f,
					0.0, KPy[i](1,1)/2.0f, (KPy[i](1,2)+0.5)/2.0f - 0.5f,
					0.0, 0.0, 1.0f;
	}
}

void buildIKPyramid (Eigen::Matrix3f *iKPy, Eigen::Matrix3f *KPy, int lvl) {
	for (int i=0; i<lvl; i++) {
		iKPy[i] << 1.0f/KPy[i](0,0), 0.0, -(KPy[i](0,2)/KPy[i](0,0)),
				   0.0, 1.0f/KPy[i](1,1), -(KPy[i](1,2)/KPy[i](1,1)),
				   0.0, 0.0, 1.0f;
	}
}
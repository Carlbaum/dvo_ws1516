#include "GPUMem.h"

void allocGPUMem(float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_kPy, float **d_ikPy, float **d_dXPy, float **d_dYPy, float **d_res, float **d_J, int lvlNum, size_t n, float **d_R, float **d_t, float **d_redArr, float **d_redArr2, float **d_A, float **d_b) {
	size_t pyLvlSize = (size_t)n;
	for (int i=0; i<lvlNum; i++) {  // TODO: allocate one big space per pyramid / for all pyramids and create a pointer Array for accessing the levels
		cudaMalloc(&d_iPyRef[i], pyLvlSize*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_dPyRef[i], pyLvlSize*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_iPyCrr[i], pyLvlSize*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_dPyCrr[i], pyLvlSize*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_kPy[i], 9*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_ikPy[i], 9*sizeof(float));  CUDA_CHECK;
		cudaMalloc(&d_dXPy[i], pyLvlSize*sizeof(float)); CUDA_CHECK;
		cudaMalloc(&d_dYPy[i], pyLvlSize*sizeof(float)); CUDA_CHECK;
		cudaMemset(d_iPyRef[i], 0, pyLvlSize*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_dPyRef[i], 0, pyLvlSize*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_iPyCrr[i], 0, pyLvlSize*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_dPyCrr[i], 0, pyLvlSize*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_kPy[i], 0, 9*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_ikPy[i], 0, 9*sizeof(float));  CUDA_CHECK;
		cudaMemset(d_dXPy[i], 0, pyLvlSize*sizeof(float)); CUDA_CHECK;
		cudaMemset(d_dYPy[i], 0, pyLvlSize*sizeof(float)); CUDA_CHECK;
		pyLvlSize = (pyLvlSize+1) / 2;
	}
	cudaMalloc(d_res, n*sizeof(float));  CUDA_CHECK;
	cudaMalloc(d_J, 6*n*sizeof(float));  CUDA_CHECK;
	cudaMalloc(d_R, 9*sizeof(float));  CUDA_CHECK;
	cudaMalloc(d_t, 3*sizeof(float));  CUDA_CHECK;
	cudaMalloc(d_redArr, 27*((n+255)/256)*sizeof(float));  CUDA_CHECK;
	cudaMalloc(d_redArr2, 27*((n+65535)/65536)*sizeof(float));  CUDA_CHECK;
	cudaMalloc(d_A, 36*sizeof(float));  CUDA_CHECK;
	cudaMalloc(d_b, 6*sizeof(float));  CUDA_CHECK;
	cudaMemset(*d_res, 0, n*sizeof(float));  CUDA_CHECK;
	cudaMemset(*d_J, 0, 6*n*sizeof(float));  CUDA_CHECK;
	cudaMemset(*d_R, 0, 9*sizeof(float));  CUDA_CHECK;
	cudaMemset(*d_t, 0, 3*sizeof(float));  CUDA_CHECK;
	cudaMemset(*d_redArr, 0, 27*((n+255)/256)*sizeof(float));  CUDA_CHECK;
	cudaMemset(*d_redArr2, 0, 27*((n+65535)/65536)*sizeof(float));  CUDA_CHECK;
	cudaMemset(*d_A, 0, 36*sizeof(float));  CUDA_CHECK;
	cudaMemset(*d_b, 0, 6*sizeof(float));  CUDA_CHECK;
}

void freeGPUMem(float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_kPy, float **d_ikPy, float **d_dXPy, float **d_dYPy, float *d_res, float *d_J, int lvlNum, int w, int h, float *d_R, float *d_t, float *d_redArr, float *d_redArr2, float *d_A, float *d_b) {
	size_t pyLvlSize = (size_t)w*h;
	for (int i=0; i<lvlNum; i++) {
		cudaFree(d_iPyRef[i]);  CUDA_CHECK;
		cudaFree(d_dPyRef[i]);  CUDA_CHECK;
		cudaFree(d_iPyCrr[i]);  CUDA_CHECK;
		cudaFree(d_dPyCrr[i]);  CUDA_CHECK;
		cudaFree(d_kPy[i]);  CUDA_CHECK;
		cudaFree(d_ikPy[i]);  CUDA_CHECK;
		cudaFree(d_dXPy[i]); CUDA_CHECK;
		cudaFree(d_dYPy[i]); CUDA_CHECK;
		pyLvlSize = (pyLvlSize+1) / 2;
	}
	cudaFree(d_res);  CUDA_CHECK;
	cudaFree(d_J);  CUDA_CHECK;
	cudaFree(d_R);  CUDA_CHECK;
	cudaFree(d_t);  CUDA_CHECK;
	cudaFree(d_redArr);  CUDA_CHECK;
	cudaFree(d_redArr2);  CUDA_CHECK;
	cudaFree(d_A);  CUDA_CHECK;
	cudaFree(d_b);  CUDA_CHECK;
}
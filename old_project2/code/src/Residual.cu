#include "Residual.h"

__constant__ float c_kPy[90];
__constant__ float c_ikPy[90];

// __device__ float3 mulKVec(float* k, float3 vec) {
__device__ float3 mulKVec(float k0, float k6, float k4, float k7, float3 vec) {
    // return make_float3(k[0]*vec.x+k[6]*vec.z, k[4]*vec.y+k[7]*vec.z, vec.z);
    return make_float3(k0*vec.x+k6*vec.z, k4*vec.y+k7*vec.z, vec.z);
}

__device__ float3 mulRVec(float* R, float3 vec) {
    return make_float3(R[0]*vec.x+R[3]*vec.y+R[6]*vec.z, R[1]*vec.x+R[4]*vec.y+R[7]*vec.z, R[2]*vec.x+R[5]*vec.y+R[8]*vec.z);
}

void setConstMemR(Eigen::Matrix3f kPy, Eigen::Matrix3f ikPy, int offset) {
    cudaMemcpyToSymbol (c_kPy, kPy.transpose().data(), 9*sizeof(float), offset);
    cudaMemcpyToSymbol (c_ikPy, ikPy.transpose().data(), 9*sizeof(float), offset);
}

texture<float, 2, cudaReadModeElementType> tex;
// __global__ void computeResidual(float *iRef, float *dRef, float *iCrr, float *dCrr, float *k, float *ik, float *res, int w, int h, float *d_R, float *d_t) {
__global__ void computeResidual(float *iRef, float *dRef, float *iCrr, float *dCrr, int lvl, float *res, int w, int h, float *d_R, float *d_t) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    // check if within bounds
    if (x < w && y < h)
    {
        size_t idx = x + (size_t)w*y;
        float d = dRef[idx];
        if (d>0.0f) {
            float3 pos = make_float3((float)x,(float)y,1.0f);
            // pos = mulKVec(ik,pos);
            pos = mulKVec(c_ikPy[lvl*9], c_ikPy[lvl*9+6], c_ikPy[lvl*9+4], c_ikPy[lvl*9+7], pos);
            pos.x *= d; pos.y *= d; pos.z = d;
            pos = mulRVec(d_R, pos);
            pos = make_float3(pos.x+d_t[0], pos.y+d_t[1], pos.z+d_t[2]);
            if (pos.z>0.0f) {
                // pos = mulKVec(k,pos);
                pos = mulKVec(c_kPy[lvl*9], c_kPy[lvl*9+6], c_kPy[lvl*9+4], c_kPy[lvl*9+7], pos);
                pos.x /= pos.z; pos.y /= pos.z;
                if ((int)(pos.x) < w && (int)(pos.y) < h && (int)(pos.x) >= 0 && (int)(pos.y) >= 0)
                    res[idx] = iRef[idx] - tex2D(tex, pos.x+0.5f, pos.y+0.5f);
            }
        }
    }
}

// void computeResiduals(float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_kPy, float **d_ikPy, float *d_res, int lvl, int w, int h, float *d_R, float *d_t) {
void computeResiduals(float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float *d_res, int lvl, int w, int h, float *d_R, float *d_t) {
    dim3 block = dim3(32, 8, 1);
    dim3 grid;
    
    // for (int i=0; i<lvl; i++) {
    //     w = (w+1)/2;
    //     h = (h+1)/2;
    // }
    tex.addressMode[0] = cudaAddressModeClamp;
    // clamp x to border
    tex.addressMode[1] = cudaAddressModeClamp;
    // clamp y to border
    tex.filterMode = cudaFilterModeLinear;
    // linear interpolation
    tex.normalized = false;
    // access as (x+0.5f,y+0.5f), not as ((x+0.5f)/w,(y+0.5f)/h)
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &tex, d_iPyCrr[lvl], &desc, w, h, w*sizeof(d_iPyCrr[lvl][0]));
    grid = dim3(((size_t)w+block.x-1)/block.x, ((size_t)h+block.y-1)/block.y, 1);
    // computeResidual <<<grid, block>>> (d_iPyRef[lvl], d_dPyRef[lvl], d_iPyCrr[lvl], d_dPyCrr[lvl], d_kPy[lvl], d_ikPy[lvl], d_res, w, h, d_R, d_t);
    computeResidual <<<grid, block>>> (d_iPyRef[lvl], d_dPyRef[lvl], d_iPyCrr[lvl], d_dPyCrr[lvl], lvl, d_res, w, h, d_R, d_t);
    cudaUnbindTexture(tex);
}
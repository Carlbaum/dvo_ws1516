#include "Jacobian.h"

__constant__ float c_kPy[90];
__constant__ float c_ikPy[90];

// __device__ float3 mulKVecJ(float* k, float3 vec) {
__device__ float3 mulKVecJ(float k0, float k6, float k4, float k7, float3 vec) {
    // return make_float3(k[0]*vec.x+k[6]*vec.z, k[4]*vec.y+k[7]*vec.z, vec.z);
    return make_float3(k0*vec.x+k6*vec.z, k4*vec.y+k7*vec.z, vec.z);
}

__device__ float3 mulRVecJ(float* R, float3 vec) {
    return make_float3(R[0]*vec.x+R[3]*vec.y+R[6]*vec.z, R[1]*vec.x+R[4]*vec.y+R[7]*vec.z, R[2]*vec.x+R[5]*vec.y+R[8]*vec.z);
}

void setConstMemJ(Eigen::Matrix3f kPy, Eigen::Matrix3f ikPy, int offset) {
    cudaMemcpyToSymbol (c_kPy, kPy.transpose().data(), 9*sizeof(float), offset);
    cudaMemcpyToSymbol (c_ikPy, ikPy.transpose().data(), 9*sizeof(float), offset);
}

texture<float, 2, cudaReadModeElementType> texX;
texture<float, 2, cudaReadModeElementType> texY;
// __global__ void computeJacobian(float *dRef, float *iCrr, float *k, float *ik, int w, int h, float *R, float *t, float *J) {
__global__ void computeJacobian(float *dRef, float *iCrr, int lvl, int w, int h, float *R, float *t, float *J) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y + blockDim.y*blockIdx.y;

    // check if within bounds
    if (x < w && y < h)
    {
        size_t idx = x + (size_t)w*y;
        float d = dRef[idx];
        if (d>0.0f) {
            float3 pos = make_float3(d*(float)x,d*(float)y,d);
            // pos = mulKVecJ(ik,pos);
            pos = mulKVecJ(c_ikPy[lvl*9], c_ikPy[lvl*9+6], c_ikPy[lvl*9+4], c_ikPy[lvl*9+7], pos);
            pos = mulRVecJ(R, pos);
            pos = make_float3(pos.x+t[0], pos.y+t[1], pos.z+t[2]);
            
            if (pos.z > 0.0f) {

                // float3 pos2 = mulKVecJ(k,pos);
                float3 pos2 = mulKVecJ(c_kPy[lvl*9], c_kPy[lvl*9+6], c_kPy[lvl*9+4], c_kPy[lvl*9+7], pos);
                pos2.x /= pos2.z; pos2.y /= pos2.z;
            
                if ((int)(pos2.x) < w && (int)(pos2.y) < h && (int)(pos2.x) >= 0 && (int)(pos2.y) >= 0) {
                    // float dX = tex2D(texX, pos2.x+0.5f, pos2.y+0.5f) * k[0];
                    // float dY = tex2D(texY, pos2.x+0.5f, pos2.y+0.5f) * k[4];
                    float dX = tex2D(texX, pos2.x+0.5f, pos2.y+0.5f) * c_kPy[lvl*9];
                    float dY = tex2D(texY, pos2.x+0.5f, pos2.y+0.5f) * c_kPy[lvl*9+4];
                    // float dX = (iCrr[min((int)(pos2.x+1), w-1) + w*(int)pos2.y]-iCrr[max((int)(pos2.x-1), 0) + w*(int)pos2.y])*0.5f * k[0];
                    // float dY = (iCrr[(int)pos2.x + w*min((int)(pos2.y+1), h-1)]-iCrr[(int)pos2.x + w*max((int)(pos2.y-1), 0)])*0.5f * k[4];
                    dX /= pos.z;
                    dY /= pos.z;
                    idx *= 6;
                    J[idx]   = -1.0f * (dX);
                    J[idx+1] = -1.0f * (dY);
                    J[idx+2] = -1.0f * (-dX*pos.x/pos.z -dY*pos.y/pos.z);
                    J[idx+3] = -1.0f * (-dX*pos.x*pos.y/pos.z -dY*(pos.z+pos.y*pos.y/pos.z));
                    J[idx+4] = -1.0f * (dX*(pos.z+pos.x*pos.x/pos.z) +dY*pos.x*pos.y/pos.z);
                    J[idx+5] = -1.0f * (-dX*pos.y +dY*pos.x);
                }
            }
        }
    }
}

// void computeJacobians(float **d_dPyRef, float **d_iPyCrr, float **d_kPy, float **d_ikPy, int lvl, int w, int h, float *d_R, float *d_t, float *d_J, float *d_dX, float *d_dY) {
void computeJacobians(float **d_dPyRef, float **d_iPyCrr, int lvl, int w, int h, float *d_R, float *d_t, float *d_J, float *d_dX, float *d_dY) {
    dim3 block = dim3(32, 8, 1);
    dim3 grid;
    
    // for (int i=0; i<lvl; i++) {
    //     w = (w+1)/2;
    //     h = (h+1)/2;
    // }

    grid = dim3(((size_t)w+block.x-1)/block.x, ((size_t)h+block.y-1)/block.y, 1);
    
    texX.addressMode[0] = cudaAddressModeClamp; texX.addressMode[1] = cudaAddressModeClamp;
    texX.filterMode = cudaFilterModeLinear; texX.normalized = false;
    cudaChannelFormatDesc descX = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &texX, d_dX, &descX, w, h, w*sizeof(d_dX[0]));
    
    texY.addressMode[0] = cudaAddressModeClamp; texY.addressMode[1] = cudaAddressModeClamp;
    texY.filterMode = cudaFilterModeLinear; texY.normalized = false;
    cudaChannelFormatDesc descY = cudaCreateChannelDesc<float>();
    cudaBindTexture2D(NULL, &texY, d_dY, &descY, w, h, w*sizeof(d_dY[0]));
    
    // computeJacobian <<<grid, block>>> (d_dPyRef[lvl], d_iPyCrr[lvl], d_kPy[lvl], d_ikPy[lvl], w, h, d_R, d_t, d_J);
    computeJacobian <<<grid, block>>> (d_dPyRef[lvl], d_iPyCrr[lvl], lvl, w, h, d_R, d_t, d_J);
    cudaUnbindTexture(texX); cudaUnbindTexture(texY);
}
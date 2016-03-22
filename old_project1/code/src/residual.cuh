#ifndef GRADIENT_H
#define GRADIENT_H

#include <iostream>
#ifndef WIN64
    #define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>


void computeGradientCPU(const cv::Mat &gray, cv::Mat &gradient, int direction)
{
    int dirX = 1;
    int dirY = 0;
    if (direction == 1)
    {
        dirX = 0;
        dirY = 1;
    }

    // compute gradient manually using finite differences
    int w = gray.cols;
    int h = gray.rows;
    const float* ptrIn = (const float*)gray.data;
    gradient = cv::Mat::zeros(h, w, CV_32FC1);
    float* ptrOut = (float*)gradient.data;

    int yStart = dirY;
    int yEnd = h - dirY;
    int xStart = dirX;
    int xEnd = w - dirX;
    for (size_t y = yStart; y < yEnd; ++y)
    {
        for (size_t x = xStart; x < xEnd; ++x)
        {
            float v0;
            float v1;
            if (direction == 1)
            {
                // y-direction
                v0 = ptrIn[(y-1)*w + x];
                v1 = ptrIn[(y+1)*w + x];
            }
            else
            {
                // x-direction
                v0 = ptrIn[y*w + (x-1)];
                v1 = ptrIn[y*w + (x+1)];
            }
            ptrOut[y*w + x] = 0.5f * (v1 - v0);
        }
    }
}


__global__ void computeGradient(float *gray, float *gradient, int w, int h, int direction)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int idx = x + y*w;
    if (x < w && y < h)
    {
        if (direction == 1)
        {
            // y-direction
            if (y+1 < h && y-1 >= 0)
                gradient[idx] = 0.5f * (gray[(y+1)*w + x] - gray[(y-1)*w + x]);
            else
                gradient[idx] = 0.0f;
        }
        else
        {
            // x-direction
            if (x+1 < w && x-1 >= 0)
                gradient[idx] = 0.5f * (gray[y*w + (x+1)] - gray[y*w + (x-1)]);
            else
                gradient[idx] = 0.0f;
        }
    }
}


__global__ void computeResidual(float *gray, float *gradient, int w, int h){
    
}
void computeResidual(const cv::Mat &gray, cv::Mat &gradX, cv::Mat &gradY, bool useGpu)
{
    
    computeGradientCPU(gray, gradX, 0);
    computeGradientCPU(gray, gradY, 1);

}





float interpolate(const float* ptrImgIntensity, float x, float y, int w, int h)
{
    float valCur = std::numeric_limits<float>::quiet_NaN();

#if 0
    // direct lookup, no interpolation
    int x0 = static_cast<int>(std::floor(x + 0.5));
    int y0 = static_cast<int>(std::floor(y + 0.5));
    if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
        valCur = ptrImgIntensity[y0*w + x0];
#else
    // bilinear interpolation
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float x1_weight = x - static_cast<float>(x0);
    float y1_weight = y - static_cast<float>(y0);
    float x0_weight = 1.0 - x1_weight;
    float y0_weight = 1.0 - y1_weight;

    if (x0 < 0 || x0 >= w)
        x0_weight = 0.0;
    if (x1 < 0 || x1 >= w)
        x1_weight = 0.0;
    if (y0 < 0 || y0 >= h)
        y0_weight = 0.0;
    if (y1 < 0 || y1 >= h)
        y1_weight = 0.0;
    float w00 = x0_weight * y0_weight;
    float w10 = x1_weight * y0_weight;
    float w01 = x0_weight * y1_weight;
    float w11 = x1_weight * y1_weight;

    float sumWeights = w00 + w10 + w01 + w11;
    float sum = 0.0;
    if (w00 > 0.0)
        sum += static_cast<float>(ptrImgIntensity[y0*w + x0]) * w00;
    if (w01 > 0.0)
        sum += static_cast<float>(ptrImgIntensity[y1*w + x0]) * w01;
    if (w10 > 0.0)
        sum += static_cast<float>(ptrImgIntensity[y0*w + x1]) * w10;
    if (w11 > 0.0)
        sum += static_cast<float>(ptrImgIntensity[y1*w + x1]) * w11;

    if (sumWeights > 0.0)
        valCur = sum / sumWeights;
#endif

    return valCur;
}

#endif

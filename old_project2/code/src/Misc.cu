#include "Misc.cuh"

void convertSE3ToTf(const Eigen::VectorXf &xi, Eigen::Matrix3f &rot, Eigen::Vector3f &t)
{
	// rotation
	Sophus::SE3f se3 = Sophus::SE3f::exp(xi);
	Eigen::Matrix4f mat = se3.matrix();
	rot = mat.topLeftCorner(3, 3);
	t = mat.topRightCorner(3, 1);
}


void convertTfToSE3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Eigen::VectorXf &xi)
{
	Sophus::SE3f se3(rot, t);
	xi = Sophus::SE3f::log(se3);
}


cv::Mat loadIntensity(const std::string &filename)
{
	cv::Mat imgGray = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	// convert gray to float
	cv::Mat gray;
	imgGray.convertTo(gray, CV_32FC1, 1.0f / 255.0f);
	return gray;
}


// cv::Mat loadDepth(const std::string &filename)
// {
// 	//fill/read 16 bit depth image
// 	cv::Mat imgDepthIn = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
// 	cv::Mat imgDepth;
// 	imgDepthIn.convertTo(imgDepth, CV_32FC1, (1.0 / 5000.0));
// 	return imgDepth;
// }

/*
__device__ float3 mulKVec(float* k, float3 vec) {
    return make_float3(k[0]*vec.x+k[6]*vec.z, k[4]*vec.y+k[7]*vec.z, vec.z);
}

__device__ float3 mulRVec(float* R, float3 vec) {
    return make_float3(R[0]*vec.x+R[3]*vec.y+R[6]*vec.z, R[1]*vec.x+R[4]*vec.y+R[7]*vec.z, R[2]*vec.x+R[5]*vec.y+R[8]*vec.z);
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
*/
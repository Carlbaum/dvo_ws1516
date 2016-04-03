#include <Eigen/Dense>
#include "preprocessing.cuh"
#include "lieAlgebra.hpp"
// #include <string> //only needed for our 'debugging'

enum SolvingMethod { GAUSS_NEWTON, LEVENBERG_MARQUARDT, GRADIENT_DESCENT };
// enum DerivativeMethod { ANALYTIC, NUMERIC };
enum ResidualWeight { NONE, HUBER, TDIST };

class Tracker {

public:

/**
 * Tracker constructor
 * @param grayFirstFrame        gray image array of floats (not uchar)
 * @param depthFirstFrame       depth image array of floats
 * @param K                     Eigen 3x3 matrix with camera projection parameters
 * @param solvingMethod         Method for solving the linear system for delta ksi
 * @param weightType            Weight shape for the residuals // TODO? guillermo: not sure how far this is implemented
 * @param minLevel              Lowest level index in the pyramid to be used for alignment. Default is 0 (full resolution).
 * @param maxLevel              Highest level index of the pyramid above the full resolution image
 * @param maxIterationsPerLevel Maximum number of iterations per pyramid level
 */
Tracker(
        float* grayFirstFrame,
        float* depthFirstFrame,
        int w,
        int h,
        Eigen::Matrix3f K,
        int minLevel = 0,
        int maxLevel = 4,
        int maxIterationsPerLevel = 20,
        SolvingMethod solvingMethod = GAUSS_NEWTON,
        ResidualWeight weightType = NONE
                                    // bool useCUBLAS = true,
        ) :
        w(w),
        h(h),
        solvingMethod(solvingMethod),
        minLevel(minLevel),
        maxLevel(maxLevel),
        maxIterationsPerLevel(maxIterationsPerLevel),
        weightType(weightType),
        totalComputationTime(0),
        frameComputationTime(0),
        stepCount(0),
        xi_prev(Vector6f::Zero()),
        xi(Vector6f::Zero())
        // useCUBLAS(useCUBLAS)
{
        // if (useCUBLAS) cublasCreate(&handle);

        // make pyramid vector large enough to hold all levels
        d_cur.resize(maxLevel+1);
        d_prev.resize(maxLevel+1);

        // Create Buffers
        allocateGPUMemory();

        // allocate intrinisc matrix (camera projection) for all pyramid levels
        K_pyr.resize(maxLevel+1);
        K_pyr_inv.resize(maxLevel+1);

        fill_K_levels(K);

        // Fill image pyramid of the first frame. Already as previous frame since align fills the current frame d_cur and only swaps at the end.
        fill_pyramid(d_prev, grayFirstFrame, depthFirstFrame);
}

/**
 * destructor
 */
~Tracker() {
        // if (useCUBLAS) cublasDestroy(handle);
        deallocateGPUMemory();
}

/**
 * Provide a new frame to the tracker and calculate the transition from the previous
 * @param  grayCur  New (current) gray image of floats. Used as second frame.
 * @param  depthCur New (current) depth image of floats. It gets processed into a pyramid but its values are not actually used until the next call to align.
 * @return          Minimal transformation representation in twist coordinates. Optimal warp of the previous gray and depth onto the new (current) image.
 */
Vector6f align(float *grayCur, float *depthCur) {
        fill_pyramid(d_cur, grayCur, depthCur);

        // Use the previous xi as initial guess. It is stored in a private variable
        // other option:
        // xi = Vector6f::Zero();

        // from the highest level to the minimum level set
        for (int level = maxLevel; level >= minLevel; level--) {
                // set d_prev_err to big number      TODO: directly in GPU?

                // for a maximum number of iterations per level
                for (int i = 0; i < maxIterationsPerLevel; i++) {
                        // std::cout << level << ", " << i << std::endl;

                        // Calculate Rotation matrix and translation vector: CPU operation
                        convertSE3ToT(xi, R, t);

                        // transform_point: CUDA operation
                                // get 3D point: p=(u*d, v*d, d) // u, v are the camera coordinates
                                // unproyect from camera: p = K_inv * p     // TODO: actually precalculate R*K_inv, then copy to GPU
                                // transform: p = R * p + t
                                // proyect to camera: p = K * p
                                // get 2D camera coordinates: u = x/z; v = y/z
                                // store (u,v)_curr position for each (u,v)_prev

                        // parallel CUDA kernels:
                                // calculate_jacobian J(n,6)  // calculate_residuals r_xi(n,1) and error (mean squares of r_xi)
                                                              // calculate_weights W(n,1)

                        // parallel CUDA kernels:
                                // calculate A(6,6) = J.T * W * J   // calculate B(6,1) = -J.T * W * r
                                // TODO: best order to calculate the previous multiplications. J.T * W first? (used by both) or all together avoiding reads?

                        // solve linear system: A * delta_xi = b; with solver of Eigen library: CPU operation.      TODO: Faster to solve directly in GPU?

                        // if change in error is small      TODO: Compare directly in GPU?
                                // stop iterating in this level

                        // save error value for next iteration: d_prev_err = d_err:     TODO: use cudaMemcpy?
                }
        }


        // convertTToSE3(xi, R, t);
        // std::cout << xi << std::endl;

        // swap the pointers so we place image in the correct buffer next time this function is called
        temp_swap = d_cur; d_cur = d_prev; d_prev = temp_swap;
        // accumulate total_xi: total_xi = log(exp(xi)*exp(total_xi))
        return xi;
}

// double averageTime() {
//         return totalComputationTime / stepCount;
// }
//
// double totalComputationTime;
// double frameComputationTime;
// int stepCount;
Vector6f xi_total;



private:
struct PyramidLevel { float *gray, *depth, *gray_dx, *gray_dy; };
// host parameters
SolvingMethod solvingMethod;   // enum type of possible solving methods
int maxIterationsPerLevel;
int maxLevel;
int minLevel;   // For speed. Used if the highest precision is not required
int w;   // width of the first frame (and all frames)
int h;   // height of the first frame (and all frames)
ResidualWeight weightType;   // enum type of possible residual weighting. Defined in *_jacobian.cuh
bool useCUBLAS;   // TODO: option to NOT use cuBLAS

// cublasHandle_t handle; // not used if useCUBLAS = false

// device variables
float *d_J;   // device Jacobian array for ALL residuals
float *d_r;   // device residuals array
float *d_b;   // device linear system inhomogeneous term array
float *d_A;   // device linear system matrix array
// float *d_visualResidual;   // device image residuals array      TODO: needed? just for debugging?
float *d_error;   // mean squares error of residual
float *d_error_prev;   // mean squares error of residual in previous iteration
std::vector<PyramidLevel> d_cur;   // current vector of pointers to device pyramid level structures
std::vector<PyramidLevel> d_prev;   // previous vector of pointers to device pyramid level structures
std::vector<PyramidLevel> temp_swap;
// float *d_tdistWeightedSqSum;   // Studendt-T weights for each residual. Has the size of the image. // TODO: shouldn't this be a pyramid? Or is it just allocated in excess for higher levels?
Matrix3f R;
Vector3f t;

// host variables
std::vector<Matrix3f> K_pyr;   // stores projection matrix and downsampled version (intrinsic camera properties)
std::vector<Matrix3f> K_pyr_inv;   // inverse K_pyr
//std::vector<Matrix3f> d_K_pyr;
//std::vector<Matrix3f> d_K_pyr_inv;
Vector6f xi_prev;   // TODO: keeps last(?) frame for some reason
Vector6f xi;

//_________________PRIVATE FUNCTIONS____________________________________________
void fill_K_levels(Eigen::Matrix3f K) {
        K_pyr[0] = K;
        K_pyr_inv[0] = invertKMat(K_pyr[0]);
        for (int l = 1; l <= maxLevel; l++) {
                K_pyr[l] = downsampleK(K_pyr[l-1]);
                K_pyr_inv[l] = invertKMat(K_pyr[l]);
        }
}
/**
 * Copies the input image as the first pyramid level into the device and calculates the remaining levels upwards. Not to be used without access to private variables.
 * @param d_img    Vector of pointers to PyramidLevel structures allocated in device
 * @param grayImg  Gray image of full resolution (first level size) as an array of floats
 * @param depthImg Depth image of full resolution (first level size) as an array of floats
 */
void fill_pyramid(std::vector<PyramidLevel>& d_img, float *grayImg, float *depthImg) {
        // copy image into the basis of the pyramid
        cudaMemcpy(d_img[0].gray, grayImg, w*h*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMemcpy(d_img[0].depth, depthImg, w*h*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        int lw, lh; // width and height of downsampled images
        for (int l = 1; l <= maxLevel; l++) {
                lw = w / (1 << l); // bitwise operator to divide by 2**l
                lh = h / (1 << l);
                imresize_CUDA(d_img[l-1].gray, d_img[l].gray, 2*lw, 2*lh, lw, lh, 1, false);
                imresize_CUDA(d_img[l-1].depth, d_img[l].depth, 2*lw, 2*lh, lw, lh, 1, false);
        }

        for (int l = 0; l <= maxLevel; l++) {
                lw = w / (1 << l); // bitwise operator to divide by 2**l
                lh = h / (1 << l);
                // compute derivatives!!
                image_derivatives_CUDA(d_img[l].gray,d_img[l].gray_dx,d_img[l].gray_dy,lw,lh);
        }
        // //Debug
        // for (int l = 0; l < maxLevel; l++) {
        //         int lw = w / (1 << l); // bitwise operator to divide by 2**l
        //         int lh = h / (1 << l);
        //         cv::Mat mTest(lh, lw, CV_32FC1);
        //         float *prev = new float[lw*lh];
        //         //DEPTH
        //         cudaMemcpy(prev, d_img[l].depth, lw*lh*sizeof(float), cudaMemcpyDeviceToHost);
        //         convert_layered_to_mat(mTest, prev);
        //         double min, max;
        //         cv::minMaxLoc(mTest, &min, &max);
        //         showImage( "Depth: " + std::to_string(l), mTest/max, 100, 100); //cv::waitKey(0); // TODO oskar: .. values are probably not in the correct range.. neither [0,1] nor [0,255].. result is just black n white.. should be gray scale
        //         //GRAY
        //         cudaMemcpy(prev, d_img[l].gray, lw*lh*sizeof(float), cudaMemcpyDeviceToHost);
        //         convert_layered_to_mat(mTest, prev);
        //         showImage( "Gray: " + std::to_string(l), mTest, 300, 100); //cv::waitKey(0);
        //         //DERIVATIVES dx
        //         cudaMemcpy(prev, d_img[l].gray_dx, lw*lh*sizeof(float), cudaMemcpyDeviceToHost);
        //         convert_layered_to_mat(mTest, prev);
        //         showImage( "DX: " + std::to_string(l), mTest, 300, 100); //cv::waitKey(0);
        //         //DERIVATIVES dx
        //         cudaMemcpy(prev, d_img[l].gray_dy, lw*lh*sizeof(float), cudaMemcpyDeviceToHost);
        //         convert_layered_to_mat(mTest, prev);
        //         showImage( "DY: " + std::to_string(l), mTest, 300, 100); cv::waitKey(0);
        //         //cvDestroyAllWindows();
        // }

}

void allocateGPUMemory() {
        cudaMalloc(&d_J, w*h*6*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_r, w*h*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_b, 6*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_A, 6*6*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_error, sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_visualResidual, w*h*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_n, sizeof(int)); CUDA_CHECK;

        // allocate pyramid vector levels in device memory
        for (int l = 0; l <= maxLevel; l++) {
                int lw = w / (1 << l); // calculating bitwise the succesive powers of 2
                int lh = h / (1 << l);
                cudaMalloc(&d_cur [l].gray, lw*lh*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[l].gray, lw*lh*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_cur [l].depth, lw*lh*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[l].depth, lw*lh*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_cur [l].gray_dx, lw*lh*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[l].gray_dx, lw*lh*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_cur [l].gray_dy, lw*lh*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[l].gray_dy, lw*lh*sizeof(float)); CUDA_CHECK;
        }

        // Student-T weights allocation // TODO: shouldn't this be a pyramid? Or is it just allocated in excess for higher levels?
        if (weightType == TDIST) {
                cudaMalloc(&d_tdistWeightedSqSum, w*h*sizeof(float)); CUDA_CHECK;
        }
}

void deallocateGPUMemory() {
        cudaFree(d_J); CUDA_CHECK;
        cudaFree(d_r); CUDA_CHECK;
        cudaFree(d_b); CUDA_CHECK;
        cudaFree(d_A); CUDA_CHECK;
        cudaFree(d_visualResidual); CUDA_CHECK;
        cudaFree(d_error); CUDA_CHECK;
        cudaFree(d_n); CUDA_CHECK;

        for (int l = 0; l <= maxLevel; l++) {
                cudaFree(d_cur [l].gray); CUDA_CHECK;
                cudaFree(d_prev[l].gray); CUDA_CHECK;
                cudaFree(d_cur [l].depth); CUDA_CHECK;
                cudaFree(d_prev[l].depth); CUDA_CHECK;
                cudaFree(d_cur [l].gray_dx); CUDA_CHECK;
                cudaFree(d_prev[l].gray_dx); CUDA_CHECK;
                cudaFree(d_cur [l].gray_dy); CUDA_CHECK;
                cudaFree(d_prev[l].gray_dy); CUDA_CHECK;
        }

        if (weightType == TDIST) {
                cudaFree(d_tdistWeightedSqSum); CUDA_CHECK;
        }

}

};

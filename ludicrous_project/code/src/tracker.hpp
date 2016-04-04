#include <Eigen/Dense>
#include "preprocessing.cuh"
#include "lieAlgebra.hpp"
#include "alignment.cuh"
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
        int width,
        int height,
        Eigen::Matrix3f K,
        int minLevel = 0,
        int maxLevel = 4,
        int maxIterationsPerLevel = 20,
        SolvingMethod solvingMethod = GAUSS_NEWTON,
        ResidualWeight weightType = NONE
                                    // bool useCUBLAS = true,
        ) :
        width(width),
        height(height),
        solvingMethod(solvingMethod),
        minLevel(minLevel),
        maxLevel(maxLevel),
        maxIterationsPerLevel(maxIterationsPerLevel),
        weightType(weightType),
        // totalComputationTime(0),
        // frameComputationTime(0),
        // stepCount(0),
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
        K_inv_pyr.resize(maxLevel+1);

        fill_K_levels(K);

        load_K_to_device();

        // Fill image pyramid of the first frame. Already as previous frame since align fills the current frame d_cur and only swaps at the end.
        fill_pyramid(d_prev, grayFirstFrame, depthFirstFrame);

        define_texture_parameters();
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
        // other option, initialize as 0:
        // xi = Vector6f::Zero();

        // from the highest level to the minimum level set
        for (int level = maxLevel; level >= minLevel; level--) {
                int level_width = width / (1 << level); // calculating bitwise the succesive powers of 2
                int level_height = height / (1 << level);
                // set d_prev_err to big float number      TODO: directly in GPU?

                bind_textures(level, level_width, level_height); // used for interpolation in the current image

                // for a maximum number of iterations per level
                for (int i = 0; i < maxIterationsPerLevel; i++) {
                        // std::cout << level << ", " << i << std::endl;

                        // Calculate Rotation matrix and translation vector: CPU operation
                        convertSE3ToT(xi, R, t);

                        // calculate RK_inv = R*K_inv in CPU and copy to constant memory (both rotation and translation)
                        RK_inv = R * K_inv_pyr[level];
                        cudaMemcpyToSymbol (const_RK_inv, RK_inv.data(), 9*sizeof(float));
                        cudaMemcpyToSymbol (const_translation, t.data(), 3*sizeof(float));

                        // transform_points: CUDA operation
                        transform_points(level, level_width, level_height);

                        // parallel CUDA kernels: two streams
                                // calculate_jacobian J(n,6)  // calculate_residuals r_xi(n,1) and error (mean squares of r_xi)
                                                              // calculate_weights W(n,1)
                        calculate_jacobian(level, level_width, level_height, stream1);
                        calculate_residuals(level, level_width, level_height, stream2);
                        cudaDeviceSynchronize();

                        // parallel CUDA kernels: two streams
                                // calculate A(6,6) = J.T * W * J   // calculate B(6,1) = -J.T * W * r
                                // TODO: best order to calculate the previous multiplications. J.T * width first? (used by both) or all together avoiding reads?

                        // solve linear system: A * delta_xi = b; with solver of Eigen library: CPU operation.      TODO: Faster to solve directly in GPU?

                        // if change in error is small      TODO: Compare directly in GPU?
                                // stop iterating in this level

                        // save error value for next iteration: d_prev_err = d_err:     TODO: use cudaMemcpy?
                }

                unbind_textures();  // leave texture references free for binding at level below
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
int width;   // width of the first frame (and all frames)
int height;   // height of the first frame (and all frames)
ResidualWeight weightType;   // enum type of possible residual weighting. Defined in *_jacobian.cuh
// bool useCUBLAS;   // TODO: option to NOT use cuBLAS

// cublasHandle_t handle; // not used if useCUBLAS = false

// device variables
float *d_x_prime; // 3D x position in the second frame
float *d_y_prime; // 3D y position in the second frame
float *d_z_prime; // 3D z position in the second frame
float *d_u_warped;  // warped x position of every pixel in the first image onto the second image
float *d_v_warped;  // warped y position of every pixel in the first image onto the second image
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
Matrix3f RK_inv;
Vector3f t;

// host variables
// watch out: Eigen::Matrix are stored column wise
std::vector<Matrix3f> K_pyr;   // stores projection matrix and downsampled version (intrinsic camera properties)
std::vector<Matrix3f> K_inv_pyr;   // inverse K_pyr
//std::vector<Matrix3f> d_K_pyr;
//std::vector<Matrix3f> d_K_pyr_inv;
Vector6f xi_prev;   // TODO: keeps last(?) frame for some reason
Vector6f xi;

//_________________PRIVATE FUNCTIONS____________________________________________
void fill_K_levels(Eigen::Matrix3f K) {
        K_pyr[0] = K;
        K_inv_pyr[0] = invertKMat(K_pyr[0]);
        for (int level = 1; level <= maxLevel; level++) {
                K_pyr[level] = downsampleK(K_pyr[level-1]);
                K_inv_pyr[level] = invertKMat(K_pyr[level]);
        }
}

// watch out: K is stored column-wise !!
void load_K_to_device() {
        float *p_const;
        for (int level = 0; level <= maxLevel; level++) {
                // get the pointer to constant memory
                cudaGetSymbolAddress((void **)&p_const, const_K_pyr);
                // write each level at an incremented device pointer position
                cudaMemcpy(p_const + 9*level, K_pyr[level].data(), 9*sizeof(float), cudaMemcpyHostToDevice);
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
        cudaMemcpy(d_img[0].gray, grayImg, width*height*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMemcpy(d_img[0].depth, depthImg, width*height*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

        int level_width, level_height; // width and height of downsampled images
        for (int level = 1; level <= maxLevel; level++) {
                level_width = width / (1 << level); // bitwise operator to divide by 2**level
                level_height = height / (1 << level);
                imresize_CUDA(d_img[level-1].gray, d_img[level].gray, 2*level_width, 2*level_height, level_width, level_height, 1, false);
                imresize_CUDA(d_img[level-1].depth, d_img[level].depth, 2*level_width, 2*level_height, level_width, level_height, 1, true); // TODO: Check properly if isDepthImage is working. Looks like it does
        }

        for (int level = 0; level <= maxLevel; level++) {
                level_width = width / (1 << level); // bitwise operator to divide by 2**level
                level_height = height / (1 << level);
                // compute derivatives!!
                image_derivatives_CUDA(d_img[level].gray,d_img[level].gray_dx,d_img[level].gray_dy,level_width,level_height);
        }
        // //Debug
        // for (int level = 0; level < maxLevel; level++) {
        //         int level_width = width / (1 << level); // bitwise operator to divide by 2**level
        //         int level_height = height / (1 << level);
        //         cv::Mat mTest(level_height, level_width, CV_32FC1);
        //         float *prev = new float[level_width*level_height];
        //         //DEPTH
        //         cudaMemcpy(prev, d_img[level].depth, level_width*level_height*sizeof(float), cudaMemcpyDeviceToHost);
        //         convert_layered_to_mat(mTest, prev);
        //         double min, max;
        //         cv::minMaxLoc(mTest, &min, &max);
        //         showImage( "Depth: " + std::to_string(level), mTest/max, 100, 100); //cv::waitKey(0); // TODO oskar: .. values are probably not in the correct range.. neither [0,1] nor [0,255].. result is just black n white.. should be gray scale
        //         //GRAY
        //         cudaMemcpy(prev, d_img[level].gray, level_width*level_height*sizeof(float), cudaMemcpyDeviceToHost);
        //         convert_layered_to_mat(mTest, prev);
        //         showImage( "Gray: " + std::to_string(level), mTest, 300, 100); //cv::waitKey(0);
        //         //DERIVATIVES dx
        //         cudaMemcpy(prev, d_img[level].gray_dx, level_width*level_height*sizeof(float), cudaMemcpyDeviceToHost);
        //         convert_layered_to_mat(mTest, prev);
        //         showImage( "DX: " + std::to_string(level), mTest, 300, 100); //cv::waitKey(0);
        //         //DERIVATIVES dx
        //         cudaMemcpy(prev, d_img[level].gray_dy, level_width*level_height*sizeof(float), cudaMemcpyDeviceToHost);
        //         convert_layered_to_mat(mTest, prev);
        //         showImage( "DY: " + std::to_string(level), mTest, 300, 100); cv::waitKey(0);
        //         //cvDestroyAllWindows();
        // }

}

void define_texture_parameters() {
        texRef_grayImg.normalized = false;
        texRef_grayImg.filterMode = cudaFilterModeLinear;

        texRef_gray_dx.normalized = false;
        texRef_gray_dx.filterMode = cudaFilterModeLinear;

        texRef_gray_dy.normalized = false;
        texRef_gray_dy.filterMode = cudaFilterModeLinear;
}

void bind_textures(int level, int level_width, int level_height) {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); // number of bits for each texture
        int pitch = width*sizeof(float);
        cudaBindTexture2D(NULL, &texRef_grayImg, d_cur[level].gray, &desc, level_width, level_height, pitch);
        cudaBindTexture2D(NULL, &texRef_gray_dx, d_cur[level].gray_dx, &desc, level_width, level_height, pitch);
        cudaBindTexture2D(NULL, &texRef_gray_dy, d_cur[level].gray_dy, &desc, level_width, level_height, pitch);
}

void unbind_textures() {
        cudaUnbindTexture(texRef_grayImg);
        cudaUnbindTexture(texRef_gray_dx);
        cudaUnbindTexture(texRef_gray_dy);
}

/**
 * Calculates the transformed positions on the second frame for every pixel in the first
 */
void transform_points(int level, int level_width, int level_height) {
          // Block = 2D array of threads
          dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

          // Grid = 2D array of blocks
          // gridSizeX = ceil( width / nBlocksX )
          // gridSizeY = ceil( height / nBlocksX )
          int   gridSizeX = (width  + dimBlock.x-1) / dimBlock.x;
          int   gridSizeY = (height + dimBlock.y-1) / dimBlock.y;
          dim3  dimGrid( gridSizeX, gridSizeY, 1 );

          d_transform_points <<< dimGrid, dimBlock >>> (d_x_prime, d_y_prime, d_z_prime, d_u_warped, d_v_warped, d_prev[level].depth, level_width, level_height, level);
}

/**
 * Calculates the jacobian at each pixel
 */
void calculate_jacobian(int level, int level_width, int level_height, cudaStream_t stream) {
          // Block = 2D array of threads
          dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

          // Grid = 2D array of blocks
          // gridSizeX = ceil( width / nBlocksX )
          // gridSizeY = ceil( height / nBlocksX )
          int   gridSizeX = (width  + dimBlock.x-1) / dimBlock.x;
          int   gridSizeY = (height + dimBlock.y-1) / dimBlock.y;
          dim3  dimGrid( gridSizeX, gridSizeY, 1 );

          d_calculate_jacobian <<< dimGrid, dimBlock, 0, stream >>> (d_J, d_x_prime, d_y_prime, d_z_prime, d_u_warped, d_v_warped, level_width, level_height, level); // texture is accessed directly. No argument needed
}

/**
 * Calculates the residual at each pixel
 */
void calculate_residuals(int level, int level_width, int level_height, cudaStream_t stream) {
          // Block = 2D array of threads
          dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

          // Grid = 2D array of blocks
          // gridSizeX = ceil( width / nBlocksX )
          // gridSizeY = ceil( height / nBlocksX )
          int   gridSizeX = (width  + dimBlock.x-1) / dimBlock.x;
          int   gridSizeY = (height + dimBlock.y-1) / dimBlock.y;
          dim3  dimGrid( gridSizeX, gridSizeY, 1 );

          d_calculate_residuals <<< dimGrid, dimBlock, 0, stream >>> (d_r, d_prev[level].gray, d_u_warped, d_v_warped, level_width, level_height, level); // texture is accessed directly. No argument needed
}

void allocateGPUMemory() {
        cudaMalloc(&d_J, width*height*6*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_r, width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_x_prime, width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_y_prime, width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_z_prime, width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_u_warped, width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_v_warped, width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_b, 6*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_A, 6*6*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_error, sizeof(float)); CUDA_CHECK;
        // cudaMalloc(&d_visualResidual, width*height*sizeof(float)); CUDA_CHECK;
        // cudaMalloc(&d_n, sizeof(int)); CUDA_CHECK;

        // allocate pyramid vector levels in device memory
        for (int level = 0; level <= maxLevel; level++) {
                int level_width = width / (1 << level); // calculating bitwise the succesive powers of 2
                int level_height = height / (1 << level);
                cudaMalloc(&d_cur [level].gray, level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[level].gray, level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_cur [level].depth, level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[level].depth, level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_cur [level].gray_dx, level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[level].gray_dx, level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_cur [level].gray_dy, level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[level].gray_dy, level_width*level_height*sizeof(float)); CUDA_CHECK;
        }

        // Student-T weights allocation // TODO: shouldn't this be a pyramid? Or is it just allocated in excess for higher levels?
        if (weightType == TDIST) {
                // cudaMalloc(&d_tdistWeightedSqSum, width*height*sizeof(float)); CUDA_CHECK;
        }
}

void deallocateGPUMemory() {
        cudaFree(d_J); CUDA_CHECK;
        cudaFree(d_r); CUDA_CHECK;
        cudaFree(d_x_prime); CUDA_CHECK;
        cudaFree(d_y_prime); CUDA_CHECK;
        cudaFree(d_z_prime); CUDA_CHECK;
        cudaFree(d_u_warped); CUDA_CHECK;
        cudaFree(d_v_warped); CUDA_CHECK;
        cudaFree(d_b); CUDA_CHECK;
        cudaFree(d_A); CUDA_CHECK;
        cudaFree(d_error); CUDA_CHECK;
        // cudaFree(d_visualResidual); CUDA_CHECK;
        // cudaFree(d_n); CUDA_CHECK;

        for (int level = 0; level <= maxLevel; level++) {
                cudaFree(d_cur [level].gray); CUDA_CHECK;
                cudaFree(d_prev[level].gray); CUDA_CHECK;
                cudaFree(d_cur [level].depth); CUDA_CHECK;
                cudaFree(d_prev[level].depth); CUDA_CHECK;
                cudaFree(d_cur [level].gray_dx); CUDA_CHECK;
                cudaFree(d_prev[level].gray_dx); CUDA_CHECK;
                cudaFree(d_cur [level].gray_dy); CUDA_CHECK;
                cudaFree(d_prev[level].gray_dy); CUDA_CHECK;
        }

        if (weightType == TDIST) {
                // cudaFree(d_tdistWeightedSqSum); CUDA_CHECK;
        }

}

};

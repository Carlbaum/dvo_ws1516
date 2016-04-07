#include <Eigen/Dense>
#include "preprocessing.cuh"
#include "lieAlgebra.hpp"
#include "alignment.cuh"
// cuBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
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
        bool useWeights = true,
        int maxIterationsPerLevel = 20,
        SolvingMethod solvingMethod = GAUSS_NEWTON,
        ResidualWeight weightType = NONE,
        bool useCUBLAS = true
        ) :
        width(width),
        height(height),
        solvingMethod(solvingMethod),
        minLevel(minLevel),
        maxLevel(maxLevel),
        maxIterationsPerLevel(maxIterationsPerLevel),
        //weightType(weightType),
        // totalComputationTime(0),
        // frameComputationTime(0),
        // stepCount(0),
        xi_prev(Vector6f::Zero()),
        xi(Vector6f::Zero()),
        xi_total(Vector6f::Zero()),
        useCUBLAS(useCUBLAS),
        useWeights(useWeights)
{
        if (useCUBLAS) {
                stat = cublasCreate(&handle);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                        printf ("\n\n!---------------cuBLAS initialization failed---------------!\n\n");
                        //return EXIT_FAILURE;
                }
                else printf ("\n\n!--------------cuBLAS initialization succesful--------------!\n\n");
        }

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
        if (useCUBLAS) cublasDestroy(handle);
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
                // cvDestroyAllWindows(); // TODO: Remove this.. it's only for debugging
                int level_width = width / (1 << level); // calculating bitwise the succesive powers of 2
                int level_height = height / (1 << level);
                int n = level_width*level_height;
                // set d_prev_err to big float number      TODO: directly in GPU?
                float error_prev = std::numeric_limits<float>::max(); // initialize as a great value to make sure loop continues for at least one iteration below
                // std::cout << "    At pyramid level: " << level << std::endl;
                bind_textures(level, level_width, level_height); // used for interpolation in the current image

                // T distribution variance, (initial)
                float variance = VARIANCE_INITIAL;

                //cudaMemcpy(d_sigma, &SIGMA_INITIAL, sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK;

                // for a maximum number of iterations per level
                for (int i = 0; i < maxIterationsPerLevel; i++) {
                        // std::cout << "          At iteration: " << i << std::endl;
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

                        // JTW = J' * W, that is the transpose of the jacobian multiplied by the diagonal matrix W, storing weights calculated using the residuals
                        if(useWeights) {
                            calculate_jtw(level, level_width, level_height, variance);
                            std::cout << "We are using weights!!" << std::endl;
                        }

                        // DEBUG --- PLOT RESIDUAL IMAGES
                    //     if(level < 2){
                    //         cv::Mat mTest(level_height, level_width, CV_32FC1);
                    //         cv::Mat mTest2(level_height, level_width, CV_32FC1);
                    //         float *prev = new float[level_width*level_height];
                    //         cudaMemcpy(prev, d_r, level_width*level_height*sizeof(float), cudaMemcpyDeviceToHost);
                    //         convert_layered_to_mat(mTest, prev);
                    //         mTest2 = cv::abs(mTest);
                    //         //cv::convertScaleAbs(mTest, mTest);
                    //         showImage( "Residual at level: " + std::to_string(level) + ", iteration: " + std::to_string(i) , mTest2, 300, 100); cv::waitKey(0);
                    //     /*    cudaMemcpy(prev, d_W, level_width*level_height*sizeof(float), cudaMemcpyDeviceToHost);
                    //         convert_layered_to_mat(mTest, prev);
                    //         double min, max;
                    //         cv::minMaxLoc(mTest, &min, &max);
                    //         showImage( "Weights at level: " + std::to_string(level) + ", iteration: " + std::to_string(i) , mTest/max, 300, 100); cv::waitKey(0);
                    //     */
                    //    }

                        // Use nVidia's cuBLAS library for linear algebra calculations
                        if(useCUBLAS) {

                                // Matrix multiplication using cuBLAS looks like this:
                                //      C = alpha * D * F + beta * C,    where C,D,F are matrices and alpha & beta are scalars
                                //
                                // For specifying whether or not the matrices D & F are transposed we use these constants:
                                //      CUBLAS_OP_N => non-transpose,
                                //      CUBLAS_OP_T => transpose,
                                //      CUBLAS_OP_C => transpose conjugate

                                float alpha = 1.f, beta = 0.f;

                                // We want to calculate A = J' * W * J : (6x6) matrix
                                //      W = IdentityMatrix => A = J' * J : (6x6) matrix
                                //          using cuBLAS: A = alpha * J' * J + beta * A
                                if(!useWeights)
                                        stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 6, 6, n, &alpha, d_J, n, d_J, n, &beta, d_A, 6);
                                else
                                        stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 6, 6, n, &alpha, d_JTW, n, d_J, n, &beta, d_A, 6);

                                if (stat != CUBLAS_STATUS_SUCCESS) {
                                        printf ("\n\n!----------cuBLAS matrix multiplication: A = J'*J FAILED!----------!\n\n");
                                        // return EXIT_FAILURE;
                                }
                                else {
                                        //printf ("\n\n!----------cuBLAS matrix multiplication: A = J'*J SUCCESSFUL!----------!\n\n");
                                        cudaMemcpy(A.data(), d_A, 6*6*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
                                }
                                //std::cout << "Matrix A: \n" << A << std::endl;
                                // We want to calculate b = J' * W * r : (6x1) vector
                                //      W = IdentityMatrix => b = J' * r : (6x1) vector
                                //          using cuBLAS: b = alpha * J' * r + beta * b
                                if(!useWeights)
                                        stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 6, 1, n, &alpha, d_J, n, d_r, n, &beta, d_b, 6);
                                else
                                        stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 6, 1, n, &alpha, d_JTW, n, d_r, n, &beta, d_b, 6);

                                if (stat != CUBLAS_STATUS_SUCCESS) {
                                        printf ("\n\n!----------cuBLAS matrix multiplication: b = J'*r FAILED!----------!\n\n");
                                        // return EXIT_FAILURE;
                                }
                                else {cudaMemcpy(b.data(), d_b, 6*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;}

                                // Solving the linear system: A * xi_delta = b, using the Eigen library.
                                xi_delta = (A.ldlt().solve(-b)); // Solve using Cholesky LDLT decomposition

                                // Convert the twist coordinates in xi & xi_delta to transformation matrices using Lie algebra
                                // Convert the combined transformation matrix back to twist coordinates
                                xi = lieLog(lieExp(xi_delta) * lieExp(xi));

                                // Calculate the mean error from the residuals. sum(errors) = r' * r => mean(sum(errors)) = (r' * r )/n
                                float error;
                                cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, n, &alpha, d_r, n, d_r, n, &beta, d_error, 1);
                                cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
                                error /= n;
                                // if the change in error is very small, break iterations loop and go to higher resolution in pyramid
                                if (error / error_prev > 0.995 || error == 0) break;

                                error_prev = error;
                        }

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
        xi_total = lieLog(lieExp(xi_total)*lieExp(xi).inverse());
        return xi_total;
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
bool useCUBLAS;   // TODO: option to NOT use cuBLAS
bool useWeights;  // Whether or not weighting by student t-distribution should be used
Matrix6f A; // A = J' * W * J
Vector6f b; // b = J' * r
float SIGMA_INITIAL = 0.025f;
float VARIANCE_INITIAL = 0.000625f;

cublasHandle_t handle; // not used if useCUBLAS = false
cublasStatus_t stat;

// device variables
float *d_x_prime; // 3D x position in the second frame
float *d_y_prime; // 3D y position in the second frame
float *d_z_prime; // 3D z position in the second frame
float *d_u_warped;  // warped x position of every pixel in the first image onto the second image
float *d_v_warped;  // warped y position of every pixel in the first image onto the second image
float *d_J;   // device Jacobian array for ALL residuals
float *d_JTW; // device Jacobian transpose * weighting function array
float *d_W;   // device Weight array
float *d_r;   // device residuals array
float *d_b;   // device linear system inhomogeneous term array
float *d_A;   // device linear system matrix array
// float *d_visualResidual;   // device image residuals array      TODO: needed? just for debugging?
float *d_error;   // mean squares error of residual
float *d_error_prev;   // mean squares error of residual in previous iteration
//float *d_sigma;
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
Vector6f xi_delta;

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
        int pitch = level_width*sizeof(float);
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

// calculate weights from current residuals and update d_jtw
void calculate_jtw(int level, int level_width, int level_height, float &variance_init) {
          // Block = 2D array of threads
          dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

          // Grid = 2D array of blocks
          int   gridSizeX = (level_width  + dimBlock.x-1) / dimBlock.x;
          int   gridSizeY = (level_height + dimBlock.y-1) / dimBlock.y;
          int   n = level_width * level_height;
          dim3  dimGrid( gridSizeX, gridSizeY, 1 );

          // updates the sigma and JTW

          float variance = variance_init;
          //float sigma = sigma_init;
          int iterations = 0;
          do{
                  variance_init = variance;
                  d_calculate_variance <<< dimGrid, dimBlock >>> (d_W, d_r, level_width, level_height, variance_init); CUDA_CHECK;

                  //compute new variance:
                  cublasSasum(handle, n , d_W, 1 , &variance);
                  variance /= n;
                  iterations ++;
          } while(std::abs( 1/(variance) -  1/(variance_init) ) > 1e-3 && iterations < 5);

          variance_init = variance;
          d_calculate_weights <<< dimGrid, dimBlock >>> (d_W, d_r, level_width, level_height, variance_init); CUDA_CHECK;


          //cout << "Tdist estimate scale in  " << iterations << " iterations" << endl;

          dim3 dimBlockJ(g_CUDA_blockSize2DX, 6,1);
          gridSizeX = (n + dimBlock.x-1) / dimBlock.x;
          dim3 dimGridJ( gridSizeX, 1, 1 );

          d_calculate_jtw <<<dimGridJ, dimBlockJ>>>(d_JTW, d_J, d_W, n, 6); CUDA_CHECK;
}


void allocateGPUMemory() {
        cudaMalloc(&d_J,      6*width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_JTW,    6*width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_W,        width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_r,        width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_x_prime,  width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_y_prime,  width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_z_prime,  width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_u_warped, width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_v_warped, width*height*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_b,                   6*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_A,                 6*6*sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_error,                 sizeof(float)); CUDA_CHECK;
        //cudaMalloc(&d_sigma,                 sizeof(float)); CUDA_CHECK;
        // cudaMalloc(&d_visualResidual, width*height*sizeof(float)); CUDA_CHECK;
        // cudaMalloc(&d_n, sizeof(int)); CUDA_CHECK;

        // allocate pyramid vector levels in device memory
        for (int level = 0; level <= maxLevel; level++) {
                int level_width = width / (1 << level); // calculating bitwise the succesive powers of 2
                int level_height = height / (1 << level);
                cudaMalloc(&d_cur [level].gray,    level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[level].gray,    level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_cur [level].depth,   level_width*level_height*sizeof(float)); CUDA_CHECK;
                cudaMalloc(&d_prev[level].depth,   level_width*level_height*sizeof(float)); CUDA_CHECK;
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
        cudaFree(d_J);        CUDA_CHECK;
        cudaFree(d_JTW);      CUDA_CHECK;
        cudaFree(d_W);        CUDA_CHECK;
        cudaFree(d_r);        CUDA_CHECK;
        cudaFree(d_x_prime);  CUDA_CHECK;
        cudaFree(d_y_prime);  CUDA_CHECK;
        cudaFree(d_z_prime);  CUDA_CHECK;
        cudaFree(d_u_warped); CUDA_CHECK;
        cudaFree(d_v_warped); CUDA_CHECK;
        cudaFree(d_b);        CUDA_CHECK;
        cudaFree(d_A);        CUDA_CHECK;
        cudaFree(d_error);    CUDA_CHECK;
        //cudaFree(d_sigma);    CUDA_CHECK;
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

#include <Eigen/Dense>
#include "preprocessing.cuh"
#include "lieAlgebra.hpp"
#include "alignment.cuh"
// #include <string> //only needed for our 'debugging'

enum SolvingMethod { GAUSS_NEWTON, LEVENBERG_MARQUARDT, GRADIENT_DESCENT };
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
        int maxIterationsPerLevel = 20, // small for development, 20 later
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
        xi(Vector6f::Zero()),
        A(Matrix6f::Zero()),
        b(Vector6f::Zero())
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
                std::cout << "Level: " << level << std::endl;

                // calculate size of image in current level
                int level_width = width / (1 << level); // calculating bitwise the succesive powers of 2
                int level_height = height / (1 << level);
                // set d_prev_err to big float number      TODO: directly in GPU?
                cudaMemcpy ( d_error_prev, &BIG_FLOAT, sizeof(float), cudaMemcpyHostToDevice ); CUDA_CHECK;

                bind_textures(level, level_width, level_height); // used for interpolation in the current image

                // for a maximum number of iterations per level
                for (int i = 0; i < maxIterationsPerLevel; i++) {
                        std::cout << "Iteration #" << i ;

                        // Calculate Rotation matrix and translation vector: CPU operation
                        convertSE3ToT(xi, R, t);

                        // calculate RK_inv = R*K_inv in CPU and copy to constant memory (both rotation and translation)
                        RK_inv = R * K_inv_pyr[level];
                        cudaMemcpyToSymbol (const_RK_inv, RK_inv.data(), 9*sizeof(float)); CUDA_CHECK;
                        cudaMemcpyToSymbol (const_translation, t.data(), 3*sizeof(float)); CUDA_CHECK;

                        // transform_points: CUDA operation
                        transform_points(level, level_width, level_height);

                        // parallel CUDA kernels: two streams
                                // calculate_jacobian J(n,6)  // calculate_residuals r_xi(n,1) and error (mean squares of r_xi)
                                                              // calculate_weights W(n,1)
                        stream1 =0; stream2 = 0;
                        calculate_residuals(level, level_width, level_height); //, stream2); // +3ms
                        calculate_weights(level, level_width, level_height); //, stream2);   // +1ms
                        calculate_jacobian(level, level_width, level_height); //, stream1);  // +7ms
                        check_error(level, level_width, level_height); //, stream2); // +6m      // cudamalloc is synchronous, so this does not work fine with parallel
                        cudaDeviceSynchronize(); CUDA_CHECK; // needed here?

                        // parallel CUDA kernels: two streams
                                // calculate A(6,6) = J.T * W * J   // calculate B(6,1) = -J.T * W * r
                                // TODO: best order to calculate the previous multiplications. J.T * width first? (used by both) or all together avoiding reads?
                                // probably A and be separated are better. Less R&W. Less code.
                        calculate_A ( level, level_width, level_height ); //, stream1 );
                        cudaMemcpy ( A.data(), d_A, 6*6*sizeof(float), cudaMemcpyDeviceToHost);

                        calculate_b ( level, level_width, level_height ); //, stream1 );
                        cudaMemcpy ( b.data(), d_b, 6*sizeof(float), cudaMemcpyDeviceToHost);

                        // solve linear system: A * delta_xi = b; with solver of Eigen library: CPU operation.      TODO: Faster to solve directly in GPU?
                        xi_delta = -(A.ldlt().solve(b)); // Solve using Cholesky LDLT decomposition

                        // Convert the twist coordinates in xi & xi_delta to transformation matrices using Lie algebra
                        // Convert the combined transformation matrix back to twist coordinates
                        xi = lieLog(lieExp(xi_delta) * lieExp(xi));

                        cudaMemcpy(&error_ratio, d_error_ratio, sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
                        // if the change in error is very small, break iterations loop and go to higher resolution in pyramid
                        std::cout << "  error ratio: " << error_ratio << std::endl;
                        if (error_ratio > 0.995) break;

                        // save error value for next iteration: d_prev_err = d_err
                        cudaMemcpy(d_error_prev, d_error, sizeof(float), cudaMemcpyDeviceToDevice); CUDA_CHECK;

                        // // DEBUG display image
                        // cv::Mat mTest(level_height, level_width, CV_32FC1);
                        // float *res = new float[level_width*level_height];
                        // cudaMemcpy(res, d_r, level_width*level_height*sizeof(float), cudaMemcpyDeviceToHost);
                        // convert_layered_to_mat(mTest, res);
                        // double min, max;
                        // cv::minMaxLoc(mTest, &min, &max);
                        // showImage( "Depth: " + std::to_string(level), mTest/max, 100, 100); cv::waitKey(0);
                        // // DEBUG print out values
                        // if (i==0) {
                        //     // dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );
                        //     //
                        //     // // Grid = 2D array of blocks
                        //     // // gridSizeX = ceil( width / nBlocksX )
                        //     // // gridSizeY = ceil( height / nBlocksX )
                        //     // int   gridSizeX = (level_width  + dimBlock.x-1) / dimBlock.x;
                        //     // int   gridSizeY = (level_height + dimBlock.y-1) / dimBlock.y;
                        //     // dim3  dimGrid( gridSizeX, gridSizeY, 1 );
                        //     // print_device_array <<< dimGrid, dimBlock >>> (d_W, level_width, level_height , level );
                        //
                        //     std::cout << A << std::endl;
                        //     std::cout << b << std::endl;
                        // }

                }

                unbind_textures();  // leave texture references free for binding at level below
        }

        // swap the pointers so we place image in the correct buffer next time this function is called
        temp_swap = d_cur; d_cur = d_prev; d_prev = temp_swap;
        // TODO? accumulate total_xi: total_xi = log(exp(xi)*exp(total_xi))
        return xi;
}




private:
// structure saving image data for each level of a pyramid: gray & depth images, and derivatives of gray
struct PyramidLevel { float *gray, *depth, *gray_dx, *gray_dy; };
// host parameters
SolvingMethod solvingMethod;   // enum type of possible solving methods
int maxIterationsPerLevel;
int maxLevel;
int minLevel;   // For speed. Used if the highest precision is not required
int width;   // width of the first frame (and all frames)
int height;   // height of the first frame (and all frames)
ResidualWeight weightType;   // enum type of possible residual weighting. Defined in *_jacobian.cuh
float BIG_FLOAT = 1e10;
float error_ratio;
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
float *d_W;   // device image residuals array      TODO: needed? just for debugging?
float *d_error;   // mean squares error of residual
float *d_error_prev;   // mean squares error of residual in previous iteration
float *d_error_ratio;   // error / error_prev
std::vector<PyramidLevel> d_cur;   // current vector of pointers to device pyramid level structures
std::vector<PyramidLevel> d_prev;   // previous vector of pointers to device pyramid level structures
std::vector<PyramidLevel> temp_swap;
Matrix3f R;
Matrix3f RK_inv;
Vector3f t;
Matrix6f A;
Vector6f b;

// host variables
// watch out: Eigen::Matrix are stored column wise
std::vector<Matrix3f> K_pyr;   // stores projection matrix and downsampled version (intrinsic camera properties)
std::vector<Matrix3f> K_inv_pyr;   // inverse K_pyr
Vector6f xi_delta;
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
                cudaGetSymbolAddress((void **)&p_const, const_K_pyr); CUDA_CHECK;
                // write each level at an incremented device pointer position
                cudaMemcpy(p_const + 9*level, K_pyr[level].data(), 9*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
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
                imresize_CUDA(d_img[level-1].gray, d_img[level].gray, 2*level_width, 2*level_height, level_width, level_height, 1, false); CUDA_CHECK;
                imresize_CUDA(d_img[level-1].depth, d_img[level].depth, 2*level_width, 2*level_height, level_width, level_height, 1, true); CUDA_CHECK; // TODO: Check properly if isDepthImage is working. Looks like it does
        }

        for (int level = 0; level <= maxLevel; level++) {
                level_width = width / (1 << level); // bitwise operator to divide by 2**level
                level_height = height / (1 << level);
                // compute derivatives!!
                image_derivatives_CUDA(d_img[level].gray,d_img[level].gray_dx,d_img[level].gray_dy,level_width,level_height); CUDA_CHECK;
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
        texRef_grayImg.normalized = false; CUDA_CHECK;
        texRef_grayImg.filterMode = cudaFilterModeLinear; CUDA_CHECK;

        texRef_gray_dx.normalized = false; CUDA_CHECK;
        texRef_gray_dx.filterMode = cudaFilterModeLinear; CUDA_CHECK;

        texRef_gray_dy.normalized = false; CUDA_CHECK;
        texRef_gray_dy.filterMode = cudaFilterModeLinear; CUDA_CHECK;
}

void bind_textures(int level, int level_width, int level_height) {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); // number of bits for each texture
        int pitch = level_width * sizeof(float);
        cudaBindTexture2D(NULL, &texRef_grayImg, d_cur[level].gray, &desc, level_width, level_height, pitch); CUDA_CHECK;
        cudaBindTexture2D(NULL, &texRef_gray_dx, d_cur[level].gray_dx, &desc, level_width, level_height, pitch); CUDA_CHECK;
        cudaBindTexture2D(NULL, &texRef_gray_dy, d_cur[level].gray_dy, &desc, level_width, level_height, pitch); CUDA_CHECK;
}

void unbind_textures() {
        cudaUnbindTexture(texRef_grayImg); CUDA_CHECK;
        cudaUnbindTexture(texRef_gray_dx); CUDA_CHECK;
        cudaUnbindTexture(texRef_gray_dy); CUDA_CHECK;
}

/**
 * Calculates the transformed positions on the second frame for every pixel in the first
 */
void transform_points(int level, int level_width, int level_height) {
          // Block = 2D array of threads
          dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

          // Grid = 2D array of blocks
            // gridSizeX = ceil( width / nBlocksX )
          int   gridSizeX = (level_width  + dimBlock.x-1) / dimBlock.x;
            // gridSizeY = ceil( height / nBlocksX )
          int   gridSizeY = (level_height + dimBlock.y-1) / dimBlock.y;
          dim3  dimGrid( gridSizeX, gridSizeY, 1 );

          d_transform_points <<< dimGrid, dimBlock >>> (d_x_prime, d_y_prime, d_z_prime, d_u_warped, d_v_warped, d_prev[level].depth, level_width, level_height, level); CUDA_CHECK;
}

/**
 * Calculates the jacobian at each pixel
 */
void calculate_jacobian(int level, int level_width, int level_height, cudaStream_t stream=0) {
          // Block = 2D array of threads
          dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

          // Grid = 2D array of blocks
            // gridSizeX = ceil( width / nBlocksX )
          int   gridSizeX = (level_width  + dimBlock.x-1) / dimBlock.x;
            // gridSizeY = ceil( height / nBlocksX )
          int   gridSizeY = (level_height + dimBlock.y-1) / dimBlock.y;
          dim3  dimGrid( gridSizeX, gridSizeY, 1 );

          d_calculate_jacobian <<< dimGrid, dimBlock, 0, stream >>> (d_J, d_x_prime, d_y_prime, d_z_prime, d_u_warped, d_v_warped, level_width, level_height, level); // texture is accessed directly. No argument needed
        //   CUDA_CHECK;
}

/**
 * Calculates the residual at each pixel
 */
void calculate_residuals(int level, int level_width, int level_height, cudaStream_t stream=0) {
          // Block = 2D array of threads
          dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

          // Grid = 2D array of blocks
            // gridSizeX = ceil( width / nBlocksX )
          int   gridSizeX = (level_width  + dimBlock.x-1) / dimBlock.x;
            // gridSizeY = ceil( height / nBlocksX )
          int   gridSizeY = (level_height + dimBlock.y-1) / dimBlock.y;
          dim3  dimGrid( gridSizeX, gridSizeY, 1 );

          d_calculate_residuals <<< dimGrid, dimBlock, 0, stream >>> (d_r, d_prev[level].gray, d_u_warped, d_v_warped, level_width, level_height, level); // texture is accessed directly. No argument needed
        //   CUDA_CHECK;
}

/**
 * Calculates the error and compares with the previous one
 */
void check_error(int level, int level_width, int level_height, cudaStream_t stream=0) {
        int size = level_width * level_height;
        // threads per block equals maximum possible
        int blocklength = 1024;
        // number of needed blocsk is ceil(l_w * l_h / blocklength)
        int nblocks = (size + blocklength -1)/blocklength;

        // alloc auxiliar array
        float *d_aux = NULL;
        cudaMalloc(&d_aux, nblocks*sizeof(float));// CUDA_CHECK;  // to avoid overwriting the residuals array
        // alloc pointer for swapping
        float *d_swap;

        dim3 block = dim3(blocklength,1,1);
        dim3 grid = dim3(nblocks, 1, 1 );

        // first reduction. Return reduction by 1024 in aux
        d_squares_sum <<< grid, block, blocklength*sizeof(float), stream >>> (d_r, d_aux, size);// CUDA_CHECK;

        // now aux is the input, and size is its size
        size = nblocks;
        // nblocks is the output of the next reduction
        nblocks = (size + blocklength -1)/blocklength;
        grid = dim3(nblocks, 1, 1 );

        // alloc another auxiliar array
        float *d_aux2 = NULL;
        cudaMalloc(&d_aux2, nblocks*sizeof(float));// CUDA_CHECK;  // to avoid overwriting the residuals array

        // reductions until size 1
        while (true) {
                d_sum <<< grid, block, blocklength*sizeof(float), stream >>> (d_aux, d_aux2, size);// CUDA_CHECK;
                // if no more reductions are needed, break. Result is in d_aux2
                if (nblocks == 1) break;

                // // copy d_aux2 to the beginning of d_aux
                // cudaMemcpy( d_aux, d_aux2, nblocks*sizeof(float), cudaMemcpyDeviceToDevice ); CUDA_CHECK;
                // swap pointers of aux and aux2
                d_swap = d_aux; d_aux = d_aux2; d_aux2 = d_swap;
                // now aux2 is the input, copied into d_aux, and size is its size
                size = nblocks;
                // nblocks is the output of the next reduction
                nblocks = (size + blocklength -1)/blocklength;
                grid = dim3(nblocks, 1, 1 );
        }

        // d_check_error <<< dimGrid, dimBlock, 0, stream >>> (d_error, d_error_prev, d_error_ratio, d_r, level_width, level_height, level);
        d_check_error <<< 1, 1, 0, stream >>> (d_aux2, d_error, d_error_prev, d_error_ratio, level_width, level_height);// CUDA_CHECK;

        cudaFree(d_aux); cudaFree(d_aux2);

        // // DEBUG
        // float test;
        // cudaMemcpy( &test, d_error, sizeof(float), cudaMemcpyDeviceToHost );
        // std::cout << "elegant: " << test << std::endl;
}

/**
 * Calculates the weights
 */
void calculate_weights(int level, int level_width, int level_height, cudaStream_t stream=0) {
        // Block = 2D array of threads
        dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );

        // Grid = 2D array of blocks
            // gridSizeX = ceil( width / nBlocksX )
        int   gridSizeX = (level_width  + dimBlock.x-1) / dimBlock.x;
            // gridSizeY = ceil( height / nBlocksX )
        int   gridSizeY = (level_height + dimBlock.y-1) / dimBlock.y;
        dim3  dimGrid( gridSizeX, gridSizeY, 1 );

        d_set_uniform_weights <<< dimGrid, dimBlock, 0, stream >>> (d_W, level_width, level_height); //CUDA_CHECK;
}

/**
 * Calculate matrix A of the linear system
 */
void calculate_A (int level, int level_width, int level_height, cudaStream_t stream=0) {
        int size = level_width * level_height;
        // threads per block equals maximum possible
        int blocklength = 1024;
        // number of needed blocks in 3D
            // size of A is 6x6
        int numblocksX = 6;
        int numblocksY = 6;
            // number of subproducts for each element of A = ceil(size/blocklength)
        int numblocksZ = (size + blocklength -1)/blocklength;
            // total
        int gridVolume = numblocksX*numblocksY*numblocksZ;

        if (numblocksZ > 1024) std::cout << "Warning: calculate_A reduction is not ready for this image size" << std::endl;
        // std::cout << size << " " << blocklength << " " << numblocksX << " " << numblocksY << " " << numblocksZ << " " << gridVolume << std::endl;

        // alloc auxiliar array for all the matrix sub-products forming the 3D "volume" (stored in an array) previous to computing A
        float *d_pre_A = NULL;
        cudaMalloc(&d_pre_A, gridVolume*sizeof(float)); CUDA_CHECK;  // to avoid overwriting the residuals array

        dim3 block = dim3(blocklength,1,1);
        dim3 grid = dim3( numblocksX, numblocksY, numblocksZ );

        // J'*W*J pre-calculation, yet to be reduced. Gets stored into d_pre_A (previous to A)
        d_product_JacT_W_Jac <<< grid, block, blocklength*sizeof(float), stream >>> (d_pre_A, d_J, d_W, size); CUDA_CHECK;

        // now aux is the input, and size is its size
        size = numblocksZ;
        // d_A is the output of the next reduction, is 6x6
        grid = dim3( numblocksX, numblocksY, 1 );

        d_reduce_pre_A_to_A <<< grid, block, blocklength*sizeof(float), stream >>> (d_A, d_pre_A, size); CUDA_CHECK;

        // reductions until size 1 // not implemented, required for larger images TODO
        // while (true) {
        //         d_sum <<< grid, block, blocklength*sizeof(float), stream >>> (d_aux, d_aux2, size); CUDA_CHECK;
        //         // if no more reductions are needed, break. Result is in d_aux2
        //         if (nblocks == 1) break;
        //
        //         // // copy d_aux2 to the beginning of d_aux
        //         // cudaMemcpy( d_aux, d_aux2, nblocks*sizeof(float), cudaMemcpyDeviceToDevice ); CUDA_CHECK;
        //         // swap pointers of aux and aux2
        //         d_swap = d_aux; d_aux = d_aux2; d_aux2 = d_swap;
        //         // now aux2 is the input, copied into d_aux, and size is its size
        //         size = nblocks;
        //         // nblocks is the output of the next reduction
        //         nblocks = (size + blocklength -1)/blocklength;
        //         grid = dim3(nblocks, 1, 1 );
        // }
}

/**
 * Calculate array b of the linear system
 */
void calculate_b (int level, int level_width, int level_height, cudaStream_t stream=0) {
        int size = level_width * level_height;
        // threads per block equals maximum possible
        int blocklength = 1024;
        // number of needed blocks in 3D. Now it is actually 2D becaus numblocksY=1, but it keeps the structure of calculate_A, so it is 3D.
            // size of b is 6x1
        int numblocksX = 6;
        int numblocksY = 1;
            // number of subproducts for each element of b = ceil(size/blocklength)
        int numblocksZ = (size + blocklength -1)/blocklength;
            // total
        int gridVolume = numblocksX*numblocksY*numblocksZ;

        if (numblocksZ > 1024) std::cout << "Warning: calculate_A reduction is not ready for this image size" << std::endl;
        // std::cout << size << " " << blocklength << " " << numblocksX << " " << numblocksY << " " << numblocksZ << " " << gridVolume << std::endl;

        // alloc auxiliar array for all the matrix sub-products forming the 3D "volume" (stored in an array) previous to computing A
        float *d_pre_b = NULL;
        cudaMalloc(&d_pre_b, gridVolume*sizeof(float)); CUDA_CHECK;  // to avoid overwriting the residuals array

        dim3 block = dim3(blocklength,1,1);
        dim3 grid = dim3( numblocksX, numblocksY, numblocksZ );

        // J'*W*J pre-calculation, yet to be reduced. Gets stored into d_pre_b (previous to A)
        d_product_JacT_W_res <<< grid, block, blocklength*sizeof(float), stream >>> (d_pre_b, d_J, d_W, d_r, size); CUDA_CHECK;

        // now aux is the input, and size is its size
        size = numblocksZ;
        // d_b is the output of the next reduction, is 6x1
        grid = dim3( numblocksX, numblocksY, 1 );   // actually (numblocksX, 1, 1)

        d_reduce_pre_b_to_b <<< grid, block, blocklength*sizeof(float), stream >>> (d_b, d_pre_b, size); CUDA_CHECK;

        // reductions until size 1 // not implemented, required for larger images TODO
        // while (true) {
        //         d_sum <<< grid, block, blocklength*sizeof(float), stream >>> (d_aux, d_aux2, size); CUDA_CHECK;
        //         // if no more reductions are needed, break. Result is in d_aux2
        //         if (nblocks == 1) break;
        //
        //         // // copy d_aux2 to the beginning of d_aux
        //         // cudaMemcpy( d_aux, d_aux2, nblocks*sizeof(float), cudaMemcpyDeviceToDevice ); CUDA_CHECK;
        //         // swap pointers of aux and aux2
        //         d_swap = d_aux; d_aux = d_aux2; d_aux2 = d_swap;
        //         // now aux2 is the input, copied into d_aux, and size is its size
        //         size = nblocks;
        //         // nblocks is the output of the next reduction
        //         nblocks = (size + blocklength -1)/blocklength;
        //         grid = dim3(nblocks, 1, 1 );
        // }
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
        cudaMalloc(&d_error_prev, sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_error_ratio, sizeof(float)); CUDA_CHECK;
        cudaMalloc(&d_W, width*height*sizeof(float)); CUDA_CHECK;
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
        cudaFree(d_error_prev); CUDA_CHECK;
        cudaFree(d_error_ratio); CUDA_CHECK;
        cudaFree(d_W); CUDA_CHECK;
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

#include <cuda_runtime.h>
#include <stdio.h>
// #include <Eigen/Dense>  //TODO: viable to use Eigen in CUDA?


__global__ void d_transform_points( float *x_prime,
                                    float *y_prime,
                                    float *z_prime,
                                    float *u_warped,    //used as valid/non-valid mask too
                                    float *v_warped,    //used as valid/non-valid mask too
                                    const float *depthImg,
                                    const int width,
                                    const int height,
                                    const int level ) {
        // Get the 2D-coordinate of the pixel of the current thread
        const int   x = blockIdx.x * blockDim.x + threadIdx.x;
        const int   y = blockIdx.y * blockDim.y + threadIdx.y;
        const int pos = x + y * width;

        // If the 2D position is outside the image, do nothing
        if ( (x >= width) || (y >= height) )
                return;

        // if the depth value is not valid
        if ( (depthImg[pos] == 0) ) {
                u_warped[pos] = -1; // mark as not valid
                v_warped[pos] = -1; // mark as not valid
                return;
        }

        // get 3D point: p=(u*d, v*d, d) // u, v are the camera coordinates
        float p[3] = { x * depthImg[pos],
                       y * depthImg[pos],
                           depthImg[pos] };

        // unproject and transform: aux = RK_inv * p + t      // unproyect from camera: p = K_inv * p
                                                              // transform: p = R * p + t
        // matrices in constant memory are stored column-wise
        float aux[3];  // auxiliar variable
        for (int i = 0; i < 3; i++) {
                aux[i] = const_translation[i]
                         + p[0] * const_RK_inv[0 + i]
                         + p[1] * const_RK_inv[3 + i]
                         + p[2] * const_RK_inv[6 + i];
        }
        x_prime[pos] = aux[0];  y_prime[pos] = aux[1];  z_prime[pos] = aux[2];  // TODO: watch out if z_prime is 0. Looks unlikely...

        // proyect to camera: p = K * aux
        for (int i = 0; i < 3; i++) {
                p[i] =   aux[0] * const_K_pyr[0 + i + 9*level]
                       + aux[1] * const_K_pyr[3 + i + 9*level]
                       + aux[2] * const_K_pyr[6 + i + 9*level];
        }

        // get 2D camera coordinates in second frame: u2 = x/z; v2 = y/z
        // store (x',y') position for each (x,y)
        u_warped[pos] = p[0] / p[2];
        v_warped[pos] = p[1] / p[2];

        // if (x', y') is out of bounds in the second frame (not interpolable)
        if (    (u_warped[pos] < 0)
             || (u_warped[pos] > width-1)
             || (v_warped[pos] < 0)
             || (v_warped[pos] > height-1) ) {
                u_warped[pos] = -1; // mark as not valid
                v_warped[pos] = -1; // mark as not valid
        }

        // const_K_pyr DEBUG
        // if (x == 0 && y == 0 && level == 0) {
        //     for (int i = 0; i < 9*MAX_LEVELS; i++)
        //         printf("%4.2f \n", const_K_pyr[i]);
        // }
}





__global__ void d_calculate_jacobian( float *J,
                                    const float *x_prime,
                                    const float *y_prime,
                                    const float *z_prime,
                                    const float *u_warped,   // This is -1 for non-valid points
                                    const float *v_warped,   // This is -1 for non-valid points
                                    const int width,
                                    const int height,
                                    const int level ) {
        // Get the 2D-coordinate of the pixel of the current thread
        const int   x = blockIdx.x * blockDim.x + threadIdx.x;
        const int   y = blockIdx.y * blockDim.y + threadIdx.y;
        const int pos = x + y * width;

        // If the 2D position is outside the first image do nothing
        if ( (x >= width) || (y >= height) )
                return;
        // if the projection is outside the second frame gray values, set residual to 0
        if ( (u_warped[pos] < 0) ) {
                for (int i = 0; i < 6; i++)
                    J[pos + i*width*height] = 0.0f;
                return;
        }

        // dxfx is the image gradient in x direction times the fx of the intrinsic camera calibration
        float dxfx, dyfy;   // factors common to all jacobian positions
        dxfx = tex2D( texRef_gray_dx, u_warped[pos], v_warped[pos] ) * const_K_pyr[0 + 9*level];
        dyfy = tex2D( texRef_gray_dy, u_warped[pos], v_warped[pos] ) * const_K_pyr[4 + 9*level];
        // // DEBUG
        // if (dxfx != dxfx)
        //              printf ("dxfx: pix %d intrp %f x %d y %d u %f v %f w %d h %d l %d \n", pos, tex2D( texRef_gray_dx, u_warped[pos], v_warped[pos] ), x, y, u_warped[pos], v_warped[pos], width, height, level );
        // if (dyfy != dyfy)
        //              printf ("dyfy: pix %d intrp %f x %d y %d u %f v %f w %d h %d l %d \n", pos, tex2D( texRef_gray_dy, u_warped[pos], v_warped[pos] ), x, y, u_warped[pos], v_warped[pos], width, height, level );
        // if (dxfx != dxfx)
        //              printf ("%d %f %d %d %f %f %d %d %d \n", pos, tex2D( texRef_gray_dx, u_warped[pos], v_warped[pos] ), x, y, u_warped[pos], v_warped[pos], width, height, level );
        // if (dyfy != dyfy)
        //              printf ("%d %f %d %d %f %f %d %d %d \n", pos, tex2D( texRef_gray_dy, u_warped[pos], v_warped[pos] ), x, y, u_warped[pos], v_warped[pos], width, height, level );

        float xp, yp, zp;   // for easyness of reading and debugging
        xp = x_prime[pos]; yp = y_prime[pos]; zp = z_prime[pos];

        if (zp == 0) printf("WARRNING! DIVIDE BY ZERO!!!"); // TODO

        J[pos + 0*width*height] = - dxfx / zp;
        J[pos + 1*width*height] = - dyfy / zp;
        J[pos + 2*width*height] = + ( dxfx*xp + dyfy*yp )
                                    / ( zp * zp );
        J[pos + 3*width*height] = + ( dxfx*xp*yp + dyfy*yp*yp )
                                    / ( zp * zp )
                                  + dyfy;
        J[pos + 4*width*height] = - ( dyfy*xp*yp + dxfx*xp*xp )
                                    / ( zp * zp )
                                  - dxfx;
        J[pos + 5*width*height] = + dxfx*yp / zp
                                  - dyfy*xp / zp;

        // // DEBUG // mostly big numbers??? // sometimes dxfx *or* dyfy are wrong. Not both.
        // for (int i = 0; i < 6; i++) {
        //     if ( J[pos + i*width*height] !=  J[pos + i*width*height])
        //             printf ("pix %d var %d val %f \n", pos, i, J[pos + i*width*height] );
        // }

}

__global__ void d_calculate_residuals( float *r,
                                    const float *grayPrev,
                                    const float *u_warped,   // This is -1 for non-valid points
                                    const float *v_warped,   // This is -1 for non-valid points
                                    const int width,
                                    const int height,
                                    const int level ) {
        // Get the 2D-coordinate of the pixel of the current thread
        const int   x = blockIdx.x * blockDim.x + threadIdx.x;
        const int   y = blockIdx.y * blockDim.y + threadIdx.y;
        const int pos = x + y * width;

        // If the 2D position is outside the first image do nothing
        if ( (x >= width) || (y >= height) )
                return;
        // if the projection is outside the second frame gray values, set residual to 0
        if ( (u_warped[pos] < 0) ) {
                r[pos] = 0.0f;
                return;
        }

        r[pos] = grayPrev[pos] - tex2D( texRef_grayImg, u_warped[pos], v_warped[pos] );
}

__global__ void d_set_uniform_weights( float *W,
                                      const int width,
                                      const int height ) {
        // Get the 2D-coordinate of the pixel of the current thread
        const int   x = blockIdx.x * blockDim.x + threadIdx.x;
        const int   y = blockIdx.y * blockDim.y + threadIdx.y;
        const int pos = x + y * width;

        // If the 2D position is outside the first image or the projection outside the second, do nothing
        if ( (x >= width) || (y >= height) )
                return;

        W[pos] = 1.0;
}

__global__ void d_check_error ( float *square_sum,  // square root of the squares of the residuals
                               float *error,
                               float *error_prev,
                               float *error_ratio,
                               const int width,
                               const int height ) {
        *error = *square_sum / (width*height);
        *error_ratio = *error / *error_prev;
}

__global__ void d_squares_sum(float *input, float *results, int n) {
        extern __shared__ float sdata[];
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int tx = threadIdx.x;
        // load input into __shared__ memory
        if (i < n) {
                sdata[tx] = input[i] * input[i];
                __syncthreads();
        } else {
                sdata[tx] = 0;
                __syncthreads();
        }
        if (i < n) {
                // block-wide reduction in __shared__ mem
                for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
                        if(tx < offset) {
                                // add a partial sum upstream to our own
                                sdata[tx] += sdata[tx + offset];
                        }
                        __syncthreads();
                }
                // finally, thread 0 writes the result
                if(threadIdx.x == 0) {
                        // note that the result is per-block
                        // not per-thread
                        results[blockIdx.x] = sdata[0];
                }
        }
}

__global__ void d_sum(float *input, float *results, int n) {
        extern __shared__ float sdata[];
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int tx = threadIdx.x;
        // load input into __shared__ memory
        if (i < n) {
                sdata[tx] = input[i];
                __syncthreads();
        } else {
                sdata[tx] = 0;
                __syncthreads();
        }
        if (i < n) {
                // block-wide reduction in __shared__ mem
                for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
                        if(tx < offset) {
                                // add a partial sum upstream to our own
                                sdata[tx] += sdata[tx + offset];
                        }
                        __syncthreads();
                }
                // finally, thread 0 writes the result
                if(threadIdx.x == 0) {
                        // note that the result is per-block
                        // not per-thread
                        results[blockIdx.x] = sdata[0];
                }
        }
}

__global__ void d_product_JacT_W_Jac(   float *pre_A,
                                        const float *J,
                                        const float *W,
                                        const int level_size ) {
        extern __shared__ float sdata[];

        int rowJacT = blockIdx.x;   // row index of the transposed Jacobian to operate with
        int colJac = blockIdx.y;    // column index of the Jacobian to operate with
        int subBlockIdx = blockIdx.z;   // how far along each [ column of the Jacobian ]/[ row of the transposed Jacobian ] the operation starts for the block
        int tx = threadIdx.x;
        int idxJacT = tx + subBlockIdx * blockDim.x + rowJacT * level_size ;  // thread index at the d_J array for the Jacobian transposed
        int idxJac = tx + subBlockIdx * blockDim.x + colJac * level_size ;  // thread index at the d_J array for the Jacobian
        int idxW = tx + subBlockIdx * blockDim.x;   // thread index for the Weights

        // load input into __shared__ memory
        if ( idxW < level_size ) {  // check if W is out of bounds and simultaneously correct indexing of idxJac and idxJacT
                sdata[tx] = J[idxJacT] * W[idxW] * J[idxJac];   // J.T * W * J
                __syncthreads();
        } else {
                sdata[tx] = 0;
                __syncthreads();
        }
        if ( idxW < level_size ) {
                // block-wide reduction in __shared__ mem
                for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
                        if(tx < offset) {
                                // add a partial sum upstream to our own
                                sdata[tx] += sdata[tx + offset];
                        }
                        __syncthreads();
                }
                // finally, thread 0 writes the result
                if(threadIdx.x == 0) {
                        // note that the result is per-block
                        // not per-thread
                        pre_A[ blockIdx.z     // index along the dimension of pre_A to be later reduced to get A
                                + (blockIdx.x + blockIdx.y * 6)     // index inside A, which will be stored column-wise
                                        * gridDim.z    // size of each array to be reduced to get each A element. They are stored head to tail column-wise along pre_A
                             ] = sdata[0];
                }
        }
}

/**
 *
 * @param *sizeZ    number of pre_A floats to be reduced to get each element of A. It is the Z component of the 3D matrix pre_A (stored as a linear array)
 */
__global__ void d_reduce_pre_A_to_A(    float *A,
                                        float *pre_A,
                                        int sizeZ ) {
        extern __shared__ float sdata[];
        // blockIdx.x is the row of A
        // blockIdx.y is the column of A
        int idx = threadIdx.x + (blockIdx.x + blockIdx.y * 6) * sizeZ;    // position along pre_A of thread pixel to load in memory
        int tx = threadIdx.x;
        // load input into __shared__ memory
        if (idx < sizeZ) {
                sdata[tx] = pre_A[idx];
                __syncthreads();
        } else {
                sdata[tx] = 0;
                __syncthreads();
        }
        if (idx < sizeZ) {
                // block-wide reduction in __shared__ mem
                for(int offset = blockDim.x / 2; offset > 0; offset /= 2) {
                        if(tx < offset) {
                                // add a partial sum upstream to our own
                                sdata[tx] += sdata[tx + offset];
                        }
                        __syncthreads();
                }
                // finally, thread 0 writes the result
                if(threadIdx.x == 0) {
                        // note that the result is per-block
                        // not per-thread
                        A[ blockIdx.x + blockIdx.y * 6 ] = sdata[0];
                }
        }
}

// // DEBUG
// __global__ void print_device_array( float *arr, const int width, const int height, const int level) {
//         // Get the 2D-coordinate of the pixel of the current thread
//         const int   x = blockIdx.x * blockDim.x + threadIdx.x;
//         const int   y = blockIdx.y * blockDim.y + threadIdx.y;
//         const int pos = x + y * width;
//
//         if ( (x >= width) || (y >= height))
//                 return;
//
//         if( arr[pos] != arr[pos] )
//             printf("lev %d val %f pos %d ", level, arr[pos], pos);
//         if( arr[pos] == INFINITY || arr[pos] == -INFINITY )
//             printf("inf: lev %d val %f pos %d ", level, arr[pos], pos);
//
// }
//
// __global__ void print_device_array_J( float *arr, const int width, const int height, const int level) {
//         // Get the 2D-coordinate of the pixel of the current thread
//         const int   x = blockIdx.x * blockDim.x + threadIdx.x;
//         const int   y = blockIdx.y * blockDim.y + threadIdx.y;
//         const int pos = x + y * width;
//
//         if ( (x >= width) || (y >= height))
//                 return;
//         int posJ;
//         for (int i=0; i < 6; i++) {
//             posJ = pos + i*width*height;
//             if( arr[posJ] != arr[posJ] )
//                 printf("lev %d val %f pos %d \n", level, arr[posJ], posJ);
//             if( arr[posJ] == INFINITY || arr[posJ] == -INFINITY )
//                 printf("inf: lev %d val %f pos %d \n", level, arr[posJ], posJ);
//         }
// }

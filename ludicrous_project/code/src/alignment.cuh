#include <cuda_runtime.h>
#include <stdio.h>
// #include <Eigen/Dense>  //TODO: viable to use Eigen in CUDA?


__global__ void d_transform_points( float *x_prime, float *y_prime, const float *depthImg, int width, int height, int level ) {
        // Get the 2D-coordinate of the pixel of the current thread
        const int   x = blockIdx.x * blockDim.x + threadIdx.x;
        const int   y = blockIdx.y * blockDim.y + threadIdx.y;
        const int pos = x + y * width;

        // If the 2D position is outside the image, do nothing
        if ( (x >= width) || (y >= height) )
                return;

        // get 3D point: p=(u*d, v*d, d) // u, v are the camera coordinates
        float p[3] = { x * depthImg[pos],
                       y * depthImg[pos],
                           depthImg[pos] };

        // unproyect and transform: aux = RK_inv * p + t      // unproyect from camera: p = K_inv * p
                                                              // transform: p = R * p + t
        float aux[3];  // auxiliar variable
        for (int i = 0; i < 3; i++) {
                aux[i] = const_translation[i]
                         + p[0] * const_RK_inv[0 + i]
                         + p[1] * const_RK_inv[3 + i]
                         + p[2] * const_RK_inv[6 + i];
        }
        // proyect to camera: p = K * aux
        for (int i = 0; i < 3; i++) {
                p[i] = aux[0] * const_K_pyr[0 + i + 9*level]
                       + aux[1] * const_K_pyr[3 + i + 9*level]
                       + aux[2] * const_K_pyr[6 + i + 9*level];
        }

        // get 2D camera coordinates in second frame: u2 = x/z; v2 = y/z
        // store (x',y') position for each (x,y)
        x_prime[pos] = p[0] / p[2];
        y_prime[pos] = p[1] / p[2];

        // const_K_pyr DEBUG
        // if (x == 0 && y == 0 && level == 0) {
        //     for (int i = 0; i < 9*MAX_LEVELS; i++)
        //         printf("%4.2f \n", const_K_pyr[i]);
        // }
}

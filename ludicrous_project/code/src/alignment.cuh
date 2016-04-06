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
        x_prime[pos] = aux[0];  y_prime[pos] = aux[1];  z_prime[pos] = aux[2];

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

        // If the 2D position is outside the first image or the projection outside the second, do nothing
        if ( (x >= width) || (y >= height) )
                return;
        if ( u_warped[pos] < 0 ) {
                for(int i=0; i < 6 ; i++ )
                    J[pos + i*width*height]  = 0.0f;
                return;
        }

        // dxfx is the image gradient in x direction times the fx of the intrinsic camera calibration
        float dxfx, dyfy;   // factors common to all jacobian positions
        dxfx = tex2D( texRef_gray_dx, u_warped[pos], v_warped[pos] ) * const_K_pyr[0 + 9*level];
        dyfy = tex2D( texRef_gray_dy, u_warped[pos], v_warped[pos] ) * const_K_pyr[4 + 9*level];

        float xp, yp, zp;   // for easyness of reading and debugging
        xp = x_prime[pos]; yp = y_prime[pos]; zp = z_prime[pos];

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

        // If the 2D position is outside the first image or the projection outside the second, do nothing
        if ( (x >= width) || (y >= height))
                return;
        if ( u_warped[pos] < 0 ) {
                r[pos] = 0.0f;
                return;
        }

        r[pos] = grayPrev[pos] - tex2D( texRef_grayImg, u_warped[pos], v_warped[pos] );
}

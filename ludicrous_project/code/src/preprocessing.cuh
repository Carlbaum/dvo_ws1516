#pragma once
#include <Eigen/Dense>
#include "Exception.h"
#include <cuda_runtime.h>


Matrix3f downsampleK(const Matrix3f K) {
  Matrix3f K_d = K;
  K_d(0, 2) += 0.5f; K_d(1, 2) += 0.5f;
  K_d.topLeftCorner(2, 3) *= 0.5f;
  K_d(0, 2) -= 0.5f; K_d(1, 2) -= 0.5f;
  return K_d;
}

/**
 * Calculates the inverse of a 3x3 Matrix with the following 'shape'
 *
 *  | a   0   b |          | 1/a   0   -b/a |
 *  | 0   c   d |    =>    | 0    1/c  -d/c |
 *  | 0   0   1 |          | 0     0    1   |
 *
 * @param iKPy [description]
 * @param KPy  [description]
 * @param lvl  [description]
 */
Matrix3f invertKMat (const Eigen::Matrix3f K) {
    Matrix3f K_inv = K;
	K_inv <<   1.0f/K_inv(0,0)	, 0.0f				, -(K_inv(0,2)/K_inv(0,0)),
			   0.0f				, 1.0f/K_inv(1,1)	, -(K_inv(1,2)/K_inv(1,1)),
			   0.0f				, 0.0f				, 1.0f;
   return K_inv;
}


//############################################################################
__global__  void gaussFilter2D_horizontal_CUDA_kernel(
                                            const float  *data_in,
                                            float        *data_out,
                                            int          width,
                                            int          height,
                                            float        sigma,
                                            int          radius,
                                            int          borderMethod )
{
  // Get the 2D-coordinate of the pixel of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;


  //##########################################################################
  //  Padding #radius pixels around the 2D CUDA block into shared memory
  //##########################################################################

  // Copy data into shared memory
  // (Size gets determined by CUDA execution arguments)
  extern __shared__ float  data[];

  // Copy the pixelwise values into the shared memory
  //   p      = Current pixel's position in the global memory
  //   pLocal = Current pixel's position in the shared memory
  int     bWidthPadded = blockDim.x + 2*radius;
  int     pLocal       = threadIdx.y * bWidthPadded + threadIdx.x + radius;
  int     p            = y * width + x;



  // Directly copy all non boundary pixels
  data[ pLocal ] = data_in[ p ];

  //-------------------  Left border of the 2D CUDA block  -------------------
  if ( threadIdx.x == 0 )
  {
    for ( int ix=radius; ix>=0; ix-- )
    {
      int   posLocal = pLocal - ix;
      int   pos      = p      - ix;

      // Left border of the image
      if ( x-ix < 0 )
      {
        if ( borderMethod == BORDER_ZERO )            //Zero-padding
          data[ posLocal ] = 0;
        else if ( borderMethod == BORDER_REPLICATE )  //Copy nearest pixel
          data[ posLocal ] = data_in[ y*width+0 ];
      }
      else
        data[ posLocal ] = data_in[ pos ];            //Normal copy of the pixel
    }
  }
  //----------------------- Right border -----------------------
  if ( threadIdx.x == blockDim.x-1 )
  {
    for ( int ix=radius; ix>=0; ix-- )
    {
      int   posLocal = pLocal + ix;
      int   pos      = p      + ix;

      // Right border of the image
      if ( x+ix >= width )
      {
        if ( borderMethod == BORDER_ZERO )            //Zero-padding
          data[ posLocal ] = 0;
        else if ( borderMethod == BORDER_REPLICATE )  //Copy nearest pixel
          data[ posLocal ] = data_in[ y*width+width-1 ];
      }
      else
        data[ posLocal ] = data_in[ pos ];            //Normal copy of the pixel
    }
  }

  //Synchronize - wait until the whole subarray is in the shared memory
  __syncthreads();



  float   result = 0.0f;
  float   ksum   = 0.0f;

  // Note that we do not need to check wether we are at an image border
  // because we already padded the surrounding data in the shared memory
  // (qLocal = moving position in the sub-window)
  for ( int dx=-radius; dx<=radius; dx++ )
  {
    int     qLocal   = (int)threadIdx.y * bWidthPadded +
                       (int)threadIdx.x + radius + dx;

    float   exponent = -(dx*dx) / (2.0f*sigma*sigma);
    float   kernel   = 1.0f / ( sqrtf( 2.0f * M_PI ) * sigma ) * __expf( exponent );

    result += kernel * data[qLocal];
    ksum   += kernel;
  }
  data_out[p] = result / ksum;
}





//############################################################################
__global__  void gaussFilter2D_vertical_CUDA_kernel(
                                            const float  *data_in,
                                            float        *data_out,
                                            int          width,
                                            int          height,
                                            float        sigma,
                                            int          radius,
                                            int          borderMethod )
{
  // Get the 2D-coordinate of the pixel of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;


  //##########################################################################
  //  Padding #radius pixels around the 2D CUDA block into shared memory
  //##########################################################################

  // Copy data into shared memory
  // (Size gets determined by CUDA execution arguments)
  extern __shared__ float  data[];

  // Copy the pixelwise values into the shared memory
  //   p      = Current pixel's position in the global memory
  //   pLocal = Current pixel's position in the shared memory
  int     pLocal = (threadIdx.y + radius) * blockDim.x + threadIdx.x;
  int     p      = y * width + x;



  // Directly copy all non boundary pixels
  data[ pLocal ] = data_in[ p ];

  //-------------------  Upper border of the 2D CUDA block  -------------------
  if ( threadIdx.y == 0 )
  {
    for ( int iy=radius; iy>=0; iy-- )
    {
      int   posLocal = pLocal - iy * blockDim.x;
      int   pos      = p      - iy * width;

      // Upper border of the image
      if ( y-iy < 0 )
      {
        if ( borderMethod == BORDER_ZERO )            //Zero-padding
          data[ posLocal ] = 0;
        else if ( borderMethod == BORDER_REPLICATE )  //Copy nearest pixel
          data[ posLocal ] = data_in[ 0*width+x ];
      }
      else
        data[ posLocal ] = data_in[ pos ];            //Normal copy of the pixel
    }
  }
  //----------------------- Lower border -----------------------
  if ( threadIdx.y == blockDim.y-1 )
  {
    for ( int iy=radius; iy>=0; iy-- )
    {
      int   posLocal = pLocal + iy * blockDim.x;
      int   pos      = p      + iy * width;

      // Lower border of the image
      if ( y+iy >= height )
      {
        if ( borderMethod == BORDER_ZERO )            //Zero-padding
          data[ posLocal ] = 0;
        else if ( borderMethod == BORDER_REPLICATE )  //Copy nearest pixel
          data[ posLocal ] = data_in[ (height-1)*width+x ];
      }
      else
        data[ posLocal ] = data_in[ pos ];            //Normal copy of the pixel
    }
  }


  //Synchronize - wait until the whole subarray is in the shared memory
  __syncthreads();



  float   result = 0.0f;
  float   ksum   = 0.0f;

  // Note that we do not need to check wether we are at an image border
  // because we already padded the surrounding data in the shared memory
  // (qLocal = moving position in the sub-window)
  for ( int dy=-radius; dy<=radius; dy++ )
  {
    int     qLocal   = (int)(threadIdx.y + radius + dy) * blockDim.x +
                       (int)threadIdx.x;

    float   exponent = -(dy*dy) / (2.0f*sigma*sigma);
    float   kernel   = 1.0f / ( sqrtf( 2.0f * M_PI ) * sigma ) * __expf( exponent );

    result += kernel * data[qLocal];
    ksum   += kernel;
  }
  data_out[p] = result / ksum;
}

//############################################################################
__global__  void gaussFilter2D_horizontal_CUDA_kernel_nonzero(
                                            const float  *data_in,
                                            float        *data_out,
                                            int          width,
                                            int          height,
                                            float        sigma,
                                            int          radius,
                                            int          borderMethod )
{
  // Get the 2D-coordinate of the pixel of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;


  //##########################################################################
  //  Padding #radius pixels around the 2D CUDA block into shared memory
  //##########################################################################

  // Copy data into shared memory
  // (Size gets determined by CUDA execution arguments)
  extern __shared__ float  data[];

  // Copy the pixelwise values into the shared memory
  //   p      = Current pixel's position in the global memory
  //   pLocal = Current pixel's position in the shared memory
  int     bWidthPadded = blockDim.x + 2*radius;
  int     pLocal       = threadIdx.y * bWidthPadded + threadIdx.x + radius;
  int     p            = y * width + x;



  // Directly copy all non boundary pixels
  data[ pLocal ] = data_in[ p ];

  //-------------------  Left border of the 2D CUDA block  -------------------
  if ( threadIdx.x == 0 )
  {
    for ( int ix=radius; ix>=0; ix-- )
    {
      int   posLocal = pLocal - ix;
      int   pos      = p      - ix;

      // Left border of the image
      if ( x-ix < 0 )
      {
        if ( borderMethod == BORDER_ZERO )            //Zero-padding
          data[ posLocal ] = 0;
        else if ( borderMethod == BORDER_REPLICATE )  //Copy nearest pixel
          data[ posLocal ] = data_in[ y*width+0 ];
      }
      else
        data[ posLocal ] = data_in[ pos ];            //Normal copy of the pixel
    }
  }
  //----------------------- Right border -----------------------
  if ( threadIdx.x == blockDim.x-1 )
  {
    for ( int ix=radius; ix>=0; ix-- )
    {
      int   posLocal = pLocal + ix;
      int   pos      = p      + ix;

      // Right border of the image
      if ( x+ix >= width )
      {
        if ( borderMethod == BORDER_ZERO )            //Zero-padding
          data[ posLocal ] = 0;
        else if ( borderMethod == BORDER_REPLICATE )  //Copy nearest pixel
          data[ posLocal ] = data_in[ y*width+width-1 ];
      }
      else
        data[ posLocal ] = data_in[ pos ];            //Normal copy of the pixel
    }
  }

  //Synchronize - wait until the whole subarray is in the shared memory
  __syncthreads();



  float   result = 0.0f;
  float   ksum   = 0.0f;

  // Note that we do not need to check wether we are at an image border
  // because we already padded the surrounding data in the shared memory
  // (qLocal = moving position in the sub-window)
  for ( int dx=-radius; dx<=radius; dx++ )
  {
    int     qLocal   = (int)threadIdx.y * bWidthPadded +
                       (int)threadIdx.x + radius + dx;

    float   exponent = -(dx*dx) / (2.0f*sigma*sigma);
    float   kernel   = 1.0f / ( sqrtf( 2.0f * M_PI ) * sigma ) * __expf( exponent );

    result += kernel * data[qLocal];
    ksum   += ( data[qLocal] != 0 ? kernel : 0);
  }
  data_out[p] = result / ksum;
}





//############################################################################
__global__  void gaussFilter2D_vertical_CUDA_kernel_nonzero(
                                            const float  *data_in,
                                            float        *data_out,
                                            int          width,
                                            int          height,
                                            float        sigma,
                                            int          radius,
                                            int          borderMethod )
{
  // Get the 2D-coordinate of the pixel of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // If the 2D position is outside the image, do nothing
  if ( (x >= width) || (y >= height) )
    return;


  //##########################################################################
  //  Padding #radius pixels around the 2D CUDA block into shared memory
  //##########################################################################

  // Copy data into shared memory
  // (Size gets determined by CUDA execution arguments)
  extern __shared__ float  data[];

  // Copy the pixelwise values into the shared memory
  //   p      = Current pixel's position in the global memory
  //   pLocal = Current pixel's position in the shared memory
  int     pLocal = (threadIdx.y + radius) * blockDim.x + threadIdx.x;
  int     p      = y * width + x;



  // Directly copy all non boundary pixels
  data[ pLocal ] = data_in[ p ];

  //-------------------  Upper border of the 2D CUDA block  -------------------
  if ( threadIdx.y == 0 )
  {
    for ( int iy=radius; iy>=0; iy-- )
    {
      int   posLocal = pLocal - iy * blockDim.x;
      int   pos      = p      - iy * width;

      // Upper border of the image
      if ( y-iy < 0 )
      {
        if ( borderMethod == BORDER_ZERO )            //Zero-padding
          data[ posLocal ] = 0;
        else if ( borderMethod == BORDER_REPLICATE )  //Copy nearest pixel
          data[ posLocal ] = data_in[ 0*width+x ];
      }
      else
        data[ posLocal ] = data_in[ pos ];            //Normal copy of the pixel
    }
  }
  //----------------------- Lower border -----------------------
  if ( threadIdx.y == blockDim.y-1 )
  {
    for ( int iy=radius; iy>=0; iy-- )
    {
      int   posLocal = pLocal + iy * blockDim.x;
      int   pos      = p      + iy * width;

      // Lower border of the image
      if ( y+iy >= height )
      {
        if ( borderMethod == BORDER_ZERO )            //Zero-padding
          data[ posLocal ] = 0;
        else if ( borderMethod == BORDER_REPLICATE )  //Copy nearest pixel
          data[ posLocal ] = data_in[ (height-1)*width+x ];
      }
      else
        data[ posLocal ] = data_in[ pos ];            //Normal copy of the pixel
    }
  }


  //Synchronize - wait until the whole subarray is in the shared memory
  __syncthreads();



  float   result = 0.0f;
  float   ksum   = 0.0f;

  // Note that we do not need to check wether we are at an image border
  // because we already padded the surrounding data in the shared memory
  // (qLocal = moving position in the sub-window)
  for ( int dy=-radius; dy<=radius; dy++ )
  {
    int     qLocal   = (int)(threadIdx.y + radius + dy) * blockDim.x +
                       (int)threadIdx.x;

    float   exponent = -(dy*dy) / (2.0f*sigma*sigma);
    float   kernel   = 1.0f / ( sqrtf( 2.0f * M_PI ) * sigma ) * __expf( exponent );

    result += kernel * data[qLocal];
    ksum   += ( data[qLocal] != 0 ? kernel : 0); //This line makes zero value pixels non-contributing to final output
  }
  data_out[p] = result / ksum;
}





//############################################################################
void  gaussFilter2D_CUDA( const float   *img_src,
                          float         *img_dst,
                          int           width,
                          int           height,
                          int           channels,
                          float         sigma,
                          int           radius,
                          int           borderMethod )
{
  // Reserve shared memory for the block + "radius" padding pixels at each border
  // Last term is some extra buffer (function parameters, template arguments, ..)
  int sharedMemSize = (g_CUDA_blockSize2DX + 2*radius) *
                      (g_CUDA_blockSize2DY + 2*radius) * sizeof(float) + 256;

  //DEBUG
  //printf( "gaussFilter_CUDA: sigma=%f , radius=%d\n"
  //        "sharedMemSize=%d , g_CUDA_maxSharedMemSize=%d\n",
  //        sigma, radius, sharedMemSize, g_CUDA_maxSharedMemSize);

  if ( sharedMemSize > g_CUDA_maxSharedMemSize )
  {
    throw Exception( "Error gaussFilter_CUDA(): Needed "
                     "sharedMemSize (%dKB) and only have maxSharedMemSize (%dKB)!\n",
                     sharedMemSize/1024, g_CUDA_maxSharedMemSize/1024 );
  }

  // Block = 2D array of threads
  dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );


  // Grid = 2D array of blocks
  // gridSizeX = ceil( width / nBlocksX )
  // gridSizeY = ceil( height / nBlocksX )
  int   gridSizeX = (width  + dimBlock.x-1) / dimBlock.x;
  int   gridSizeY = (height + dimBlock.y-1) / dimBlock.y;
  dim3  dimGrid( gridSizeX, gridSizeY, 1 );


  // Allocate intermediate storage
  float   *d_tmp;
   cudaMalloc((void**) &d_tmp, width*height*sizeof(float) ); CUDA_CHECK;


  // And finally apply the 2D Gaussian convolution on all image channels
  for ( int ch=0; ch<channels; ch++ )
  {
    int   offset = width * height * ch;

    gaussFilter2D_horizontal_CUDA_kernel<<< dimGrid, dimBlock, sharedMemSize >>>(
                             &img_src[offset], d_tmp,
                             width, height, sigma, radius, borderMethod );
    cudaDeviceSynchronize();

    gaussFilter2D_vertical_CUDA_kernel<<< dimGrid, dimBlock, sharedMemSize >>>(
                             d_tmp, &img_dst[offset],
                             width, height, sigma, radius, borderMethod );
    cudaDeviceSynchronize();
  }
  // CHECK_FOR_CUDA_ERRORS( "gaussFilter2D_CUDA" );

  // Cleanup intermediate storage
   cudaFree( d_tmp ); CUDA_CHECK;
}








//############################################################################
/**
* \brief  Resize the image pImgSrc by bilinear interpolation to the
*         size of image pImgDst
*
* \param  pImgSrc     The source image
* \param  pImgDst     The destination image
* \param  src_width   Width of the source image
* \param  src_height  Height of the source image
* \param  dst_width   Width of the destination image
* \param  dst_height  Height of the destination image
* \param  nChannels   Number of image channels
* \param  fUsePixelCenter  Whether to use pixel centers (+0.5)
*/
template< typename T >
__global__ void  scaleImage_CUDA_kernel( const T      *pImgSrc,
                                         T            *pImgDst,
                                         int          src_width,
                                         int          src_height,
                                         int          dst_width,
                                         int          dst_height,
                                         int          nChannels,
                                         bool         fUsePixelCenter=false,
                                         bool         isDepthImage=false )
{
  //TODO: use texture?

  // Get the 2D-coordinate of the pixel of the current thread
  const int   x = blockIdx.x * blockDim.x + threadIdx.x;
  const int   y = blockIdx.y * blockDim.y + threadIdx.y;


  // Do nothing for the pixels which are outside the image
  if ( ( x >= dst_width) || ( y >= dst_height) )
    return;

  // Get linear address of current pixel in the destination image
  int       iPosDst = y * dst_width + x;

  // Compute scaling factor and its inverse
  float    scaleX = (float)dst_width  / (float)src_width;
  float    scaleY = (float)dst_height / (float)src_height;


  float   pixelCenterOffset = 0.0f;
  if ( fUsePixelCenter )
    pixelCenterOffset = 0.5f;


  // Get src coordinate (u,v) = (x,y) and get integer pixel adress there
  float    u = ( x + pixelCenterOffset ) / scaleX - pixelCenterOffset;
  float    v = ( y + pixelCenterOffset ) / scaleY - pixelCenterOffset;
  int      iu = (int)u;
  int      iv = (int)v;
  int      iPosSrc = iv * src_width + iu;

  // At the right-border and lower-border of the image, where no
  // interpolation is possible, just copy the integer pixel
  if ( ( iu == src_width-1 ) || ( iv == src_height-1 ) )
  {
    for ( int ch=0; ch<nChannels; ch++ )
    {
      pImgDst[ ch*dst_width*dst_height + iPosDst ] =
      pImgSrc[ ch*src_width*src_height + iPosSrc ];
    }
  }
  else
  {
    for ( int ch=0; ch<nChannels; ch++ )
    {

      // Pointer to integer pixel p1 in the src image
      const T       *ptr = &(pImgSrc[ ch*src_width*src_height + iPosSrc ]);

      //Interpolating between the pixels:
      //.. p1 p2 ..
      //.. p3 p4 ..
      //.. .. .. ..
      T     p1 = *(ptr);
      T     p2 = *(ptr + 1);
      T     p3 = *(ptr + src_width);
      T     p4 = *(ptr + src_width + 1);

      float   du = u - iu;
      float   dv = v - iv;

      float   du_inv = 1.0f - du;

      float validPixels = 4.0f;
      if (isDepthImage) {
          validPixels = (p1 > 0) + (p2 > 0) + (p3 > 0) + (p4 > 0);
      }
      pImgDst[ ch*dst_width*dst_height + iPosDst ] = validPixels / 4.0f
                                                     * ( (1.0f - dv) * ( p1 * du_inv + p2 * du ) +
                                                                 dv  * ( p3 * du_inv + p4 * du ) );
    }//for all channels
  }//bilinear interpolation

}





//############################################################################
void  imresize_CUDA( const float   *pImgSrc,
                     float         *pImgDst,
                     int           src_width,
                     int           src_height,
                     int           dst_width,
                     int           dst_height,
                     int           channels ,
                     bool          isDepthImage)
{
  //DEBUG
  //printf( "imgSrc: %dx%dx%d -> imgDst: %dx%dx%d\n",
  //        pImgSrc->width, pImgSrc->height, pImgSrc->channels,
  //        pImgDst->width, pImgDst->height, pImgDst->channels );


  // If the images are having exactly the same size, do no interpolation and
  // just copy the image
  if ( ( src_width == dst_width )  && ( src_height == dst_height ) )
  {
    cudaMemcpy( pImgDst,
                pImgSrc,
                src_width * src_height * channels * sizeof(float),
                cudaMemcpyDeviceToDevice ); CUDA_CHECK;
    return;
  }



  // Block = 2D array of threads
  dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );


  // Grid = 2D array of blocks
  // gridSizeX = ceil( width / nBlocksX )
  // gridSizeY = ceil( height / nBlocksX )
  int   gridSizeX = (dst_width  + dimBlock.x-1) / dimBlock.x;
  int   gridSizeY = (dst_height + dimBlock.y-1) / dimBlock.y;
  dim3  dimGrid( gridSizeX, gridSizeY, 1 );



  bool    fUsePixCenter = true;


  // Image Downscaling? => Apply Gaussian blur beforehand. Only if it is NOT a DEPTH image
  if ( ( dst_width  < src_width ) && ( dst_height < src_height ) && (!isDepthImage) )
  {
    float   *I_gauss = NULL;
    cudaMalloc( (void**) &I_gauss,
                src_width*src_height*channels*sizeof(float) ); CUDA_CHECK;

    float     scaleFactorX = (float)dst_width  / (float)src_width;
    float     scaleFactorY = (float)dst_height / (float)src_height;
    float     scaleFactor  = min<float>( scaleFactorX, scaleFactorY );

    // Choice of Gaussian parameters based on:

    // 1.) "A Convex Approach for Variational Super-Resolution", Unger
    // => Similar to Matlab imresize
    float   sigma  = 0.5f * sqrt( (1/scaleFactor)*(1/scaleFactor) - 1.0f );

    // 2.) "Convex Approaches for High Performance Video Processing", Werlberger, p.58
    // => Results are way to pixeled / coarse
    //float   sigma  = 0.3f * sqrt( scaleFactor );

    // 3.) "Efficient Minimal Surface Regularization", Graber 2015
    // => Bad results, filter too big?
    //float   sigma  = (1/scaleFactor) / 3.0f;

    int     radius = (int)std::round( 3.0f * sigma );



    //DEBUG
    //printf( "scaleFactor=%f -> sigma=%f, kernel-radius=%d\n",
    //        scaleFactor, sigma, radius );

    // Run Gauss filtering
    gaussFilter2D_CUDA( pImgSrc, I_gauss, src_width, src_height, channels,
                        sigma, radius, BORDER_REPLICATE );
    cudaDeviceSynchronize();


    //DEBUG
    //Image<float> *pImg_f = copyImageDeviceToHost<float,float>( I_gauss );
    //saveImage( pImg_f->convertToUCharImage(), "dbg_imresizeGauss_gpu.ppm" );


    // Run bilinear image scaling
    scaleImage_CUDA_kernel<float><<< dimGrid, dimBlock >>>(
                               I_gauss, pImgDst,
                               src_width, src_height,
                               dst_width, dst_height,
                               channels, fUsePixCenter );

  cudaFree(I_gauss); CUDA_CHECK;
  }//if Gauss filter
  else
  {
    // Run bilinear image scaling
    scaleImage_CUDA_kernel<float><<< dimGrid, dimBlock >>>(
                                 pImgSrc, pImgDst,
                                 src_width, src_height,
                                 dst_width, dst_height,
                                 channels, fUsePixCenter , isDepthImage );
  }//if no Gauss filter

  cudaDeviceSynchronize();
}

/**
 * CUDA function. Takes in an image and returns its derivatives along both axes. Derivatives are centered in the inside and respectively sided on the edges.
 * @param  input        input image 1D array with size w*h
 * @param  dX           output image derivatives in horizontal direction as a 1D array
 * @param  dY           output image derivatives in vertical direction as a 1D array
 * @param  w            image width
 * @param  h            image height
 */
__global__ void compute_image_derivatives_CUDA (const float *input, float *dX, float *dY, int w, int h)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int ind = x + y * w;

    if (x<w && y<h) {
        // do centered derivatives where possible
        // else do left or right-sided derivatives (on the edges)
        dX[ind] = ( input[ min(x+1,w-1) + w*y  ] - input[ max(x-1, 0) + w*y ] )*0.5f;
        dY[ind] = ( input[ x + w*min(y+1, h-1) ] - input[ x + w*max(y-1, 0) ] )*0.5f;
    }
}

/**
 * Calls compute_image_derivatives_CUDA
 * @param  *pImgSrc     Input image array of length w*h
 * @param  *pImgDX      Output image array of horizontal derivatives
 * @param  *pImgDY      Output image array of vertical derivatives
 * @param  width
 * @param  height
 */
void  image_derivatives_CUDA( const float   *pImgSrc,
                              float         *pImgDX,
                              float         *pImgDY,
                              int           width,
                              int           height)
{
    // Block = 2D array of threads
    dim3  dimBlock( g_CUDA_blockSize2DX, g_CUDA_blockSize2DY, 1 );


    // Grid = 2D array of blocks
    // gridSizeX = ceil( width / nBlocksX )
    // gridSizeY = ceil( height / nBlocksX )
    int   gridSizeX = (width  + dimBlock.x-1) / dimBlock.x;
    int   gridSizeY = (height + dimBlock.y-1) / dimBlock.y;
    dim3  dimGrid( gridSizeX, gridSizeY, 1 );

    compute_image_derivatives_CUDA <<<dimGrid, dimBlock>>> (pImgSrc, pImgDX, pImgDY, width, height);
    cudaDeviceSynchronize();
}

#include "helpers.hpp"
#include "downsample.cuh"
#include "cuda_timer.cuh"
//#include "numeric_jacobian.cuh"
#include "analytic_jacobian.cuh"

enum SolvingMethod { GAUSS_NEWTON, LEVENBERG_MARQUARDT, GRADIENT_DESCENT };
enum DerivativeMethod { ANALYTIC, NUMERIC };

class Tracker {
private:
  struct PyramidLevel { float *gray, *depth; };

public:
  /**
   * Tracker constructor
   * @param grayFirstFrame  openCV Mat gray image of floats (not uchar)
   * @param depthFirstFrame openCV Mat depth image of floats
   * @param K               Eigen 3x3 matrix with camera projection parameters
   * @param solvingMethod   Method for solving the linear system for delta ksi
   * @param weightType      Weight shape for the residuals // TODO? guillermo: not sure how far this is implemented
   * @param minLevel        Lowest level in the pyramid to be used for alignment. Default is 0 (full resolution).
   * @param maxLevel        Highest level number of the pyramid above the full resolution image
   * @param iterationsCount Maximum number of iterations per pyramid level
   */
  Tracker(
    cv::Mat grayFirstFrame,
    cv::Mat depthFirstFrame,
    Eigen::Matrix3f K,
    SolvingMethod solvingMethod = GAUSS_NEWTON,
    ResidualWeight weightType = NONE,
    int minLevel = 0,
    int maxLevel = 4,
    int iterationsCount = 20,
    bool useCUBLAS = true
  ) :
    solvingMethod(solvingMethod),
    minLevel(minLevel),
    maxLevel(maxLevel),
    iterationsCount(iterationsCount),
    weightType(weightType),
    totalComputationTime(0),
    frameComputationTime(0),
    stepCount(0),
    lastFrameXi(Vector6f::Zero()),
    xi(Vector6f::Zero()),
    useCUBLAS(useCUBLAS)
  {
    if (useCUBLAS) cublasCreate(&handle);

    width = grayFirstFrame.cols;
    height = grayFirstFrame.rows;
    int w = width;  //TODO oskar: No point calling it something else.. Might aswell just use width and height. Language economy?
    int h = height;

    // Create Buffers
    cudaMalloc(&d_J, w*h*6*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_r, w*h*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_b, 6*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_A, 6*6*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_error, sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_visualResidual, w*h*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_n, sizeof(int)); CUDA_CHECK;

    // make pyramid vector large enough to hold all levels
    d_cur.resize(maxLevel+1);
    d_prev.resize(maxLevel+1);
    // allocate pyramid vector levels in device memory
    for (int l = 0; l <= maxLevel; l++) {
      int lw = w / (1 << l);  // calculating bitwise the succesive powers of 2
      int lh = h / (1 << l);
      cudaMalloc(&d_cur [l].gray, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_prev[l].gray, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_cur [l].depth, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_prev[l].depth, lw*lh*sizeof(float)); CUDA_CHECK;
    }

    // Student-T weights allocation // TODO: shouldn't this be a pyramid? Or is it just allocated in excess for higher levels?
    if (weightType == TDIST) {
      cudaMalloc(&d_tdistWeightedSqSum, w*h*sizeof(float)); CUDA_CHECK;
    }

    // Intrinisc matrix (camera projection matrix)
    Ks.resize(maxLevel+1);
    Ks[0] = K;
    for (int l = 1; l <= maxLevel; l++) { Ks[l] = downsampleK(Ks[l-1]); }

    // Fill pyramid of the first frame. Already as previous frame since align fills the current frame d_cur and only swaps at the end.
    fill_pyramid(d_prev, (float*)grayFirstFrame.data, (float*)depthFirstFrame.data);
  }

  /**
   * destructor
   */
  ~Tracker() {
    if (useCUBLAS) cublasDestroy(handle);

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
    }

    if (weightType == TDIST) {
      cudaFree(d_tdistWeightedSqSum); CUDA_CHECK;
    }
  }

  /**
   * Provide a new frame to the tracker and calculate the transition from the previous
   * @param  grayCur  New (current) gray image of floats. Used as second frame.
   * @param  depthCur New (current) depth image of floats. It gets processed into a pyramid but its values are not actually used until the next call to align.
   * @return          Minimal transformation representation in twist coordinates. Optimal warp of the previous gray and depth onto the new (current) image.
   */
  Vector6f align(cv::Mat &grayCur, cv::Mat &depthCur) {
    Vector6f frameXi = lastFrameXi; // Good initial guess for twist

    int w = width; //TODO oskar: No point calling it something else.. Might aswell just use width and height
    int h = height;

    fill_pyramid(d_cur, (float*)grayCur.data, (float*)depthCur.data); // TODO: not sure about why casting is needed

    // Align images
    float tmr = (float)cv::getTickCount();
    for (int l = maxLevel; l >= minLevel; --l) {
      int lw = w / (1 << l);
      int lh = h / (1 << l);

      float lambda = 0.1;
      float errorLast = std::numeric_limits<float>::max();
      float scale = TDIST_SCALE0;

      int iterations = 0;

      CUDATimer timer("");
  	  timer.start();

      for (int itr = 0; itr < iterationsCount; ++itr) {
        int n = 0; // Height of jacobian

        { // Compute Jacobian and residual
          cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK; // TODO oskar: &n? what is this for? copying a memory address between Host and Device? Must be a better way
          calcResidualAndJacobian(
            d_J, d_r, d_n, scale, d_visualResidual,
            d_prev[l].gray, d_prev[l].depth, d_cur[l].gray,
            frameXi, Ks[l], lw, lh, weightType
          );
          cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;
        }

        // b = J' * r : 6x1
        Vector6f b;
        {
          float alpha = 1.f, beta = 0.f;
          cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 6, 1, n, &alpha, d_J, lw*lh, d_r, n, &beta, d_b, 6);
          cudaMemcpy(b.data(), d_b, 6*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        }

        // A = J' * J : 6x6
        Matrix6f A;
        if (solvingMethod == GAUSS_NEWTON || solvingMethod == LEVENBERG_MARQUARDT) {
          float alpha = 1.f, beta = 0.f;
          cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 6, 6, n, &alpha, d_J, lw*lh, d_J, lw*lh, &beta, d_A, 6);
          cudaMemcpy(A.data(), d_A, 6*6*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        }

        // Compute update
        Vector6f stepDeltaXi;
        if (solvingMethod == GRADIENT_DESCENT) {
          stepDeltaXi = -0.001 * b  * (1.0 / b.norm()); // (step size 0.001)
        } else if (solvingMethod == GAUSS_NEWTON) {
          stepDeltaXi = -(A.ldlt().solve(b)); // Solve using Cholesky LDLT decomposition
        } else if (solvingMethod == LEVENBERG_MARQUARDT) {
          Matrix6f diagMatA = Matrix6f::Identity();
          diagMatA.diagonal() = lambda * A.diagonal();
          stepDeltaXi = -(A + diagMatA).ldlt().solve(b);
        }

        // Apply update
        lastFrameXi = frameXi;
        frameXi = Sophus::SE3f::log(Sophus::SE3f::exp(stepDeltaXi) * Sophus::SE3f::exp(frameXi));

        iterations++;

        { // Compute error and possibly break early
          float alpha = 1.f, beta = 0.f;
          cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, 1, n, &alpha, d_r, n, d_r, n, &beta, d_error, 1);

          float error;
          cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
          error /= n;

          if (solvingMethod == LEVENBERG_MARQUARDT) {
            if (error >= errorLast) {
              lambda = lambda * 5.0;
              frameXi = lastFrameXi;
              if (lambda > 5.0) { break; }
            } else {
              lambda = lambda / 1.5;
            }
          } else if (solvingMethod == GAUSS_NEWTON || solvingMethod == GRADIENT_DESCENT) {
            if (error / errorLast > 0.995) { break; }
          }

          errorLast = error;
        }
      } // end iteration loop

      timer.stop();
      cout << "\033[31m" << iterations << "\033[0m iterations in pyramid level: \033[31m" << l << "\033[0m";
      timer.print();
    } // end level loop

    // Update xi
    xi = Sophus::SE3f::log(Sophus::SE3f::exp(xi) * Sophus::SE3f::exp(frameXi).inverse());

    // Timing
    frameComputationTime = ((double)cv::getTickCount() - tmr) / cv::getTickFrequency() * 1000.0;
    totalComputationTime += frameComputationTime;
  // }

    ++stepCount;

    // Swap buffers
    { std::vector<PyramidLevel> tmp = d_cur;  d_cur = d_prev; d_prev = tmp; } // TODO oskar: do we really need to swap? d_cur is completely reset the next time this function is called anyhow

    return frameXi;
  }

  double averageTime() { return totalComputationTime / stepCount; }

  double totalComputationTime;
  double frameComputationTime;
  int stepCount;
  Vector6f xi;

private:
  // host parameters
  SolvingMethod solvingMethod; // enum type of possible solving methods
  int iterationsCount;
  int maxLevel;
  int minLevel; // For speed. Used if the highest precision is not required
  int width;
  int height;
  ResidualWeight weightType; // enum type of possible residual weighting. Defined in *_jacobian.cuh
  bool useCUBLAS; // TODO: option to NOT use cuBLAS

  cublasHandle_t handle; // not used if useCUBLAS = false

  // device variables
  float *d_J; // device Jacobian array for ALL residuals
  float *d_r; // device residuals array
  float *d_b; // device linear system inhomogeneous term array
  float *d_A; // device linear system matrix array
  float *d_visualResidual; // device image residuals array
  float *d_error; // device single float value
  int *d_n; // TODO device single integer value
  std::vector<PyramidLevel> d_cur; // current vector of pointers to device pyramid level structures
  std::vector<PyramidLevel> d_prev; // previous vector of pointers to device pyramid level structures
  float *d_tdistWeightedSqSum; // Studendt-T weights for each residual. Has the size of the image. // TODO: shouldn't this be a pyramid? Or is it just allocated in excess for higher levels?

  // host variables
  std::vector<Matrix3f> Ks; // stores projection matrix and downsampled version
  Vector6f lastFrameXi; // TODO: keeps last(?) frame for some reason

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

    int lw, lh; // width and height of downsampled images
    for (int l = 1; l <= maxLevel; l++) {
      lw = width / (1 << l); // bitwise operator to divide by 2**l
      lh = height / (1 << l);
      downsampleGray(d_img[l].gray, d_img[l-1].gray, lw, lh);
      downsampleDepth(d_img[l].depth, d_img[l-1].depth, lw, lh);
    }
  };

};

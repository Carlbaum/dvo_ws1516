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
  Tracker(
    cv::Mat grayFirstFrame,
    cv::Mat depthFirstFrame,
    Eigen::Matrix3f K,
    SolvingMethod solvingMethod = GAUSS_NEWTON,
    ResidualWeight weightType = NONE,
    int minLevel = 0,
    int maxLevel = 4,
    int iterationsCount = 20
  ) :
    solvingMethod(GAUSS_NEWTON),
    minLevel(minLevel),
    maxLevel(maxLevel),
    iterationsCount(iterationsCount),
    weightType(weightType),
    totalComputationTime(0),
    frameComputationTime(0),
    count(0),
    lastFrameXi(Vector6f::Zero()),
    xi(Vector6f::Zero())
  {
    cublasCreate(&handle);

    width = grayFirstFrame.cols;
    height = grayFirstFrame.rows;
    int w = width;  //TODO oskar: No point calling it something else.. Might aswell just use width and height
    int h = height;

    // Create Buffers
    cudaMalloc(&d_J, w*h*6*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_r, w*h*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_b, 6*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_A, 6*6*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_error, sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_visualResidual, w*h*sizeof(float)); CUDA_CHECK;
    cudaMalloc(&d_n, sizeof(int)); CUDA_CHECK;

    d_cur.resize(maxLevel+1);
    d_prev.resize(maxLevel+1);
    for (int l = 0; l <= maxLevel; l++) {
      int lw = w / (1 << l);
      int lh = h / (1 << l);
      cudaMalloc(&d_cur [l].gray, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_prev[l].gray, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_cur [l].depth, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_prev[l].depth, lw*lh*sizeof(float)); CUDA_CHECK;
    }

    if (weightType == TDIST) {
      cudaMalloc(&d_tdistWeightedSqSum, w*h*sizeof(float)); CUDA_CHECK;
    }

    // Intrinisc matrix
    Ks.resize(maxLevel+1);
    Ks[0] = K;
    for (int l = 1; l <= maxLevel; l++) { Ks[l] = downsampleK(Ks[l-1]); }

    // Fill pyramid
    align(grayFirstFrame, depthFirstFrame); // TODO oskar: this function call returns a Vector6f, don't we need a variable to catch it? Maybe nicer with a initializing funciton here
  }

  ~Tracker() {
    cublasDestroy(handle);

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

  Vector6f align(cv::Mat &grayCur, cv::Mat &depthCur) {
    Vector6f frameXi = lastFrameXi; // Good initial guess

    int w = width; //TODO oskar: No point calling it something else.. Might aswell just use width and height
    int h = height;

    // Fill Pyramid
    cudaMemcpy(d_cur[0].gray, grayCur.data, w*h*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_cur[0].depth, depthCur.data, w*h*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    for (int l = 1; l <= maxLevel; l++) { // TODO oskar: Maybe it's better to calculate the downsamples in the main for loop below. So we don't waste time if we get to break early
      int lw = w / (1 << l);
      int lh = h / (1 << l);
      downsampleGray(d_cur[l].gray, d_cur[l-1].gray, lw, lh);
      downsampleDepth(d_cur[l].depth, d_cur[l-1].depth, lw, lh);
    }

    // Align images
    if (count > 1) { // TODO oskar: what is this count for? it is set to 0 when the tracker is created. So, 2 iterations where this is false? AHAAAA.. they call this function from the constructor once as an initializer, but still, why not: counter>0? Maybe its nicer/cleaner code to have a initializing function?
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
    }

    ++count;

    // Swap buffers
    { std::vector<PyramidLevel> tmp = d_cur;  d_cur = d_prev; d_prev = tmp; } // TODO oskar: do we really need to swap? d_cur is completely reset the next time this function is called anyhow

    return frameXi;
  }

  double averageTime() { return totalComputationTime / count; }

  double totalComputationTime;
  double frameComputationTime;
  int count;
  Vector6f xi;

private:
  SolvingMethod solvingMethod;
  int iterationsCount;
  int maxLevel;
  int minLevel;
  int width;
  int height;
  ResidualWeight weightType;

  cublasHandle_t handle;
  Vector6f lastFrameXi;

  float *d_J;
  float *d_r;
  float *d_b;
  float *d_A;
  float *d_visualResidual;
  float *d_error;
  int *d_n;
  std::vector<PyramidLevel> d_cur;
  std::vector<PyramidLevel> d_prev;
  std::vector<Matrix3f> Ks;
  float *d_tdistWeightedSqSum;
};

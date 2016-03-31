#include <Eigen/Dense>
#include "preprocessing.cuh"

enum SolvingMethod { GAUSS_NEWTON, LEVENBERG_MARQUARDT, GRADIENT_DESCENT };
// enum DerivativeMethod { ANALYTIC, NUMERIC };
enum ResidualWeight { NONE, HUBER, TDIST };

class Tracker {
private:
  struct PyramidLevel { float *gray, *depth, *dx, *dy; };

public:
  /**
   * Tracker constructor
   * @param grayFirstFrame  gray image array of floats (not uchar)
   * @param depthFirstFrame depth image array of floats
   * @param K               Eigen 3x3 matrix with camera projection parameters
   * @param solvingMethod   Method for solving the linear system for delta ksi
   * @param weightType      Weight shape for the residuals // TODO? guillermo: not sure how far this is implemented
   * @param minLevel        Lowest level in the pyramid to be used for alignment. Default is 0 (full resolution).
   * @param maxLevel        Highest level number of the pyramid above the full resolution image
   * @param iterationsCount Maximum number of iterations per pyramid level
   */
  Tracker(
    float* grayFirstFrame,
    float* depthFirstFrame,
    int w,
    int h,
    Eigen::Matrix3f K,
    int minLevel = 0,
    int maxLevel = 4,
    int iterationsCount = 20,
    SolvingMethod solvingMethod = GAUSS_NEWTON,
    ResidualWeight weightType = NONE
    // bool useCUBLAS = true,
  ) :
    w(w),
    h(h),
    solvingMethod(solvingMethod),
    minLevel(minLevel),
    maxLevel(maxLevel),
    iterationsCount(iterationsCount),
    weightType(weightType),
    totalComputationTime(0),
    frameComputationTime(0),
    stepCount(0),
    lastFrameXi(Vector6f::Zero()),
    xi(Vector6f::Zero())
    // useCUBLAS(useCUBLAS)
  {
    // if (useCUBLAS) cublasCreate(&handle);

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
      cudaMalloc(&d_cur [l].dx, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_prev[l].dx, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_cur [l].dy, lw*lh*sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_prev[l].dy, lw*lh*sizeof(float)); CUDA_CHECK;
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
    fill_pyramid(d_prev, grayFirstFrame, depthFirstFrame);

    // // Debug
    // for (int l = 0; l < maxLevel; l++) {
    //   int lw = w / (1 << l); // bitwise operator to divide by 2**l
    //   int lh = h / (1 << l);
    //   cv::Mat mTest(lh, lw, CV_32FC1);
    //   float *prev = new float[lw*lh];
    //   cudaMemcpy(prev, d_prev[l].gray, lw*lh*sizeof(float), cudaMemcpyDeviceToHost);
    //   convert_layered_to_mat(mTest, prev);
    //   showImage("Pyramid", mTest, 100, 100); cv::waitKey(0);
    //   cvDestroyAllWindows();
    // }
  }

  /**
   * destructor
   */
  ~Tracker() {
    // if (useCUBLAS) cublasDestroy(handle);

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
      cudaFree(d_cur [l].dx); CUDA_CHECK;
      cudaFree(d_prev[l].dx); CUDA_CHECK;
      cudaFree(d_cur [l].dy); CUDA_CHECK;
      cudaFree(d_prev[l].dy); CUDA_CHECK;
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
  Vector6f align(float *grayCur, float *depthCur) {
    return Vector6f::Zero();

    ++stepCount;
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
  int w; // width of the first frame (and all frames)
  int h; // height of the first frame (and all frames)
  ResidualWeight weightType; // enum type of possible residual weighting. Defined in *_jacobian.cuh
  bool useCUBLAS; // TODO: option to NOT use cuBLAS

  // cublasHandle_t handle; // not used if useCUBLAS = false

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
    cudaMemcpy(d_img[0].gray, grayImg, w*h*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_img[0].depth, depthImg, w*h*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

    int lw, lh; // width and height of downsampled images
    for (int l = 1; l <= maxLevel; l++) {
      lw = w / (1 << l); // bitwise operator to divide by 2**l
      lh = h / (1 << l);
      imresize_CUDA(d_img[l-1].gray, d_img[l].gray, 2*lw, 2*lh, lw, lh, 1);
      imresize_CUDA(d_img[l-1].depth, d_img[l].depth, 2*lw, 2*lh, lw, lh, 1);
    }

    for (int l = 0; l <= maxLevel; l++) {
      lw = w / (1 << l); // bitwise operator to divide by 2**l
      lh = h / (1 << l);
      // compute derivatives!!
    }
  };

};

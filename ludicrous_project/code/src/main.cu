/**
 * \file
 * \brief   This file is mostly a loop calling the tracker class, which is the one
 *          really implementing the paper.
 * \author  Oskar Carlbaum, Guillermo Gonzalez de Garibay, Georg Kuschk
 * \date    April 2016
 *
 * Program implementing the paper Robust Odometry Estimation for RGB-D Cameras
 * by Christian Kerl, Juergen Sturm, and Daniel Cremers.
 *
 * Only Gaussian and Student-T weights are implemented in this code.
 *
 * Parts of the code included in this project was contributed by Georg Kuschk
 * and the tutors of previous semesters. Parts of the code and program structure
 * have between borrowed from other student groups doing the same implementation
 * in previous semesters.
 *
 */

#include "helper.h"
#include <iostream>
#include "tum_benchmark.hpp"
#include "dataset.hpp"

//using namespace std;

// TODO: is this the proper way of using global variables inside the tracker class?
// global variables
const int MAX_LEVELS = 7;
// CUDA related
int devID;
cudaDeviceProp props;
int g_CUDA_maxSharedMemSize;
const int g_CUDA_blockSize2DX = 16;
const int g_CUDA_blockSize2DY = 16;
const int BORDER_ZERO = 1;
const int BORDER_REPLICATE = 2;

// tracker uses these global variables, so it has to be included after them
__constant__ float const_K_pyr[9*MAX_LEVELS];     // Allocates constant memory in excess for K and K downscaled. Stored column-wise and matrix after matrix
__constant__ float const_RK_inv[9];     // Allocates space for the concatenation of a rotation and an intrinsic matrix. Stored column-wise
__constant__ float const_translation[3];     // Allocates space for a translation vector
texture <float, 2, cudaReadModeElementType> texRef_grayImg;
texture <float, 2, cudaReadModeElementType> texRef_gray_dx;
texture <float, 2, cudaReadModeElementType> texRef_gray_dy;

#include "tracker.hpp"

int main(int argc, char *argv[]) {
        //_______________________________________________________
        //_______________________________________________________
        //________ SET ENVIRONMENT
        //_______________________________________________________
        //_______________________________________________________

        std::cout << std::endl
                  << "*******************************************************\n"
                  << "*********DENSE VISUAL ODOMETRY PROGRAM STARTED*********\n"
                  << "*******************************************************\n" << std::endl;

#ifdef ENABLE_CUBLAS
        std::cout << "Using cuBLAS" << std::endl;
#endif

        // ---------- PARAMETERS ----------
        // these can be give through command line:

        // Path to data set
        // this program will use all the images described in the txt files
        // e.g. "-path ../data/mypath_to_dataset"
        std::string path = "../data/freiburg1_xyz_first_10";
        getParam("path", path, argc, argv);
        std::cout << "Path to dataset: " << path << std::endl;

        // gives the number of levels of the pyramids,
        // This number cannot be arbitrary and will be checked later.
        // e.g. "-numberOfLevels 4"
        int numberOfLevels = 5;
        getParam("numberOfLevels", numberOfLevels, argc, argv);
        numberOfLevels = std::max(1, numberOfLevels);
        numberOfLevels = std::min(MAX_LEVELS, numberOfLevels); // 1/512 size reduction is in some cases already too large

        // set to true to use Student-T weights.
        // e.g. "-tDistWeights 1" for true
        bool tDistWeights = false;
        getParam("tDistWeights", tDistWeights, argc, argv);
        std::cout << "tDistWeights: " << tDistWeights << std::endl;

        // ------- END OF PARAMETERS -------


        // Dataset instantiation
        Dataset dataset(path);
        std::vector<double> timestamps;
        Eigen::Matrix3f K = dataset.K;

        // These poses will eventually be printed as output
        std::vector<Eigen::Matrix4f> poses;

        // Load images for frame 0, for initialization purposes
        cv::Mat mGray = loadIntensity(dataset.frames[0].colorPath);
        cv::Mat mDepth = loadDepth(dataset.frames[0].depthPath);

        // get image dimensions
        int w = mGray.cols;
        int h = mGray.rows;

        // Determine number of downscaling levels based on the size of input images
        // Compute the number of scale levels, based on the image size, the
        // pyramid scale factor and the minimum image size at the coarsest level
        const int MIN_IMAGE_SIZE = 32;
        int m_downscaleFactor = 2;
        int m_nLevels = 1;
        int w_tmp = w;
        int h_tmp = h;
        while( w_tmp/m_downscaleFactor >= MIN_IMAGE_SIZE
               && h_tmp/m_downscaleFactor >= MIN_IMAGE_SIZE ) {
                w_tmp = (int) (w_tmp / m_downscaleFactor);
                h_tmp = (int) (h_tmp / m_downscaleFactor);
                m_nLevels++;
        }
        numberOfLevels = std::max(1, std::min(m_nLevels, numberOfLevels));
        std::cout << "number of levels in pyramids: " << numberOfLevels << std::endl;

        // allocate raw input intensity and depth arrays
        float *imgGray = new float[(size_t)w*h];
        float *imgDepth = new float[(size_t)w*h];

        //_______________________________________________________
        //_______________________________________________________
        //________ EXECUTION
        //_______________________________________________________
        //_______________________________________________________

        // convert opencv images to arrays
        convert_mat_to_layered(imgGray, mGray);
        convert_mat_to_layered(imgDepth, mDepth);

        // initialize the tracker
        Tracker tracker(imgGray, imgDepth, w, h, K, 0, numberOfLevels-1,tDistWeights);

        // Store pose for frame 0
        poses.push_back(Matrix4f::Identity());
        timestamps.push_back(dataset.frames[0].timestamp);

        float total_time = 0.0f;
        std::cout << "\nStarting main loop, reading images and calculating trajectory. Take a chill pill, this may take a while!\n" << std::endl;

        // main loop
        Vector6f xi_current;
        for (size_t i = 1; i < dataset.frames.size(); ++i) {
                Timer timer; timer.start();

                // Load in the images of the next frame
                mGray = loadIntensity(dataset.frames[i].colorPath);
                mDepth = loadDepth(dataset.frames[i].depthPath);

                // convert opencv images to arrays
                convert_mat_to_layered(imgGray, mGray);
                convert_mat_to_layered(imgDepth, mDepth);

                // std::cout << "Image number: " << i << std::endl;
                xi_current = tracker.align(imgGray, imgDepth);

                timer.end();  float t = 1000 * timer.get(); // elapsed time in seconds
                total_time += t;
                // std::cout << "Time of loading + doing calculations on image #" << i << ": " << t << " ms" << std::endl;
                // show input image
                // showImage("Input " + std::to_string(i), mGray, 100+20*i, 100+10*i);  // show at position (x_from_left=100,y_from_above=100)

                // Update and push absolute pose
                poses.push_back(lieExp(xi_current));
                timestamps.push_back(dataset.frames[i].timestamp);
        }

        //_______________________________________________________
        //_______________________________________________________
        //________ OUTPUT
        //_______________________________________________________
        //_______________________________________________________

        // Save poses to disk
        std::cout << std::endl  << "Total time for loading + doing calculations on "
                  << dataset.frames.size() << " images took " << total_time
                  << " ms.\nThis gives us an average of "
                  << total_time/dataset.frames.size()
                  << " ms per frame.\n" << std::endl;

        std::string options = "";
        if (tDistWeights) {
                options += "_tdist";
        } else {
                options += "_gdist";
        }
    #ifdef ENABLE_CUBLAS
        options += "_cublas";
    #else
        options += "_nocublas";
    #endif

        savePoses( path +options+ "_trajectory.txt", poses, timestamps);

        //_______________________________________________________
        //_______________________________________________________
        //________ CLOSE
        //_______________________________________________________
        //_______________________________________________________

        cv::waitKey(0);
        cvDestroyAllWindows();
        std::cout << "All done! Check out the output file: " << path << options << "_trajectory.txt for the resulting trajectory!\n" << std::endl;
        return 0;
}

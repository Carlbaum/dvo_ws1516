/**
 * Some documentations should probably go here ;)
 * Dense visual odometry somethingsomething
 */

#include "helper.h"
#include <iostream>
#include "tum_benchmark.hpp"
#include "dataset.hpp"

using namespace std;

int main(int argc, char *argv[]) {

    // ---------- PARAMETERS ----------

    // Path to data set
    std::string path = "../data/freiburg1_xyz_first_10";
    getParam("path", path, argc, argv);
    std::cout << "Path to dataset: " << path << std::endl;

    /* FROM THE EXERCISES, DON'T THINK WE NEED THIS
        // input image
        string image = "";
        bool ret = getParam("i", image, argc, argv);
        if (!ret) cerr << "ERROR: no image specified" << endl;
        if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
    */
    // ------- END OF PARAMETERS -------


    // Dataset declarations
    Dataset dataset(path);
    std::vector<double> timestamps;
    Eigen::Matrix3f K = dataset.K;

    // This will eventually be printed as output
    //std::vector<Eigen::Matrix4f> poses;

    // Load images for frame 0, for initialization purposes
    cv::Mat gray = loadIntensity(dataset.frames[0].colorPath);
    cv::Mat depth = loadDepth(dataset.frames[0].depthPath);

    // TODO: WE NEED TO INITIALIZE THE IMAGES BEFORE THE MAIN LOOP

    /* This is not really needed yet, uncomment when we need it
    // Store pose for frame 0
    poses.push_back(Matrix4f::Identity());
    timestamps.push_back(dataset.frames[0].timestamp);
    */


    std::cout << "Hello world" << std::endl;
    for (int i = 1; i < dataset.frames.size(); ++i) {
        // Load in the images of the next frame
        cv::Mat gray = loadIntensity(dataset.frames[i].colorPath);
        cv::Mat depth = loadDepth(dataset.frames[i].depthPath);

        Timer timer; timer.start();
        // TODO: THIS IS WHERE WE SHOULD CALL THE ALIGN FUNCITON
        timer.end();  float t = timer.get();  // elapsed time in seconds
        cout << "Time of image " << i  << ": " << t*1000 << " ms" << endl;
        // show input image
        //showImage("Input " + std::to_string(i), gray, 100+20*i, 100+10*i);  // show at position (x_from_left=100,y_from_above=100)
    }


    cv::waitKey(0);
    cvDestroyAllWindows();
    std::cout << "Goodbye world" << std::endl;
    return 0;
}

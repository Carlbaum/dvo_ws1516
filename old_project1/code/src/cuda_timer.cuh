#pragma once
#include <stdexcept>

using namespace std;

class CUDATimer {
public:
    CUDATimer(const string &label, const cudaStream_t stream = 0) :
        label(label),
        stream(stream),
        running(false),
        count(0),
        totalTime(0) {}

    void start() {
        if (running) { throw runtime_error("Timer is already running"); }

        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEventRecord(startEvent, stream);
        
        running = true;
        count++;
    }

    void stop() {
        if (!running) { throw runtime_error("Timer has been started"); }

        cudaEventRecord(stopEvent, stream);

        float time = 0;
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        totalTime += time;

        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);

        running = false;
    }

    void print() {
        if (running) { throw runtime_error("Timer is still running"); }
        if (count == 0) { throw runtime_error("Timer was never run"); }

        if (count > 1) {
            cout << "\033[36m" << label <<  "\033[0m" << " took on average ";
            cout << (totalTime / count) << "ms ";
            cout << "(" << count << " runs)" << endl;
        } else {
            cout << "\033[36m" << label <<  "\033[0m" << " took ";
            cout << totalTime << "ms " << endl;
        }
    }


private:
    string label;
    cudaStream_t stream;
    cudaEvent_t startEvent, stopEvent;
    bool running;
    int count;
    float totalTime;
};

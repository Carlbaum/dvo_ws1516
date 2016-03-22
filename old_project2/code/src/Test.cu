#include "Test.h"

void testPyramidLevels(int w, int h, float **d_iPyRef, float **d_iPyCrr, float **d_dPyRef, float **d_dPyCrr, float *d_res, int showLvl, bool showCur, int showType) {
	// calculate width and height of the image to show (level dependent)
	size_t n_w = w;
	size_t n_h = h;
	for (int i = 0; i<showLvl; i++) {
		n_w = (n_w+1)/2;
		n_h = (n_h+1)/2;
	}
	// initialize image output for testing
	cv::Mat pyLvlOut(n_h, n_w, CV_32FC1);
	// select image from pyramids depending on args and show it
	switch (showType) {
		case 0: {
			if (showCur) {cudaMemcpy((void*)pyLvlOut.data, d_iPyCrr[showLvl], n_w*n_h*sizeof(float), cudaMemcpyDeviceToHost);  CUDA_CHECK;}
			else {cudaMemcpy((void*)pyLvlOut.data, d_iPyRef[showLvl], n_w*n_h*sizeof(float), cudaMemcpyDeviceToHost);  CUDA_CHECK;}
			break;
		}
		case 1: {
			if (showCur) {cudaMemcpy((void*)pyLvlOut.data, d_dPyCrr[showLvl], n_w*n_h*sizeof(float), cudaMemcpyDeviceToHost);  CUDA_CHECK;}
			else {cudaMemcpy((void*)pyLvlOut.data, d_dPyRef[showLvl], n_w*n_h*sizeof(float), cudaMemcpyDeviceToHost);  CUDA_CHECK;}
			break;
		}
		case 2: {
			cudaMemcpy((void*)pyLvlOut.data, d_res, n_w*n_h*sizeof(float), cudaMemcpyDeviceToHost);  CUDA_CHECK;
			break;
		}
		default: {
			break;
		}
	}
	showImage("PyramidLvl", pyLvlOut, w+110, 100);
	cv::waitKey(0);
	// cv::imwrite("residual.png",pyLvlOut*255.f);
}

void testJacobian(float *d_J, int lvl, int w, int h) {
	
	for (int i=0; i<lvl; i++) {
		w = (w+1)/2;
		h = (h+1)/2;
	}

	float *j1 = new float[w*h];
	float *j2 = new float[w*h];
	float *j3 = new float[w*h];
	float *j4 = new float[w*h];
	float *j5 = new float[w*h];
	float *j6 = new float[w*h];
	float *j = new float[w*h*6];
	cudaMemcpy(j, d_J, w*h*6*sizeof(float), cudaMemcpyDeviceToHost);  CUDA_CHECK;
	cudaDeviceSynchronize();
	for (int i=0; i<w*h; i++) {
		j1[i] = j[i*6];
		j2[i] = j[i*6+1];
		j3[i] = j[i*6+2];
		j4[i] = j[i*6+3];
		j5[i] = j[i*6+4];
		j6[i] = j[i*6+5];
	}

	cv::Mat pyLvlOut1(h, w, CV_32FC1);
	cv::Mat pyLvlOut2(h, w, CV_32FC1);
	cv::Mat pyLvlOut3(h, w, CV_32FC1);
	cv::Mat pyLvlOut4(h, w, CV_32FC1);
	cv::Mat pyLvlOut5(h, w, CV_32FC1);
	cv::Mat pyLvlOut6(h, w, CV_32FC1);
	convert_layered_to_mat(pyLvlOut1, j1);
	convert_layered_to_mat(pyLvlOut2, j2);
	convert_layered_to_mat(pyLvlOut3, j3);
	convert_layered_to_mat(pyLvlOut4, j4);
	convert_layered_to_mat(pyLvlOut5, j5);
	convert_layered_to_mat(pyLvlOut6, j6);
	showImage("Jacobian lvl 5", pyLvlOut6, 1680-w, h+63);
	showImage("Jacobian lvl 4", pyLvlOut5, 1680-w, 24);
	showImage("Jacobian lvl 3", pyLvlOut4, 840-w/2, h+63);
	showImage("Jacobian lvl 2", pyLvlOut3, 840-w/2, 24);
	showImage("Jacobian lvl 1", pyLvlOut2, 65, h+63);
	showImage("Jacobian lvl 0", pyLvlOut1, 65, 24);
	cv::waitKey(0);
	delete[] j1;
	delete[] j2;
	delete[] j3;
	delete[] j4;
	delete[] j5;
	delete[] j6;
}
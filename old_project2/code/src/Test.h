#ifndef TEST_H
#define TEST_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "aux.h"

void testPyramidLevels(int w, int h, float **d_iPyRef, float **d_iPyCrr, float **d_dPyRef, float **d_dPyCrr, float *d_res, int showLvl, bool showCur, int showType);
void testJacobian(float *d_J, int lvl, int w, int h);

#endif
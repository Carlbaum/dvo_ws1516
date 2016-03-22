#ifndef GPUMEM_H
#define GPUMEM_H

#include "aux.h"

void allocGPUMem(float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_kPy, float **d_ikPy, float **d_dXPy, float **d_dYPy, float **d_res, float **d_J, int lvlNum, size_t n, float **d_R, float **d_t, float **d_redArr, float **d_redArr2, float **d_A, float **d_b);
void freeGPUMem(float **d_iPyRef, float **d_dPyRef, float **d_iPyCrr, float **d_dPyCrr, float **d_kPy, float **d_ikPy, float **d_dXPy, float **d_dYPy, float *d_res, float *d_J, int lvlNum, int w, int h, float *d_R, float *d_t, float *d_redArr, float *d_redArr2, float *d_A, float *d_b);

#endif
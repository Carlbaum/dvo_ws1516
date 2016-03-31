#pragma once
#include <Eigen/Dense>

Matrix3f downsampleK(const Matrix3f K) {
  Matrix3f K_d = K;
  K_d(0, 2) += 0.5; K_d(1, 2) += 0.5;
  K_d.topLeftCorner(2, 3) *= 0.5;
  K_d(0, 2) -= 0.5; K_d(1, 2) -= 0.5;
  return K_d;
}

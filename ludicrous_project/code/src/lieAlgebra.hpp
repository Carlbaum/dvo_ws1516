#pragma once

#include <Eigen/Dense>
#include "eigen_typedef.h"

using namespace Eigen;
/**
 * Convert xi twist coordinates to rotation matrix and translation vector
 * @param xi twist coordinates array of the form (w1, w2, w3, v1, v2, v3)
 * @param R  output rotation matrix
 * @param t  output translation vector
 */
void convertSE3ToT(const Vector6f &xi, Matrix3f &R, Vector3f &t) {
    float norm_w = xi.tail(3).norm();
    Matrix3f w_hat;
    w_hat <<     0, -xi(5),  xi(4),
             xi(5),      0, -xi(3),
            -xi(4),  xi(3),      0;

    // do not divide by zero if norm_w=0
    if (norm_w > 0) {
        // R = exp(w_hat), t = A*v, where v = xi.head(3)
        R = Matrix3f::Identity(3,3)
            + sinf(norm_w) / norm_w * w_hat
            + (1-cosf(norm_w)) / (norm_w * norm_w) * w_hat*w_hat;
        t = ( Matrix3f::Identity(3,3)
              + (1-cosf(norm_w)) / (norm_w * norm_w) * w_hat
              + (norm_w-sinf(norm_w)) / (norm_w * norm_w *norm_w) * (w_hat * w_hat)
            ) // A matrix
            * xi.head(3); // * v
    } else {
        // R = I, A = I
        R = Matrix3f::Identity(3,3);
        t = xi.head(3);
    }
}

/**
 * Convert rotation matrix and translation vector to xi twist coordinates
 * @param xi output twist coordinates array of the form (w1, w2, w3, v1, v2, v3)
 * @param R  rotation matrix
 * @param t  translation vector
 */
void convertTToSE3(Vector6f &xi, const Matrix3f &R, const Vector3f &t) {
    // norm_w is the Euclidean norm of w
    float norm_w  = acosf((R.trace() -1 ) * 0.5f );
    Matrix3f w_hat;

    // do not divide by zero if norm_w=0
    if (norm_w > 0) {
        // log(R) = w_hat
        w_hat = norm_w/(2*sinf(norm_w)) * (R-R.transpose());
        // v = xi.head(3) = inverse(A) * t
        xi.head(3) = ( Matrix3f::Identity(3,3)
                       + (1-cosf(norm_w))*w_hat /(norm_w*norm_w)
                       + (norm_w-sinf(norm_w)) * w_hat * w_hat / (norm_w * norm_w * norm_w)
                     ).inverse() // a^-1 matrix
                     * t; // * t
    } else {
        w_hat = Matrix3f::Zero(3,3);
        xi.head(3) = t;
    }

    xi.tail(3) << w_hat(2,1) , w_hat(0,2) , w_hat(1,0);
}

#include <Eigen/Dense>

/**
 * [convertSE3ToT description]
 * @param xi [description]
 * @param R  [description]
 * @param t  [description]
 */
void convertSE3ToT(const Vector6f &xi, Matrix3f &R, Vector3f &t) {
    // xi is of the form (w1, w2, w3, v1, v2, v3)
    float norm_w = xi.head(3).norm();
    Matrix3f w_hat;
    w_hat <<     0, -xi(2),  xi(1),
             xi(2),      0, -xi(0),
            -xi(1),  xi(0),      0;

    // do not divide by zero if norm_w=0
    if (norm_w > 0) {
        // R = exp(w_hat), t = A*v, where v = xi.tail(3)
        R = Matrix3f::Identity(3,3)
            + sinf(norm_w) / norm_w * w_hat
            + (1-cosf(norm_w)) / (norm_w * norm_w) * w_hat*w_hat;
        t = ( Matrix3f::Identity(3,3)
              + (1-cosf(norm_w)) / (norm_w * norm_w) * w_hat
              + (norm_w-sinf(norm_w)) / (norm_w * norm_w *norm_w) * (w_hat * w_hat)
            ) // A matrix
            * xi.tail(3); // * v
    } else {
        // R = I, A = I
        R = Matrix3f::Identity(3,3);
        t = xi.tail(3);
    }
}

/**
 * [convertTToSE3 description]
 * @param xi [description]
 * @param R  [description]
 * @param t  [description]
 */
void convertTToSE3(Vector6f &xi, const Matrix3f &R, const Vector3f &t) {
    // norm_w is the Euclidean norm of w
    float norm_w  = acosf((R.trace() -1 ) * 0.5f );
    Matrix3f w_hat;

    // do not divide by zero if norm_w=0
    if (norm_w > 0) {
        // log(R) = w_hat
        w_hat = norm_w/(2*sinf(norm_w)) * (R-R.transpose());
        // v = xi.tail(3) = inverse(A) * t
        xi.tail(3) = ( Matrix3f::Identity(3,3)
                       + (1-cosf(norm_w))*w_hat /(norm_w*norm_w)
                       + (norm_w-sinf(norm_w)) * w_hat * w_hat / (norm_w * norm_w * norm_w)
                     ).inverse() // a^-1 matrix
                     * t; // * t
    } else {
        w_hat = Matrix3f::Zero(3,3);
        xi.tail(3) = t;
    }

    xi.head(3) << w_hat(2,1) , w_hat(0,2) , w_hat(1,0);

}

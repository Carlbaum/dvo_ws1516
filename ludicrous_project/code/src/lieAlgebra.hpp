#include <Eigen/Dense>

/**
 * [convertSE3ToT description]
 * @param xi [description]
 * @param R  [description]
 * @param t  [description]
 */
void convertSE3ToT(const Vector6f &xi, Matrix3f &R, Vector3f &t) {
    // xi is of the form (wx, wy, wz, vx, vy, vz)
    float norm_w = xi.head(3).norm();
    Matrix3f w_hat;
    w_hat <<    0 , -xi(2),  xi(1),
             xi(2),     0 , -xi(0),
            -xi(1),  xi(0),     0 ;

    // R = exp(w_hat), t = A*v, where v = xi.tail(3)
    R = Matrix3f::Identity(3,3) + sinf(norm_w)*w_hat /norm_w + (1-cosf(norm_w)) * w_hat*w_hat / (norm_w * norm_w) ;
    t =(Matrix3f::Identity(3,3) + (1-cosf(norm_w))*w_hat /(norm_w*norm_w) + (norm_w-sinf(norm_w)) * w_hat*w_hat / (norm_w * norm_w *norm_w)) * xi.tail(3) ;
}

void convertTToSE3(Vector6f &xi, const Matrix3f &R, const Vector3f &t) {
    float norm_w  = acosf((R.trace() -1 ) * 0.5f );

    // log(R) = w_hat
    Matrix3f w_hat = norm_w/(2*sinf(norm_w)) * (R-R.transpose());

    xi.head(3) << w_hat(1,2) , w_hat(2,0) , w_hat(0,1);

    // v = xi.tail(3) = inverse(A) * t
    xi.tail(3) = ( Matrix3f::Identity(3,3)
                   + (1-cosf(norm_w))*w_hat /(norm_w*norm_w)
                   + (norm_w-sinf(norm_w)) * w_hat * w_hat / (norm_w * norm_w * norm_w)
                 ).inverse() * t;
}

#pragma once

struct KernelVector3f { float a0, a1, a2; };
struct KernelMatrix3f { float a00, a10, a20, a01, a11, a21, a02, a12, a22; };

// Function: Skalar * Vector
inline __host__ __device__ KernelVector3f mul(float x, KernelVector3f vec) {
    return (KernelVector3f) { x*vec.a0, x*vec.a1, x*vec.a2 };
}

inline __host__ __device__ KernelVector3f mul(KernelVector3f vec, float x) {
    return mul(x, vec);
}

// Function: Vector + Vector
inline __host__ __device__ KernelVector3f add(KernelVector3f vec1, KernelVector3f vec2) {
    return (KernelVector3f) { vec1.a0+vec2.a0, vec1.a1+vec2.a1, vec1.a2+vec2.a2 };
}

// Function: Skalar * Matrix
inline __host__ __device__ KernelMatrix3f mul(float x, KernelMatrix3f mat) {
    return (KernelMatrix3f) {
        x*mat.a00, x*mat.a01, x*mat.a02,
        x*mat.a10, x*mat.a11, x*mat.a12,
        x*mat.a20, x*mat.a21, x*mat.a22
    };
}

inline __host__ __device__ KernelMatrix3f mul(KernelMatrix3f mat, float x) {
    return mul(x, mat);
}

// Function: Matrix * Vector
inline __host__ __device__ KernelVector3f mul(KernelMatrix3f mat, KernelVector3f vec) {
    return (KernelVector3f) {
        mat.a00*vec.a0 + mat.a01*vec.a1 + mat.a02*vec.a2,
        mat.a10*vec.a0 + mat.a11*vec.a1 + mat.a12*vec.a2,
        mat.a20*vec.a0 + mat.a21*vec.a1 + mat.a22*vec.a2
    };
}

// Operator: Skalar * Vector
inline __host__ __device__ KernelVector3f operator*(float x, KernelVector3f vec) {
    return mul(x, vec);
}

inline __host__ __device__ KernelVector3f operator*(KernelVector3f vec, float x) {
    return mul(x, vec);
}

// Operator: Vector + Vector
inline __host__ __device__ KernelVector3f operator+(KernelVector3f vec1, KernelVector3f vec2) {
    return add(vec1, vec2);
}

// Operator: Skalar * Matrix
inline __host__ __device__ KernelMatrix3f operator*(float x, KernelMatrix3f mat) {
    return mul(x, mat);
}

inline __host__ __device__ KernelMatrix3f operator*(KernelMatrix3f mat, float x) {
    return mul(x, mat);
}

// Operator: Matrix * Vector
inline __host__ __device__ KernelVector3f operator*(KernelMatrix3f mat, KernelVector3f vec) {
    return mul(mat, vec);
}

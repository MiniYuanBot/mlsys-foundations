// Provided kernels and baselines. Read them before writing your own.
#include "l1/kernels.cuh"

namespace l1 {

// ---------------------------------------------------------------- transpose ----
// One thread per element. Reads are coalesced (neighbouring threads read
// neighbouring columns of a row), writes are strided by M floats, so each
// warp's 32 writes land in 32 different 32-byte sectors.
__global__ void transpose_naive_kernel(const float* in, float* out, int M, int N) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) out[col * M + row] = in[row * N + col];
}

void transpose_naive(const float* in, float* out, int M, int N) {
    const dim3 block(32, 8);
    const dim3 grid(ceil_div(N, 32), ceil_div(M, 8));
    transpose_naive_kernel<<<grid, block>>>(in, out, M, N);
    CUDA_CHECK_LAUNCH();
}

// -------------------------------------------------------------- naive SGEMM ----
// One thread per C element, written the "obvious" way: threadIdx.x walks
// ROWS. Consecutive threads of a warp therefore read A rows that are K floats
// apart and write C rows that are N floats apart: nothing is coalesced.
// B1 fixes this with a one-line change in the index mapping.
__global__ void sgemm_naive_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    const dim3 block(32, 32);
    const dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
    sgemm_naive_kernel<<<grid, block>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAUNCH();
}

// ------------------------------------------------------------------ cuBLAS ----
cublasHandle_t cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();
    return handle;
}

// cuBLAS is column-major. A row-major X[r,c] is the column-major X^T[c,r], so
//   row-major C = A B   <=>   column-major C^T = B^T A^T
// i.e. call cuBLAS with the operands swapped and M/N swapped. No transposes run.
void sgemm_cublas(int M, int N, int K, const float* A, const float* B, float* C) {
    const float one = 1.f, zero = 0.f;
    // Pedantic math: real FP32 FMAs, no silent TF32 down-conversion.
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle(), CUBLAS_PEDANTIC_MATH));
    CUBLAS_CHECK(cublasSgemm(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N));
}

void hgemm_cublas(int M, int N, int K, const __half* A, const __half* B, float* C) {
    const float one = 1.f, zero = 0.f;
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle(), CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasGemmEx(cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, N, A,
                              CUDA_R_16F, K, &zero, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

}  // namespace l1

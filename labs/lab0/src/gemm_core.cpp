// Provided: naive baseline and the dispatcher.
#include "mt/gemm.hpp"

#include <stdexcept>

namespace mt {

const char *kernel_name(GemmKernel kernel) {
    switch (kernel) {
    case GemmKernel::Naive:
        return "naive";
    case GemmKernel::Reordered:
        return "reordered";
    case GemmKernel::Blocked:
        return "blocked";
    case GemmKernel::Packed:
        return "packed";
    case GemmKernel::Parallel:
        return "parallel";
    case GemmKernel::Best:
        return "best";
    }
    return "unknown";
}

// The textbook loop. For each C(i,j), walk a row of A and a COLUMN of B.
// Walking a column of a row-major B jumps ldb floats per step, so almost every
// B access is a cache miss once N is large. That is what B1 fixes.
void gemm_naive(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C,
                int ldc) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = acc;
        }
}

void gemm(GemmKernel kernel, int M, int N, int K, const float *A, int lda, const float *B, int ldb,
          float *C, int ldc) {
    if (M < 0 || N < 0 || K < 0) throw std::invalid_argument("gemm: negative size");
    if (lda < K || ldb < N || ldc < N)
        throw std::invalid_argument("gemm: leading dimension too small");
    switch (kernel) {
    case GemmKernel::Naive:
        return gemm_naive(M, N, K, A, lda, B, ldb, C, ldc);
    case GemmKernel::Reordered:
        return gemm_reordered(M, N, K, A, lda, B, ldb, C, ldc);
    case GemmKernel::Blocked:
        return gemm_blocked(M, N, K, A, lda, B, ldb, C, ldc);
    case GemmKernel::Packed:
        return gemm_packed(M, N, K, A, lda, B, ldb, C, ldc);
    case GemmKernel::Parallel:
    case GemmKernel::Best:
        return gemm_parallel(M, N, K, A, lda, B, ldb, C, ldc);
    }
}

} // namespace mt

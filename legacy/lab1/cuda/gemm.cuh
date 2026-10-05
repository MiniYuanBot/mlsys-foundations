#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void check_cuda(cudaError_t code, const char* expression, int line) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA line ") + std::to_string(line) +
                             " " + expression + ": " + cudaGetErrorString(code));
  }
}
inline void check_cublas(cublasStatus_t code, const char* expression, int line) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS line ") + std::to_string(line) +
                             " " + expression + ": status " + std::to_string(code));
  }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr, __LINE__)
#define CUBLAS_CHECK(expr) check_cublas((expr), #expr, __LINE__)

// Row-major contiguous A[M,K], B[K,N], C[M,N]. C = A*B, not C += A*B.
// All launchers use the supplied stream; they do not allocate or synchronize.
void launch_naive(const float* a, const float* b, float* c,
                  int m, int n, int k, cudaStream_t stream);
void launch_student(const float* a, const float* b, float* c,
                    int m, int n, int k, cudaStream_t stream);
void launch_cublas(cublasHandle_t handle, const float* a, const float* b,
                   float* c, int m, int n, int k);

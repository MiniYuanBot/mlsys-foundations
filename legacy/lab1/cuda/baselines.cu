#include "gemm.cuh"

namespace {
__global__ void naive_gemm(const float* a, const float* b, float* c,
                           int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float sum = 0.0f;
  for (int p = 0; p < k; ++p) sum += a[row * k + p] * b[p * n + col];
  c[row * n + col] = sum;
}
}  // namespace

void launch_naive(const float* a, const float* b, float* c,
                  int m, int n, int k, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((n + 15) / 16, (m + 15) / 16);
  naive_gemm<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

void launch_cublas(cublasHandle_t handle, const float* a, const float* b,
                   float* c, int m, int n, int k) {
  // cuBLAS is column-major. Reinterpret row-major buffers as transposes:
  // C^T[N,M] = B^T[N,K] * A^T[K,M]. No physical transpose is necessary.
  // Leading dimensions therefore are N, K, N, in that order.
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                          &alpha, b, n, a, k, &beta, c, n));
}

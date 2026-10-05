// Every launcher in lab1. All matrices are ROW-MAJOR and contiguous:
//   A[M,K], B[K,N], C[M,N],  C = A @ B  (C is overwritten, not accumulated).
// Launchers run on the default stream, do not allocate, and do not synchronize.
// Unimplemented launchers throw l1::NotImplemented.
#pragma once

#include "l1/common.cuh"

namespace l1 {

// ------------------------------------------------------- Part A: basics ----
// A1: c[i] = a[i] + b[i] for i < n.
void vector_add(const float* a, const float* b, float* c, int n);
// A2: *out = sum of x[0..n). Two-pass: per-block partial sums, then one block.
//     `partial` is scratch space of at least reduce_scratch_size() floats.
int reduce_scratch_size();
void reduce_sum(const float* x, int n, float* partial, float* out);
// A3: out[N,M] = in[M,N]^T.
void transpose_naive(const float* in, float* out, int M, int N);  // provided
void transpose_tiled(const float* in, float* out, int M, int N);   // A3a
void transpose_padded(const float* in, float* out, int M, int N);  // A3b

// -------------------------------------------- Part B: SGEMM (fp32) ladder ----
void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C);       // provided
void sgemm_coalesced(int M, int N, int K, const float* A, const float* B, float* C);   // B1
void sgemm_smem(int M, int N, int K, const float* A, const float* B, float* C);        // B2
void sgemm_blocktile_1d(int M, int N, int K, const float* A, const float* B, float* C);// B3
void sgemm_blocktile_2d(int M, int N, int K, const float* A, const float* B, float* C);// B4
// Shape requirements of the tiled kernels (launchers throw std::invalid_argument
// otherwise). B1/B2 accept any shape.
constexpr int B3_BM = 64, B3_BN = 64, B3_BK = 8, B3_TM = 8;
constexpr int B4_BM = 128, B4_BN = 128, B4_BK = 8, B4_TM = 8, B4_TN = 8;

// ------------------------------- Part C: HGEMM (fp16 in, fp32 out) on Tensor Cores ----
// A, B are half; C is float; accumulation is fp32.
void hgemm_wmma(int M, int N, int K, const __half* A, const __half* B, float* C);  // C1
void hgemm_mma(int M, int N, int K, const __half* A, const __half* B, float* C);   // C2
constexpr int C1_BM = 64, C1_BN = 64, C1_BK = 16;
constexpr int C2_BM = 128, C2_BN = 128, C2_BK = 32, C2_STAGES = 3;

// ------------------------------------------------------- baselines (provided) ----
cublasHandle_t cublas_handle();  // created on first use, uses the default stream
void sgemm_cublas(int M, int N, int K, const float* A, const float* B, float* C);
void hgemm_cublas(int M, int N, int K, const __half* A, const __half* B, float* C);

}  // namespace l1

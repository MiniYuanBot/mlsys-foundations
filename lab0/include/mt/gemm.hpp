// Single-precision GEMM kernels for lab0 Part B.
//
// All kernels compute   C[M,N] = A[M,K] @ B[K,N]
// on row-major matrices with leading dimensions (lda, ldb, ldc):
//   A(i,k) = A[i*lda + k],  B(k,j) = B[k*ldb + j],  C(i,j) = C[i*ldc + j].
// C is OVERWRITTEN (its old contents, even NaN, must not leak into the result).
#pragma once

namespace mt {

enum class GemmKernel {
    Naive,      // provided: i-j-k triple loop
    Reordered,  // B1: i-k-j loop order
    Blocked,    // B2: cache blocking on top of B1
    Packed,     // B3: Goto/BLIS style packing + register micro-kernel
    Parallel,   // B4: B3 split across threads
    Best,       // alias used by mt::matmul; points at Parallel
};

const char* kernel_name(GemmKernel kernel);

void gemm(GemmKernel kernel, int M, int N, int K, const float* A, int lda, const float* B, int ldb,
          float* C, int ldc);

void gemm_naive(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_reordered(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_blocked(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_packed(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
// threads <= 0 means std::thread::hardware_concurrency().
void gemm_parallel(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                   int threads = 0);

// ---- tuning knobs (you may change them; re-run tests and benchmarks) ----
// Register tile of the micro-kernel: it keeps an MR x NR block of C in
// registers. 6 x 16 floats = 12 AVX2 registers (or 24 NEON registers).
constexpr int MR = 6;
constexpr int NR = 16;
// Cache blocks: an MC x KC block of A should fit in L2,
// a KC x NR sliver of B in L1, a KC x NC panel of B in L3.
constexpr int MC = 96;   // multiple of MR
constexpr int KC = 256;
constexpr int NC = 2048; // multiple of NR

namespace detail {
// Exposed so the tests can check each piece of B3 separately.

// Pack the mc x kc block of A starting at A (already offset to the block's
// top-left) into row panels of height MR.  Panel p holds rows p*MR..p*MR+MR-1,
// stored k-major:   Ap[p*kc*MR + k*MR + r] = A(p*MR + r, k).
// Rows past mc are zero-filled so the micro-kernel never needs a bounds check.
// Ap must hold ceil(mc/MR)*MR*kc floats.
void pack_A(int mc, int kc, const float* A, int lda, float* Ap);

// Pack the kc x nc block of B into column panels of width NR, stored k-major:
//   Bp[q*kc*NR + k*NR + c] = B(k, q*NR + c),  columns past nc zero-filled.
// Bp must hold ceil(nc/NR)*NR*kc floats.
void pack_B(int kc, int nc, const float* B, int ldb, float* Bp);

// C(0..mr-1, 0..nr-1) += Ap_panel (MR x kc) @ Bp_panel (kc x NR),
// where only the top-left mr x nr part (mr <= MR, nr <= NR) is written back.
void micro_kernel(int kc, const float* Ap, const float* Bp, float* C, int ldc, int mr, int nr);
}  // namespace detail

}  // namespace mt

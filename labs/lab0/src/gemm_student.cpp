// Part B: from a 1 GFLOP/s triple loop towards BLAS.  Fill in every TODO(B*).
//
// Correctness:  ctest --test-dir build/lab0 -L partB --output-on-failure
// Speed:        build/lab0/bench_gemm > results/lab0/gemm.csv
//
// Read lecture L0 section 3 first: each step below has a one-paragraph
// explanation and a picture there.
#include "mt/gemm.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mt {

[[maybe_unused]] static std::logic_error not_implemented(const char *what) {
    return std::logic_error(std::string("TODO ") + what + " is not implemented yet");
}

// TODO(B1) gemm_reordered: loop order i-k-j.
// for i: (zero row i of C)  for k: a = A(i,k);  for j: C(i,j) += a * B(k,j)
// Now the innermost loop walks a ROW of B and a ROW of C: unit stride, so the
// compiler can vectorise it and every cache line is fully used.
void gemm_reordered(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C,
                    int ldc) {
    (void)M;
    (void)N;
    (void)K;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    throw not_implemented("B1 gemm_reordered");
}

// TODO(B2) gemm_blocked: cache blocking.
// Zero C first. Then iterate over blocks:
//   for jc in [0,N) step NC:  for pc in [0,K) step KC:  for ic in [0,M) step MC:
//     run the i-k-j loop of B1 restricted to the block
//       rows    ic .. min(ic+MC, M)
//       k       pc .. min(pc+KC, K)
//       columns jc .. min(jc+NC, N)
//     (accumulate with +=, because each C block receives K/KC partial sums)
// Why: the KC x NC panel of B is reused for every row block of A while it is
// still in cache, instead of being streamed from DRAM once per row of C.
void gemm_blocked(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C,
                  int ldc) {
    (void)M;
    (void)N;
    (void)K;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    throw not_implemented("B2 gemm_blocked");
}

namespace detail {

// TODO(B3a) pack_A. Layout is documented in gemm.hpp. Do not forget the
// zero padding of the last panel when mc is not a multiple of MR.
void pack_A(int mc, int kc, const float *A, int lda, float *Ap) {
    (void)mc;
    (void)kc;
    (void)A;
    (void)lda;
    (void)Ap;
    throw not_implemented("B3a pack_A");
}

// TODO(B3b) pack_B. Same idea, column panels of width NR.
void pack_B(int kc, int nc, const float *B, int ldb, float *Bp) {
    (void)kc;
    (void)nc;
    (void)B;
    (void)ldb;
    (void)Bp;
    throw not_implemented("B3b pack_B");
}

// TODO(B3c) micro_kernel: the only place where FLOPs happen.
// 1. float acc[MR * NR] = {}    one flat array: row r is acc + r*NR
// 2. for k in 0..kc:  a = Ap + k*MR,  b = Bp + k*NR
//      for r in 0..MR:  ar = a[r];  row = acc + r*NR
//        #pragma omp simd
//        for c in 0..NR:  row[c] += ar * b[c];
// 3. write back: for r < mr, c < nr:  C[r*ldc + c] += acc[r*NR + c]
// Per k-step this does MR*NR FMAs while loading only MR + NR floats.
//
// Vectorisation is FRAGILE. With GCC 13, a 2-D `float acc[MR][NR]` or a
// missing `#pragma omp simd` / -fopenmp-simd gives ~3 GFLOP/s instead of
// ~130 GFLOP/s on one Alder Lake core: the compiler spills acc to the stack.
// Always check bench_gemm (or the assembly, see the README) after a change.
// Stretch B3d: rewrite this with AVX2 (_mm256_fmadd_ps) or NEON (vfmaq_f32)
// intrinsics, which is what BLIS/OpenBLAS do, and compare.
void micro_kernel(int kc, const float *Ap, const float *Bp, float *C, int ldc, int mr, int nr) {
    (void)kc;
    (void)Ap;
    (void)Bp;
    (void)C;
    (void)ldc;
    (void)mr;
    (void)nr;
    throw not_implemented("B3c micro_kernel");
}

} // namespace detail

// B3 driver (provided): the five loops of Goto & van de Geijn / BLIS.
//
//   loop 5  jc : NC-wide column panel of B and C
//   loop 4  pc : KC-deep slice           -> pack B(pc:pc+KC, jc:jc+NC) into Bp
//   loop 3  ic : MC-tall row block of A  -> pack A(ic:ic+MC, pc:pc+KC) into Ap
//   loop 2  jr : NR-wide sliver of Bp
//   loop 1  ir : MR-tall sliver of Ap    -> micro_kernel on an MR x NR tile of C
void gemm_packed(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C,
                 int ldc) {
    for (int i = 0; i < M; ++i) std::fill(C + i * ldc, C + i * ldc + N, 0.0f);
    if (K == 0) return;
    std::vector<float> Ap(static_cast<size_t>(MC) * KC);
    std::vector<float> Bp(static_cast<size_t>(NC) * KC);
    for (int jc = 0; jc < N; jc += NC) {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int kc = std::min(KC, K - pc);
            detail::pack_B(kc, nc, B + pc * ldb + jc, ldb, Bp.data());
            for (int ic = 0; ic < M; ic += MC) {
                const int mc = std::min(MC, M - ic);
                detail::pack_A(mc, kc, A + ic * lda + pc, lda, Ap.data());
                for (int jr = 0; jr < nc; jr += NR)
                    for (int ir = 0; ir < mc; ir += MR)
                        detail::micro_kernel(kc, Ap.data() + (ir / MR) * kc * MR,
                                             Bp.data() + (jr / NR) * kc * NR,
                                             C + (ic + ir) * ldc + jc + jr, ldc,
                                             std::min(MR, mc - ir), std::min(NR, nc - jr));
            }
        }
    }
}

// TODO(B4) gemm_parallel: split the rows of C into `threads` contiguous
// slabs (round each slab up to a multiple of MR so threads do not share
// micro-tiles) and run gemm_packed on each slab in its own std::thread.
// Each thread only reads A rows / writes C rows of its own slab, so no locks
// are needed. Remember to join() every thread. With M small, use fewer threads.
void gemm_parallel(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C,
                   int ldc, int threads) {
    (void)M;
    (void)N;
    (void)K;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)threads;
    throw not_implemented("B4 gemm_parallel");
}

} // namespace mt

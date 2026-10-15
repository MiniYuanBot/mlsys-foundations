// Part C: Tensor Core HGEMM (fp16 inputs, fp32 accumulate and output).
//
//   ctest --test-dir build/lab1 -L partC --output-on-failure
//   build/lab1/bench_gemm --only hgemm
//
// C1 uses the WMMA C++ API: the hardware fragment layout is hidden.
// C2 uses raw PTX (ldmatrix + mma.sync + cp.async) and a multi-stage
// software pipeline: the structure of CUTLASS's Ampere GEMM and of most
// hand-written kernels in vLLM / FlashAttention-2. Read lecture L1 section 4.
#include "l1/kernels.cuh"
#include "l1/ptx.cuh"

#include <mma.h>

namespace l1 {

static void require_multiple_c(int value, int of, const char* what) {
    if (value % of != 0)
        throw std::invalid_argument(std::string(what) + " = " + std::to_string(value) +
                                    " must be a multiple of " + std::to_string(of));
}

// ================================================================ C1 ====
// TODO(C1) WMMA GEMM. A block of 4 warps (128 threads) computes a 64 x 64
// tile of C; warp w computes the 32 x 32 sub-tile at
//   warp_m = (w / 2) * 32,  warp_n = (w % 2) * 32
// as 2 x 2 fragments of 16 x 16. For k0 = 0, 16, 32, ...:
//   load a_frag[i] from A + (row of fragment i) * K + k0         (ldm = K)
//   load b_frag[j] from B + k0 * N + (col of fragment j)         (ldm = N)
//   acc[i][j] = a_frag[i] * b_frag[j] + acc[i][j]                (mma_sync)
// then store each acc[i][j] to C with ldm = N, mem_row_major.
// Fragments are loaded straight from global memory (no shared memory): simple,
// and already several times faster than any FP32 kernel in Part B.
// Requires M % 64 == N % 64 == K % 16 == 0.
using namespace nvcuda;

__global__ void __launch_bounds__(128)
    hgemm_wmma_kernel(int M, int N, int K, const __half* A, const __half* B, float* C) {
    const int warp = threadIdx.x / 32;
    const int row0 = blockIdx.y * C1_BM + (warp / 2) * 32;
    const int col0 = blockIdx.x * C1_BN + (warp % 2) * 32;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = 0; i < 2; ++i)
            wmma::load_matrix_sync(a_frag[i], A + (row0 + i * 16) * K + k0, K);
        for (int j = 0; j < 2; ++j)
            wmma::load_matrix_sync(b_frag[j], B + k0 * N + col0 + j * 16, N);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(C + (row0 + i * 16) * N + col0 + j * 16, acc[i][j], N,
                                    wmma::mem_row_major);
}

void hgemm_wmma(int M, int N, int K, const __half* A, const __half* B, float* C) {
    constexpr bool implemented = true;
    if (!implemented) throw NotImplemented("TODO C1 hgemm_wmma");
    require_multiple_c(M, C1_BM, "M");
    require_multiple_c(N, C1_BN, "N");
    require_multiple_c(K, C1_BK, "K");
    const dim3 grid(N / C1_BN, M / C1_BM);
    hgemm_wmma_kernel<<<grid, 128>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAUNCH();
}

// ================================================================ C2 ====
// mma.sync + ldmatrix + cp.async, 3-stage pipeline.
//
// Block tile 128 x 128 x 32, 8 warps (256 threads) arranged 2 (m) x 4 (n);
// each warp owns a 64 x 32 tile of C = 4 x 4 mma tiles of 16 x 8, i.e. 64
// fp32 accumulators per thread (acc[4][4][4]).
//
// Shared memory per stage (rows are PADDED to avoid ldmatrix bank conflicts,
// see lecture L1 section 4.4):
//   As[BM][BK + 8]   128 x 40 halves   A tile, row-major (m, k)
//   Bs[BK][BN + 8]    32 x 136 halves  B tile, row-major (k, n)
// 3 stages need 56.8 KB, more than the 48 KB static limit, so the buffer is
// dynamic shared memory (the launcher raises the per-kernel limit).
constexpr int C2_THREADS = 256;
constexpr int C2_APAD = C2_BK + 8;          // 40
constexpr int C2_BPAD = C2_BN + 8;          // 136
constexpr int C2_A_STAGE = C2_BM * C2_APAD; // halves per stage
constexpr int C2_B_STAGE = C2_BK * C2_BPAD;
constexpr size_t C2_SMEM_BYTES = size_t(C2_STAGES) * (C2_A_STAGE + C2_B_STAGE) * sizeof(__half);

// TODO(C2a) load one K-tile (k0) of A and B into stage buffers As / Bs with
// cp_async_16. Each 16-byte chunk is 8 halves.
//   A tile 128 x 32 = 512 chunks; chunk c (c = tid, tid + 256):
//       row = c / 4, col = (c % 4) * 8
//       smem &As[row * C2_APAD + col]  <-  gmem &A[(m0 + row) * K + k0 + col]
//   B tile 32 x 128 = 512 chunks; chunk c:
//       row = c / 16, col = (c % 16) * 8
//       smem &Bs[row * C2_BPAD + col]  <-  gmem &B[(k0 + row) * N + n0 + col]
__device__ __forceinline__ void c2_load_tile(__half* As, __half* Bs, const __half* A,
                                             const __half* B, int N, int K, int m0, int n0,
                                             int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int c = threadIdx.x + i * C2_THREADS;
        const int row = c / 4, col = (c % 4) * 8;
        ptx::cp_async_16(&As[row * C2_APAD + col], &A[(m0 + row) * K + k0 + col]);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int c = threadIdx.x + i * C2_THREADS;
        const int row = c / 16, col = (c % 16) * 8;
        ptx::cp_async_16(&Bs[row * C2_BPAD + col], &B[(k0 + row) * N + n0 + col]);
    }
}

// TODO(C2b) one BK = 32 slice of math on one stage: two k16 steps. For kk = 0, 16:
//   A fragments, i = 0..3 (the warp's four 16-row slices):
//       ldmatrix_x4(a[i], &As[(wm + i*16 + lane % 16) * C2_APAD + kk + (lane / 16) * 8])
//   B fragments, j = 0, 2 (one ldmatrix.trans covers two 8-wide n tiles):
//       ldmatrix_x4_trans(r, &Bs[(kk + lane % 16) * C2_BPAD + wn + j*8 + (lane / 16) * 8])
//       b[j]   = {r[0], r[1]},   b[j+1] = {r[2], r[3]}
//   then acc[i][j] += a[i] * b[j] for all 4 x 4 (i, j) with mma_16816.
// wm / wn are the warp's row / column offsets inside the block tile.
// Lecture L1 section 4.3 derives these addresses from the fragment layout.
__device__ __forceinline__ void c2_compute_stage(const __half* As, const __half* Bs,
                                                 float (&acc)[4][4][4], int wm, int wn) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int kk = 0; kk < C2_BK; kk += 16) {
        uint32_t a[4][4], b[4][2];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            ptx::ldmatrix_x4(a[i], &As[(wm + i * 16 + lane % 16) * C2_APAD + kk + (lane / 16) * 8]);
#pragma unroll
        for (int j = 0; j < 4; j += 2) {
            uint32_t r[4];
            ptx::ldmatrix_x4_trans(r,
                                   &Bs[(kk + lane % 16) * C2_BPAD + wn + j * 8 + (lane / 16) * 8]);
            b[j][0] = r[0];
            b[j][1] = r[1];
            b[j + 1][0] = r[2];
            b[j + 1][1] = r[3];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) ptx::mma_16816(acc[i][j], a[i], b[j]);
    }
}

// TODO(C2c) the multi-stage pipeline. With S = C2_STAGES and KT = K / BK tiles:
//
//   prologue: for s in 0 .. S-2:  c2_load_tile(stage s, tile s);  cp_async_commit();
//   for kt in 0 .. KT-1:
//       cp_async_wait<S - 2>();      // tile kt has landed (only S-2 newer groups may be in flight)
//       __syncthreads();             // ...for every thread; and everyone is done with tile kt-1
//       t = kt + S - 1;
//       if (t < KT) c2_load_tile(stage t % S, tile t);
//       cp_async_commit();           // commit even when nothing was issued: keeps the group count in step
//       c2_compute_stage(stage kt % S, ...);
//
// While warps compute on tile kt, tiles kt+1 .. kt+S-1 are being copied in:
// memory latency is hidden behind math. Draw the timeline (lecture 4.5) first.
//
// Epilogue (provided below): fragment acc[i][j] holds C rows
//   m0 + wm + i*16 + g  (d[0], d[1])  and  m0 + wm + i*16 + g + 8  (d[2], d[3])
// at columns n0 + wn + j*8 + 2t, +1, with g = lane / 4, t = lane % 4.
__global__ void __launch_bounds__(C2_THREADS)
    hgemm_mma_kernel(int M, int N, int K, const __half* A, const __half* B, float* C) {
    extern __shared__ __align__(16) unsigned char c2_smem[];
    __half* As = reinterpret_cast<__half*>(c2_smem);
    __half* Bs = As + C2_STAGES * C2_A_STAGE;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = (warp / 4) * 64; // 2 warps along m
    const int wn = (warp % 4) * 32; // 4 warps along n
    const int m0 = blockIdx.y * C2_BM, n0 = blockIdx.x * C2_BN;
    float acc[4][4][4] = {};
    const int KT = K / C2_BK;
#pragma unroll
    for (int s = 0; s < C2_STAGES - 1; ++s) {
        if (s < KT)
            c2_load_tile(As + s * C2_A_STAGE, Bs + s * C2_B_STAGE, A, B, N, K, m0, n0, s * C2_BK);
        ptx::cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        ptx::cp_async_wait<C2_STAGES - 2>();
        __syncthreads();
        const int t = kt + C2_STAGES - 1;
        if (t < KT) {
            const int s = t % C2_STAGES;
            c2_load_tile(As + s * C2_A_STAGE, Bs + s * C2_B_STAGE, A, B, N, K, m0, n0, t * C2_BK);
        }
        ptx::cp_async_commit();
        const int s = kt % C2_STAGES;
        c2_compute_stage(As + s * C2_A_STAGE, Bs + s * C2_B_STAGE, acc, wm, wn);
    }
    // Epilogue (provided).
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = m0 + wm + i * 16 + g, col = n0 + wn + j * 8 + 2 * t;
            *reinterpret_cast<float2*>(&C[row * N + col]) = make_float2(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<float2*>(&C[(row + 8) * N + col]) =
                make_float2(acc[i][j][2], acc[i][j][3]);
        }
    (void)M;
}

void hgemm_mma(int M, int N, int K, const __half* A, const __half* B, float* C) {
    constexpr bool implemented = true;
    if (!implemented) throw NotImplemented("TODO C2 hgemm_mma");
    require_multiple_c(M, C2_BM, "M");
    require_multiple_c(N, C2_BN, "N");
    require_multiple_c(K, C2_BK, "K");
    static bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(hgemm_mma_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(C2_SMEM_BYTES)));
        return true;
    }();
    (void)configured;
    const dim3 grid(N / C2_BN, M / C2_BM);
    hgemm_mma_kernel<<<grid, C2_THREADS, C2_SMEM_BYTES>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAUNCH();
}

} // namespace l1

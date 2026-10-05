// Part C1: FlashAttention-2 forward on Tensor Cores (mma.sync m16n8k16),
// head dim 64, fp16 in / fp16 out, fp32 softmax state. Fill in TODO(C1*).
//
//   cmake --build build/lab2 -j && python -m pytest lab2/tests/test_c_cuda.py -v
//
// This is the inner loop of FlashAttention-2 written by hand: everything you
// built in lab1 Part C (ldmatrix, mma.sync, fragment layouts) plus the
// online softmax of Part B, done on REGISTERS. Read lecture L2 section 5.
//
// Work split: one block = 4 warps = 64 query rows of one (batch*head);
// warp w owns query rows w*16 .. w*16+15 (one m16 tile). The block walks
// over K/V in tiles of 64 keys, staged through shared memory.
//
// Fragment reminder (g = lane / 4, t = lane % 4), see lab1 ptx.cuh:
//   accumulator of one m16n8 tile: d[0..1] = (row g,   cols 2t, 2t+1)
//                                  d[2..3] = (row g+8, cols 2t, 2t+1)
#include "ptx.cuh"

#include <cuda_fp16.h>
#include <cstdio>
#include <string>

namespace l2 {

constexpr int D = 64;         // head dim
constexpr int BM = 64;        // query rows per block
constexpr int BN = 64;        // keys per tile
constexpr int PITCH = D + 8;  // padded smem row (halves): kills ldmatrix bank conflicts
constexpr int THREADS = 128;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2 = 0.6931471805599453f;

// Provided: copy a [64 x 64] fp16 tile (rows row0.., all 64 columns) from a
// [S x 64] row-major matrix into padded shared memory. 512 16-byte chunks,
// 4 per thread.
__device__ __forceinline__ void load_tile(__half* smem, const __half* g, int row0) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = threadIdx.x + i * THREADS;
        const int r = c / 8, col = (c % 8) * 8;
        *reinterpret_cast<uint4*>(&smem[r * PITCH + col]) =
            *reinterpret_cast<const uint4*>(&g[(row0 + r) * D + col]);
    }
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

template <bool CAUSAL>
__global__ void __launch_bounds__(THREADS)
    flash_mma_kernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
                     __half* __restrict__ O, float* __restrict__ L, int S, float qk_scale) {
    __shared__ __align__(16) __half Qs[BM * PITCH];
    __shared__ __align__(16) __half Ks[BN * PITCH];
    __shared__ __align__(16) __half Vs[BN * PITCH];

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane / 4, t = lane % 4;
    const int m_block = blockIdx.x;
    const size_t bh_off = size_t(blockIdx.y) * S * D;
    Q += bh_off; K += bh_off; V += bh_off; O += bh_off;
    L += size_t(blockIdx.y) * S;

    // Q tile -> registers, once. qa[kk] is the A fragment for d = kk*16 .. +16.
    load_tile(Qs, Q, m_block * BM);
    __syncthreads();
    uint32_t qa[D / 16][4];
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk)
        ptx::ldmatrix_x4(qa[kk], &Qs[(warp * 16 + lane % 16) * PITCH + kk * 16 + (lane / 16) * 8]);

    // Softmax state for this thread's two rows (g and g+8), in log2 units.
    float m_row[2] = {-INFINITY, -INFINITY};
    float l_row[2] = {0.f, 0.f};
    float o[D / 8][4];  // output accumulator: 8 n-tiles (8 cols each) of the 16 x 64 warp tile
#pragma unroll
    for (int j = 0; j < D / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;

    const int q_row0 = m_block * BM + warp * 16;  // first query row of this warp
    const int n_tiles = CAUSAL ? m_block + 1 : S / BN;
    for (int n_block = 0; n_block < n_tiles; ++n_block) {
        __syncthreads();  // everyone is done with the previous K/V tile
        load_tile(Ks, K, n_block * BN);
        load_tile(Vs, V, n_block * BN);
        __syncthreads();

        // ---------------------------------------------------------- C1a ----
        // TODO(C1a) S = Q K^T for this warp: 16 rows x 64 keys = 8 n-tiles,
        // accumulated over 4 k-steps of 16 (the head dim).
        // mma wants B[k][n] = K^T[d][key] = K[key][d], and K sits in smem as
        // row-major [key][d], so a NON-transposed ldmatrix of K rows already
        // gives the B fragment (b[0] = d 0..7, b[1] = d 8..15 of the k-step).
        // One ldmatrix_x4 covers TWO n-tiles (keys nt*8 .. nt*8+15):
        //   key = nt * 8 + (lane / 16) * 8 + lane % 8
        //   d   = kk * 16 + ((lane / 8) % 2) * 8
        //   -> r[0], r[1] = B fragment of n-tile nt;  r[2], r[3] = of n-tile nt + 1
        // Then mma_16816(s[nt], qa[kk], {r0, r1}) and (s[nt+1], qa[kk], {r2, r3}).
        float s[BN / 8][4];
        for (int nt = 0; nt < BN / 8; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;

        // ---------------------------------------------------------- C1b ----
        // TODO(C1b) online softmax on the registers.
        // 1. scale: s *= qk_scale (already includes log2(e)).
        // 2. CAUSAL and n_block == m_block (the diagonal tile): set -INFINITY
        //    where key > query. This thread's element s[nt][i] is at
        //      row   q_row0 + g + (i >= 2 ? 8 : 0)
        //      key   n_block * BN + nt * 8 + 2 * t + (i % 2)
        // 3. per row r in {0 (row g), 1 (row g+8)}: max over this thread's 16
        //    values, then over the 4 threads of the quad that share the row:
        //      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));  (then 2)
        //    m_new = max(m_row[r], mx); alpha = exp2f(m_row[r] - m_new)
        // 4. p = exp2f(s - m_new) in place; row sum likewise (quad-reduce with
        //    shuffles); l_row[r] = l_row[r] * alpha + sum; m_row[r] = m_new;
        //    and rescale the output accumulator: o[j][2r], o[j][2r+1] *= alpha.

        // ---------------------------------------------------------- C1c ----
        // TODO(C1c) O += P V without going through shared memory.
        // The trick of FlashAttention-2: the ACCUMULATOR layout of two adjacent
        // m16n8 tiles (keys 16*kk2 .. +16) is exactly the A-FRAGMENT layout of
        // one m16k16 tile. So for kk2 = 0..3:
        //   pa[0] = pack_half2(s[2kk2  ][0], s[2kk2  ][1])   row g,   keys 2t..
        //   pa[1] = pack_half2(s[2kk2  ][2], s[2kk2  ][3])   row g+8
        //   pa[2] = pack_half2(s[2kk2+1][0], s[2kk2+1][1])   row g,   keys 8+2t..
        //   pa[3] = pack_half2(s[2kk2+1][2], s[2kk2+1][3])   row g+8
        // V is smem row-major [key][d], i.e. B[k = key][n = d]: load it with
        // ldmatrix_x4_trans exactly as lab1 C2b loaded B, for d-tiles j, j+1:
        //   &Vs[(kk2 * 16 + lane % 16) * PITCH + j * 8 + (lane / 16) * 8]
        //   -> {r0, r1} for d-tile j, {r2, r3} for d-tile j + 1
        // and mma_16816(o[j], pa, ...) for j = 0..7.
        (void)pack_half2; (void)Vs[0]; (void)m_row[0]; (void)qk_scale;
    }

    // Epilogue (provided): normalise, write O (fp16) and lse (natural log).
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const float inv = 1.f / l_row[r];
        const int row = q_row0 + g + 8 * r;
#pragma unroll
        for (int j = 0; j < D / 8; ++j)
            *reinterpret_cast<__half2*>(&O[row * D + j * 8 + 2 * t]) =
                __floats2half2_rn(o[j][2 * r] * inv, o[j][2 * r + 1] * inv);
        if (t == 0) L[row] = (m_row[r] + log2f(l_row[r])) * LN2;
    }
}

constexpr bool C1_DONE = false;  // set to true when C1a-C1c are written

static std::string last_error;

}  // namespace l2

// C ABI for the ctypes bridge in attn/cuda.py. Returns 0 on success,
// 1 if not implemented, 2 on a CUDA or argument error (see flash_mma_error()).
extern "C" int flash_mma_forward(const void* q, const void* k, const void* v, void* o, void* lse, int BH, int S,
                                 int causal, float sm_scale) {
    using namespace l2;
    if (!C1_DONE) {
        last_error = "TODO C1 flash_mma_kernel is not implemented yet";
        return 1;
    }
    if (S % BN != 0 || BH <= 0) {
        last_error = "S must be a positive multiple of 64";
        return 2;
    }
    const dim3 grid(S / BM, BH);
    const float qk_scale = sm_scale * LOG2E;
    auto Qp = static_cast<const __half*>(q), Kp = static_cast<const __half*>(k), Vp = static_cast<const __half*>(v);
    auto Op = static_cast<__half*>(o);
    auto Lp = static_cast<float*>(lse);
    if (causal)
        flash_mma_kernel<true><<<grid, THREADS>>>(Qp, Kp, Vp, Op, Lp, S, qk_scale);
    else
        flash_mma_kernel<false><<<grid, THREADS>>>(Qp, Kp, Vp, Op, Lp, S, qk_scale);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        last_error = cudaGetErrorString(err);
        return 2;
    }
    return 0;
}

extern "C" const char* flash_mma_error() { return l2::last_error.c_str(); }

// Part B: the SGEMM ladder, after Simon Boehm's "How to Optimize a CUDA
// Matmul Kernel" (https://siboehm.com/articles/22/CUDA-MMM). Fill in TODO(B*).
//
//   ctest --test-dir build/lab1 -L partB --output-on-failure
//   build/lab1/bench_gemm --only sgemm
//
// Each step has a diagram in lecture L1 section 3. When a step is done, set
// its `implemented` flag to true so the tests and the benchmark pick it up.
#include "l1/kernels.cuh"

namespace l1 {

static void require_multiple(int value, int of, const char* what) {
    if (value % of != 0)
        throw std::invalid_argument(std::string(what) + " = " + std::to_string(value) + " must be a multiple of " +
                                    std::to_string(of));
}

// ================================================================ B1 ====
// TODO(B1) coalesced naive GEMM: copy sgemm_naive_kernel from core.cu and
// swap the roles of x and y, so threadIdx.x walks COLUMNS of C.
// Then the 32 threads of a warp read 32 consecutive floats of a B row (one
// 128-byte transaction) and all read the same A element (a broadcast).
// Launch with block (32, 32) and grid (ceil(N/32), ceil(M/32)).
__global__ void sgemm_coalesced_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

void sgemm_coalesced(int M, int N, int K, const float* A, const float* B, float* C) {
    constexpr bool implemented = true;
    if (!implemented) throw NotImplemented("TODO B1 sgemm_coalesced");
    const dim3 block(32, 32);
    const dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
    sgemm_coalesced_kernel<<<grid, block>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAUNCH();
}

// ================================================================ B2 ====
// TODO(B2) shared-memory tiling. Block = 32 x 32 threads computes a 32 x 32
// tile of C. Loop over K in steps of 32:
//   1. each thread loads ONE element of the A tile and ONE of the B tile
//      into shared memory (write 0 when the element is outside A or B);
//   2. __syncthreads();
//   3. acc += sum over k < 32 of As[ty][k] * Bs[k][tx];
//   4. __syncthreads();   (before the next iteration overwrites the tiles)
// Every thread must reach every __syncthreads(): do the bounds check on the
// loads and on the final store, never `return` early.
// Global traffic per block drops from 2*32*K*32 to 2*32*K floats: 32x reuse.
constexpr int B2_TILE = 32;

__global__ void sgemm_smem_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[B2_TILE][B2_TILE];
    __shared__ float Bs[B2_TILE][B2_TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * B2_TILE + ty;
    const int col = blockIdx.x * B2_TILE + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += B2_TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < B2_TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

void sgemm_smem(int M, int N, int K, const float* A, const float* B, float* C) {
    constexpr bool implemented = true;
    if (!implemented) throw NotImplemented("TODO B2 sgemm_smem");
    const dim3 block(B2_TILE, B2_TILE);
    const dim3 grid(ceil_div(N, B2_TILE), ceil_div(M, B2_TILE));
    sgemm_smem_kernel<<<grid, block>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAUNCH();
}

// ================================================================ B3 ====
// TODO(B3) 1-D register blocking. Block tile BM x BN = 64 x 64, BK = 8,
// each thread computes TM = 8 consecutive ROWS of one column of C, so a block
// has 64*64/8 = 512 threads (1-D, threadIdx.x only).
//
//   thread_col = threadIdx.x % BN        (0..63)
//   thread_row = threadIdx.x / BN        (0..7, covers rows thread_row*TM .. +TM)
//
// Per BK step:
//   load  As (BM x BK = 512 floats) and Bs (BK x BN = 512 floats): one float
//         per thread each. A: row = tid / BK, col = tid % BK. B: row = tid / BN, col = tid % BN.
//   sync
//   for d < BK:  b = Bs[d][thread_col];  for r < TM: acc[r] += As[thread_row*TM + r][d] * b;
//   sync
// Why faster: one shared-memory load of b is reused TM times from a register.
// Requires M % 64 == N % 64 == K % 8 == 0 (no bounds checks needed).
__global__ void __launch_bounds__(B3_BM * B3_BN / B3_TM)
    sgemm_blocktile_1d_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    constexpr int BM = B3_BM, BN = B3_BN, BK = B3_BK, TM = B3_TM;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int thread_col = threadIdx.x % BN;
    const int thread_row = threadIdx.x / BN;
    // Move the pointers to this block's tiles; afterwards index relative to them.
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int a_row = threadIdx.x / BK, a_col = threadIdx.x % BK;
    const int b_row = threadIdx.x / BN, b_col = threadIdx.x % BN;
    float acc[TM] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        As[a_row * BK + a_col] = A[a_row * K + a_col];
        Bs[b_row * BN + b_col] = B[b_row * N + b_col];
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int d = 0; d < BK; ++d) {
            const float b = Bs[d * BN + thread_col];
#pragma unroll
            for (int r = 0; r < TM; ++r) acc[r] += As[(thread_row * TM + r) * BK + d] * b;
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < TM; ++r) C[(thread_row * TM + r) * N + thread_col] = acc[r];
}

void sgemm_blocktile_1d(int M, int N, int K, const float* A, const float* B, float* C) {
    constexpr bool implemented = true;
    if (!implemented) throw NotImplemented("TODO B3 sgemm_blocktile_1d");
    require_multiple(M, B3_BM, "M");
    require_multiple(N, B3_BN, "N");
    require_multiple(K, B3_BK, "K");
    const dim3 grid(N / B3_BN, M / B3_BM);
    sgemm_blocktile_1d_kernel<<<grid, B3_BM * B3_BN / B3_TM>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAUNCH();
}

// ================================================================ B4 ====
// TODO(B4) 2-D register blocking + vectorised (float4) loads.
// Block tile 128 x 128, BK = 8, each thread computes a TM x TN = 8 x 8
// sub-tile, so 128*128/64 = 256 threads.
//
//   thread_col = threadIdx.x % (BN / TN)   (0..15)  -> columns thread_col*TN .. +TN
//   thread_row = threadIdx.x / (BN / TN)   (0..15)  -> rows    thread_row*TM .. +TM
//
// Loads (one float4 per thread per tile; 128*8/4 = 256 float4s each):
//   A tile: a_row = tid / (BK/4), a_col4 = tid % (BK/4). Load the float4
//           A[a_row][a_col4*4 .. +4] and store it TRANSPOSED into As[k][m]:
//           As[(a_col4*4 + i) * BM + a_row] = v.{x,y,z,w}[i].
//           (Transposed so the compute loop reads As along m, contiguously.)
//   B tile: b_row = tid / (BN/4), b_col4 = tid % (BN/4). Copy the float4 as is
//           to Bs[b_row * BN + b_col4*4].
// Compute (the "outer product" form):
//   for d < BK:
//     reg_m[i] = As[d * BM + thread_row*TM + i]   for i < TM
//     reg_n[j] = Bs[d * BN + thread_col*TN + j]   for j < TN
//     acc[i][j] += reg_m[i] * reg_n[j]
// Store: write acc to C with float4 stores (TN = 8 -> two float4 per row).
// Per d-step a thread does 64 FMAs for 16 shared loads: 4 FMAs per load
// (B3: 8 FMAs for 9 loads). Use reinterpret_cast<const float4*>(ptr)[0] for
// vector loads; it needs 16-byte alignment, guaranteed by the shape checks.
// Requires M % 128 == N % 128 == K % 8 == 0.
__global__ void __launch_bounds__(B4_BM * B4_BN / (B4_TM * B4_TN))
    sgemm_blocktile_2d_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    constexpr int BM = B4_BM, BN = B4_BN, BK = B4_BK, TM = B4_TM, TN = B4_TN;
    __shared__ float As[BK * BM];  // transposed: As[k * BM + m]
    __shared__ float Bs[BK * BN];
    const int thread_col = threadIdx.x % (BN / TN);
    const int thread_row = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int a_row = threadIdx.x / (BK / 4), a_col4 = threadIdx.x % (BK / 4);
    const int b_row = threadIdx.x / (BN / 4), b_col4 = threadIdx.x % (BN / 4);
    float acc[TM][TN] = {};
    float reg_m[TM], reg_n[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        const float4 a = reinterpret_cast<const float4*>(&A[a_row * K + a_col4 * 4])[0];
        As[(a_col4 * 4 + 0) * BM + a_row] = a.x;
        As[(a_col4 * 4 + 1) * BM + a_row] = a.y;
        As[(a_col4 * 4 + 2) * BM + a_row] = a.z;
        As[(a_col4 * 4 + 3) * BM + a_row] = a.w;
        reinterpret_cast<float4*>(&Bs[b_row * BN + b_col4 * 4])[0] =
            reinterpret_cast<const float4*>(&B[b_row * N + b_col4 * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int d = 0; d < BK; ++d) {
#pragma unroll
            for (int i = 0; i < TM; ++i) reg_m[i] = As[d * BM + thread_row * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) reg_n[j] = Bs[d * BN + thread_col * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += reg_m[i] * reg_n[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; j += 4)
            reinterpret_cast<float4*>(&C[(thread_row * TM + i) * N + thread_col * TN + j])[0] =
                make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
}

void sgemm_blocktile_2d(int M, int N, int K, const float* A, const float* B, float* C) {
    constexpr bool implemented = true;
    if (!implemented) throw NotImplemented("TODO B4 sgemm_blocktile_2d");
    require_multiple(M, B4_BM, "M");
    require_multiple(N, B4_BN, "N");
    require_multiple(K, B4_BK, "K");
    const dim3 grid(N / B4_BN, M / B4_BM);
    sgemm_blocktile_2d_kernel<<<grid, B4_BM * B4_BN / (B4_TM * B4_TN)>>>(M, N, K, A, B, C);
    CUDA_CHECK_LAUNCH();
}

}  // namespace l1

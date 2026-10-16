// Part A: CUDA warm-up. Fill in every TODO(A*).
//
//   ctest --test-dir build/lab1 -L partA --output-on-failure
//   build/lab1/bench_basics
//
// Read lecture L1 sections 1-2 first (threads/warps/blocks, memory coalescing,
// shared memory banks). Each TODO has a picture in the lecture.
#include "l1/kernels.cuh"

#include <algorithm>

namespace l1 {

// ================================================================ A1 ====
// TODO(A1) vector_add with a GRID-STRIDE loop.
//   i starts at blockIdx.x * blockDim.x + threadIdx.x and advances by
//   blockDim.x * gridDim.x (the total number of threads in the grid).
// A grid-stride loop works for any n with any grid size, so the launcher can
// pick a grid that just fills the GPU instead of one thread per element.
__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n) {
    (void)a;
    (void)b;
    (void)c;
    (void)n;
}

// Provided launcher: ~8 blocks per SM is enough to fill the GPU; the
// grid-stride loop covers whatever is left. Set `implemented` when done.
void vector_add(const float *a, const float *b, float *c, int n) {
    constexpr bool implemented = false;
    if (!implemented) throw NotImplemented("TODO A1 vector_add");
    if (n <= 0) return;
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int threads = 256;
    const int blocks = std::min(ceil_div(n, threads), sms * 8);
    vector_add_kernel<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK_LAUNCH();
}

// ================================================================ A2 ====
// Reduction in three levels: thread (grid-stride accumulate in a register)
// -> warp (shuffle) -> block (shared memory) -> grid (second launch).

// TODO(A2a) warp_reduce_sum: return the sum of `v` over the 32 lanes of the
// warp (every lane may return it, or only lane 0 must be right: your choice).
// Use __shfl_down_sync(0xffffffff, v, offset) with offset = 16, 8, 4, 2, 1.
// After the step with offset o, lane i holds the sum of lanes i .. i+2o-1.
__device__ float warp_reduce_sum(float v) {
    return v;
}

// TODO(A2b) reduce_kernel: one partial sum per block.
//   1. Each thread sums x[i] over a grid-stride loop into a register.
//   2. warp_reduce_sum -> lane 0 of each warp writes its result to
//      warp_sums[warp_id] in shared memory.
//   3. __syncthreads(), then warp 0 loads warp_sums (0 for lanes beyond the
//      number of warps), reduces again, and thread 0 writes partial[blockIdx.x].
// The block has REDUCE_THREADS threads (a multiple of 32, at most 1024).
constexpr int REDUCE_THREADS = 256;
constexpr int REDUCE_MAX_BLOCKS = 1024;

__global__ void reduce_kernel(const float *x, int n, float *partial) {
    __shared__ float warp_sums[32];
    (void)x;
    (void)n;
    (void)partial;
    (void)warp_sums[0];
}

int reduce_scratch_size() {
    return REDUCE_MAX_BLOCKS;
}

// Provided: pass 1 writes one partial per block, pass 2 reduces the partials
// with a single block. Two launches, no atomics, deterministic result.
// Set `implemented` once A2a and A2b are done.
void reduce_sum(const float *x, int n, float *partial, float *out) {
    constexpr bool implemented = false;
    if (!implemented) throw NotImplemented("TODO A2 reduce_sum");
    const int blocks = std::max(1, std::min(ceil_div(n, REDUCE_THREADS), REDUCE_MAX_BLOCKS));
    reduce_kernel<<<blocks, REDUCE_THREADS>>>(x, n, partial);
    CUDA_CHECK_LAUNCH();
    reduce_kernel<<<1, REDUCE_THREADS>>>(partial, blocks, out);
    CUDA_CHECK_LAUNCH();
}

// ================================================================ A3 ====
// Tiled transpose. A block of 32 x 8 threads moves one 32 x 32 tile:
//   load : tile[ty + j][tx] = in[(row0 + ty + j) * N + col0 + tx]   j = 0, 8, 16, 24
//   sync
//   store: out[(col0 + ty + j) * M + row0 + tx] = tile[tx][ty + j]
// Both the global read and the global write are now along a row (coalesced);
// the "transpose" happens inside shared memory.
// Remember the bounds checks: M and N are not multiples of 32 in the tests.
constexpr int TILE = 32, ROWS = 8;

// TODO(A3a) transpose with a [TILE][TILE] shared tile. Bank conflicts: the
// store phase reads tile[tx][...]: 32 threads read one COLUMN, all in the
// same bank -> 32-way conflict. Measure it with bench_basics.
__global__ void transpose_tiled_kernel(const float *in, float *out, int M, int N) {
    __shared__ float tile[TILE][TILE];
    (void)in;
    (void)out;
    (void)M;
    (void)N;
    (void)tile[0][0];
}

// TODO(A3b) identical, but declare tile[TILE][TILE + 1]. The extra column
// shifts each row by one bank, so a column now hits 32 different banks.
__global__ void transpose_padded_kernel(const float *in, float *out, int M, int N) {
    (void)in;
    (void)out;
    (void)M;
    (void)N;
}

static void launch_transpose(void (*kernel)(const float *, float *, int, int), const float *in,
                             float *out, int M, int N) {
    const dim3 block(TILE, ROWS);
    const dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
    kernel<<<grid, block>>>(in, out, M, N);
    CUDA_CHECK_LAUNCH();
}

// Change `implemented` to true once the kernel is written.
void transpose_tiled(const float *in, float *out, int M, int N) {
    constexpr bool implemented = false;
    if (!implemented) throw NotImplemented("TODO A3a transpose_tiled");
    launch_transpose(transpose_tiled_kernel, in, out, M, N);
}

void transpose_padded(const float *in, float *out, int M, int N) {
    constexpr bool implemented = false;
    if (!implemented) throw NotImplemented("TODO A3b transpose_padded");
    launch_transpose(transpose_padded_kernel, in, out, M, N);
}

} // namespace l1

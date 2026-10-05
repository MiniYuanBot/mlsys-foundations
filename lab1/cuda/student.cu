#include "gemm.cuh"

namespace {
constexpr int TILE = 16;

__global__ void tiled_gemm(const float* a, const float* b, float* c,
                           int m, int n, int k) {
  __shared__ float a_tile[TILE][TILE];
  __shared__ float b_tile[TILE][TILE];
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  // TODO 1: loop over ceil(K/TILE) tiles. Every thread loads one A and one B
  // element, using zero for out-of-range accesses. Barrier after loading.
  // TODO 2: accumulate TILE products, then barrier before reusing shared memory.
  // All threads must reach every barrier; do NOT return early at an edge tile.
  // TODO 3: store sum only when row < M and col < N.
  // Avoid calling a reference implementation from here.
  (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
  (void)a_tile; (void)b_tile; (void)row; (void)col; (void)sum;
}
}  // namespace

void launch_student(const float* a, const float* b, float* c,
                    int m, int n, int k, cudaStream_t stream) {
  // Set true only after implementing the three TODOs. Enabling this alone still
  // fails: the harness poisons C with NaNs and rejects unwritten elements.
  constexpr bool implemented = false;
  if (!implemented) {
    throw std::runtime_error("NOT_IMPLEMENTED: complete student.cu tiled GEMM");
  }
  const dim3 block(TILE, TILE);
  const dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  tiled_gemm<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
}

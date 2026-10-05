// Provided: move K/V of accepted tree nodes into their final cache positions.
// For each pair i: cache[dst_slot[i]] = cache[src_slot[i]] (all kv heads), done
// through a temporary so overlapping moves are safe.  Called once per layer.
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

// tmp: [n, row] where row = num_kv_heads * 128.  grid (n), block 128.
extern "C" __global__ void gather_slots(const bf16* __restrict__ cache, const int* __restrict__ slots,
                                        bf16* __restrict__ tmp, int n, int row) {
    const int i = blockIdx.x;
    if (i >= n) return;
    for (int c = threadIdx.x; c < row; c += blockDim.x) tmp[(long)i * row + c] = cache[(long)slots[i] * row + c];
}

extern "C" __global__ void scatter_slots(const bf16* __restrict__ tmp, const int* __restrict__ slots,
                                         bf16* __restrict__ cache, int n, int row) {
    const int i = blockIdx.x;
    if (i >= n) return;
    for (int c = threadIdx.x; c < row; c += blockDim.x) cache[(long)slots[i] * row + c] = tmp[(long)i * row + c];
}

// Part C1: tree attention (SpecInfer's "tree-based parallel decoding").
//
//   cargo test --release --test kernels
//
// Start from YOUR lab4 paged_attention (one warp per (token, head), lane l
// owns dims 4l..4l+3, online softmax). Two differences:
//
//   1. Which keys a token attends to. Token t is tree node t. It attends to
//        a) every PREFIX position p in 0 .. prefix_len-1, and
//        b) every tree node j in 0 .. T-1 with bit j set in tree_mask[t]
//           (its ancestors and itself), whose K/V sit at cache position
//           prefix_len + j.
//      A cache position p maps to a slot exactly as in lab4:
//        slot = block_tables[p / block_size] * block_size + p % block_size
//      There is one sequence, so block_tables is a single row.
//   2. No token_seq / token_pos arguments: the mask replaces causality.
//
// With a CHAIN tree (node j's parent is j-1) this computes exactly what
// lab4's causal kernel computes; the test checks that first.
//
// Launch (provided, lab4 src/ops.rs): 128 threads per block, one warp per (t, h).
#include <cuda_bf16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

typedef __nv_bfloat16 bf16;
constexpr int HEAD_DIM = 128;


extern "C" __global__ void tree_attention(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                          const bf16* __restrict__ v_cache, const int* __restrict__ block_tables,
                                          const unsigned long long* __restrict__ tree_mask, int prefix_len,
                                          bf16* __restrict__ out, int T, int num_heads, int num_kv_heads,
                                          int max_blocks, int block_size, float scale) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_global >= T * num_heads) return;
    const int t = warp_global / num_heads, h = warp_global % num_heads;
    const int kvh = h / (num_heads / num_kv_heads);
    (void)q; (void)k_cache; (void)v_cache; (void)block_tables; (void)tree_mask; (void)prefix_len;
    (void)out; (void)max_blocks; (void)block_size; (void)scale; (void)lane; (void)t; (void)kvh;
}

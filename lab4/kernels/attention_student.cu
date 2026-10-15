// Part B2: fast paged attention. Fill in TODO(B2).
//
//   cargo test --release --test kernels b2
//   cargo run --release --bin bench -- --attention naive|fast ...
//
// Same contract as paged_attention_naive in provided.cu (read it first),
// but ONE WARP per (query token, query head) instead of one thread:
//
//   lane l holds 4 of the 128 head dims: d = 4*l .. 4*l+3  (q in registers, fp32)
//   for each key position j = 0 .. token_pos[t]:
//       slot  = block_tables[seq][j / block_size] * block_size + j % block_size
//       s     = warp-sum of (q[d] * k[d] over my 4 dims) * scale     (5 shuffles)
//       online softmax (lab2 B2, per key instead of per block):
//         m_new = max(m, s); alpha = exp(m - m_new); p = exp(s - m_new)
//         l = l * alpha + p;  acc[0..3] = acc[0..3] * alpha + p * v[d]
//   out[d] = acc / l   for my 4 dims
//
// Every lane computes the same s, m, l (after the shuffle-sum all lanes hold
// the full dot product), so no further communication is needed.
// Loads: 4 bf16 = 8 bytes per lane -> a warp reads one 256-byte key row with
// perfectly coalesced 8-byte loads. Use a uint2 load and unpack with
// __bfloat1622float2 on the two __nv_bfloat162 halves.
//
// Launch (provided in src/ops.rs): 4 warps per block, grid = ceil(T * num_heads / 4).
#include <cuda_bf16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

typedef __nv_bfloat16 bf16;
constexpr int HEAD_DIM = 128;

extern "C" __global__ void
paged_attention(const bf16 *__restrict__ q, const bf16 *__restrict__ k_cache,
                const bf16 *__restrict__ v_cache, const int *__restrict__ block_tables,
                const int *__restrict__ token_seq, const int *__restrict__ token_pos,
                bf16 *__restrict__ out, int T, int num_heads, int num_kv_heads, int max_blocks,
                int block_size, float scale) {
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_global >= T * num_heads) return; // whole warps exit together
    const int t = warp_global / num_heads, h = warp_global % num_heads;
    const int kvh = h / (num_heads / num_kv_heads);
    (void)q;
    (void)k_cache;
    (void)v_cache;
    (void)block_tables;
    (void)token_seq;
    (void)token_pos;
    (void)out;
    (void)max_blocks;
    (void)block_size;
    (void)scale;
    (void)lane;
    (void)t;
    (void)kvh;
}

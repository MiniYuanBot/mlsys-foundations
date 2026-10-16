// Provided kernels. Compiled at runtime by NVRTC (see src/gpu.rs).
//
// KV cache layout, per layer:   cache[num_blocks][block_size][num_kv_heads][HEAD_DIM]  (bf16)
// A token's "slot" is block * block_size + offset; its K for kv head h lives at
//     cache + ((size_t)slot * num_kv_heads + h) * HEAD_DIM
#include <cuda_bf16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

constexpr int HEAD_DIM = 128;
typedef __nv_bfloat16 bf16;

// Write the K and V of this step's T tokens into their cache slots.
// k, v: [T, num_kv_heads, HEAD_DIM]. grid (T), block (num_kv_heads * HEAD_DIM / 8) threads,
// each thread copies 8 bf16 (16 bytes).
extern "C" __global__ void reshape_and_cache(const bf16 *__restrict__ k, const bf16 *__restrict__ v,
                                             bf16 *__restrict__ k_cache, bf16 *__restrict__ v_cache,
                                             const int *__restrict__ slot_mapping, int T,
                                             int num_kv_heads) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const long slot = slot_mapping[t];
    const int n = num_kv_heads * HEAD_DIM;
    for (int e = threadIdx.x * 8; e < n; e += blockDim.x * 8) {
        *reinterpret_cast<uint4 *>(&k_cache[slot * n + e]) =
            *reinterpret_cast<const uint4 *>(&k[(long)t * n + e]);
        *reinterpret_cast<uint4 *>(&v_cache[slot * n + e]) =
            *reinterpret_cast<const uint4 *>(&v[(long)t * n + e]);
    }
}

// Reference paged attention: ONE THREAD per (query token, query head) does
// the whole computation serially, with the classic 3-pass softmax. Correct,
// easy to read, and slow. Part B2 replaces it with a warp-per-query kernel.
//
//   q:   [T, num_heads, HEAD_DIM]        (after RoPE)
//   out: [T, num_heads, HEAD_DIM]
//   block_tables: [num_seqs, max_blocks] physical block ids
//   token_seq[t]: which sequence (row of block_tables) token t belongs to
//   token_pos[t]: its position; it attends to positions 0 .. token_pos[t] (causal)
extern "C" __global__ void
paged_attention_naive(const bf16 *__restrict__ q, const bf16 *__restrict__ k_cache,
                      const bf16 *__restrict__ v_cache, const int *__restrict__ block_tables,
                      const int *__restrict__ token_seq, const int *__restrict__ token_pos,
                      bf16 *__restrict__ out, int T, int num_heads, int num_kv_heads,
                      int max_blocks, int block_size, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * num_heads) return;
    const int t = idx / num_heads, h = idx % num_heads;
    const int kvh = h / (num_heads / num_kv_heads);
    const int *table = block_tables + (long)token_seq[t] * max_blocks;
    const int n_keys = token_pos[t] + 1;
    const bf16 *qv = q + ((long)t * num_heads + h) * HEAD_DIM;
    auto key = [&](int j) {
        const long slot = (long)table[j / block_size] * block_size + j % block_size;
        return (slot * num_kv_heads + kvh) * HEAD_DIM;
    };
    float m = -INFINITY;
    for (int j = 0; j < n_keys; ++j) {
        float s = 0.f;
        const long base = key(j);
        for (int d = 0; d < HEAD_DIM; ++d)
            s += __bfloat162float(qv[d]) * __bfloat162float(k_cache[base + d]);
        m = fmaxf(m, s * scale);
    }
    float l = 0.f;
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.f;
    for (int j = 0; j < n_keys; ++j) {
        float s = 0.f;
        const long base = key(j);
        for (int d = 0; d < HEAD_DIM; ++d)
            s += __bfloat162float(qv[d]) * __bfloat162float(k_cache[base + d]);
        const float p = __expf(s * scale - m);
        l += p;
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] += p * __bfloat162float(v_cache[base + d]);
    }
    bf16 *o = out + ((long)t * num_heads + h) * HEAD_DIM;
    for (int d = 0; d < HEAD_DIM; ++d) o[d] = __float2bfloat16(acc[d] / l);
}

// out[i, :] = x[rows[i], :]     grid (n), block 256
extern "C" __global__ void gather_rows(const bf16 *__restrict__ x, const int *__restrict__ rows,
                                       bf16 *__restrict__ out, int n, int dim) {
    const int i = blockIdx.x;
    if (i >= n) return;
    const long src = (long)rows[i] * dim, dst = (long)i * dim;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) out[dst + c] = x[src + c];
}

extern "C" __global__ void bf16_to_f32(const bf16 *__restrict__ x, float *__restrict__ y, long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n;
         i += (long)gridDim.x * blockDim.x)
        y[i] = __bfloat162float(x[i]);
}

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

__device__ __forceinline__ void load4(const bf16* p, float (&v)[4]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}

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
    const int d0 = lane * 4;
    float qv[4];
    load4(q + ((long)t * num_heads + h) * HEAD_DIM + d0, qv);
    for (int i = 0; i < 4; ++i) qv[i] *= scale;
    const unsigned long long mask = tree_mask[t];
    float m = -INFINITY, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
    auto visit = [&](int pos) {
        const long slot = (long)block_tables[pos / block_size] * block_size + pos % block_size;
        const long base = (slot * num_kv_heads + kvh) * HEAD_DIM + d0;
        float kv[4], vv[4];
        load4(k_cache + base, kv);
        float s = qv[0] * kv[0] + qv[1] * kv[1] + qv[2] * kv[2] + qv[3] * kv[3];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new), p = __expf(s - m_new);
        l = l * alpha + p;
        load4(v_cache + base, vv);
        for (int i = 0; i < 4; ++i) acc[i] = acc[i] * alpha + p * vv[i];
        m = m_new;
    };
    for (int p = 0; p < prefix_len; ++p) visit(p);
    for (int j = 0; j < T; ++j)
        if ((mask >> j) & 1ull) visit(prefix_len + j);
    const float inv = 1.f / l;
    __nv_bfloat162 o01 = __floats2bfloat162_rn(acc[0] * inv, acc[1] * inv);
    __nv_bfloat162 o23 = __floats2bfloat162_rn(acc[2] * inv, acc[3] * inv);
    uint2 packed;
    packed.x = *reinterpret_cast<unsigned*>(&o01);
    packed.y = *reinterpret_cast<unsigned*>(&o23);
    *reinterpret_cast<uint2*>(out + ((long)t * num_heads + h) * HEAD_DIM + d0) = packed;
}

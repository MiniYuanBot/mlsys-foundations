// Part A1: the element-wise / row-wise kernels of a Qwen3 layer. Fill in TODO(A1*).
//
//   cargo test --release --test kernels a1       (each kernel vs a CPU reference)
//
// All activations are bf16 in memory and fp32 in registers, like Hugging Face.
// Where HF rounds to bf16 in the middle of an op, we do too (noted below), so
// our numbers stay close to the reference model.
//
// Conversions: __bfloat162float(x), __float2bfloat16(f)  (round to nearest even)
#include <cuda_bf16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float to_f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_b(float x) { return __float2bfloat16(x); }

// Provided: sum over the block, every thread gets the result. `smem` holds 32 floats.
__device__ float block_sum(float v, float* smem) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = (blockDim.x + 31) / 32;
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    v = lane < nwarps ? smem[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    return v;
}

// ------------------------------------------------------------------ A1a ----
// TODO(A1a) embedding lookup: out[t, :] = table[tokens[t], :]
// grid (T), block 256; threads stride over the hidden dim.
extern "C" __global__ void embedding(const bf16* __restrict__ table, const int* __restrict__ tokens,
                                     bf16* __restrict__ out, int T, int hidden) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const long src = (long)tokens[t] * hidden, dst = (long)t * hidden;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) out[dst + c] = table[src + c];
}

// ------------------------------------------------------------------ A1b ----
// TODO(A1b) RMSNorm of each row:  y = x / sqrt(mean(x^2) + eps) * w
// grid (rows), block 128 or 256; dim is 128 (q_norm / k_norm, applied per
// head) or `hidden`. Steps, matching HF Qwen3RMSNorm:
//   ss = block_sum(sum of x^2 over this thread's columns)    (fp32)
//   r  = rsqrtf(ss / dim + eps)
//   y  = to_b( to_f(to_b(x * r)) * to_f(w) )    <- HF casts to bf16 BEFORE the weight multiply
extern "C" __global__ void rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out,
                                   int rows, int dim, float eps) {
    __shared__ float smem[32];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const bf16* xr = x + (long)r * dim;
    float ss = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const float v = to_f(xr[c]);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, smem) / dim + eps);
    for (int c = threadIdx.x; c < dim; c += blockDim.x)
        out[(long)r * dim + c] = to_b(to_f(to_b(to_f(xr[c]) * inv)) * to_f(w[c]));
}

// ------------------------------------------------------------------ A1c ----
// TODO(A1c) fused residual add + RMSNorm, IN PLACE (vLLM's fused_add_rms_norm):
//   residual = to_b(x + residual)          (the new residual stream)
//   x        = rmsnorm(residual) * w       (same rounding as A1b)
// One read of x and residual, one write of each, instead of three kernels.
// Read the row into registers/locals once; the row is at most 1024 x 2 B per block.
extern "C" __global__ void fused_add_rmsnorm(bf16* __restrict__ x, bf16* __restrict__ residual,
                                             const bf16* __restrict__ w, int rows, int dim, float eps) {
    __shared__ float smem[32];
    const int r = blockIdx.x;
    if (r >= rows) return;
    bf16* xr = x + (long)r * dim;
    bf16* rr = residual + (long)r * dim;
    float ss = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        const bf16 s = to_b(to_f(xr[c]) + to_f(rr[c]));
        rr[c] = s;
        const float v = to_f(s);
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, smem) / dim + eps);
    for (int c = threadIdx.x; c < dim; c += blockDim.x) xr[c] = to_b(to_f(to_b(to_f(rr[c]) * inv)) * to_f(w[c]));
}

// ------------------------------------------------------------------ A1d ----
// TODO(A1d) rotary position embedding, NeoX / HF "rotate_half" style, in place
// on q [T, num_heads, 128] and k [T, num_kv_heads, 128]. For position p and
// i in [0, 64):
//   inv_freq = 1 / theta^(2i / 128)       angle = p * inv_freq      (fp32)
//   c = to_f(to_b(cosf(angle))), s = to_f(to_b(sinf(angle)))     <- HF casts cos/sin to bf16
//   (x1, x2) = (x[i], x[i + 64])
//   x[i]      = to_b( to_f(to_b(x1 * c)) - to_f(to_b(x2 * s)) )
//   x[i + 64] = to_b( to_f(to_b(x2 * c)) + to_f(to_b(x1 * s)) )
// (HF does q*cos + rotate_half(q)*sin in bf16: each product rounds, then the sum.)
// grid (T), block 256: threads stride over (head, i) pairs of q, then of k.
extern "C" __global__ void rope(bf16* __restrict__ q, bf16* __restrict__ k, const int* __restrict__ positions, int T,
                                int num_heads, int num_kv_heads, float theta) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const float p = (float)positions[t];
    const int half_d = 64;
    auto rotate = [&](bf16* base, int heads) {
        for (int e = threadIdx.x; e < heads * half_d; e += blockDim.x) {
            const int h = e / half_d, i = e % half_d;
            const float inv_freq = 1.f / powf(theta, (2.f * i) / 128.f);
            const float c = to_f(to_b(cosf(p * inv_freq))), s = to_f(to_b(sinf(p * inv_freq)));
            bf16* xv = base + (long)h * 128;
            const float x1 = to_f(xv[i]), x2 = to_f(xv[i + half_d]);
            xv[i] = to_b(to_f(to_b(x1 * c)) - to_f(to_b(x2 * s)));
            xv[i + half_d] = to_b(to_f(to_b(x2 * c)) + to_f(to_b(x1 * s)));
        }
    };
    rotate(q + (long)t * num_heads * 128, num_heads);
    rotate(k + (long)t * num_kv_heads * 128, num_kv_heads);
}

// ------------------------------------------------------------------ A1e ----
// TODO(A1e) SwiGLU activation:  out = silu(gate) * up,  silu(g) = g / (1 + exp(-g))
// HF computes act_fn(gate) as a bf16 tensor, then multiplies:  to_b( to_f(to_b(silu(g))) * u ).
// Grid-stride loop over n elements.
extern "C" __global__ void silu_mul(const bf16* __restrict__ gate, const bf16* __restrict__ up, bf16* __restrict__ out,
                                    long n) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const float g = to_f(gate[i]);
        out[i] = to_b(to_f(to_b(g / (1.f + __expf(-g)))) * to_f(up[i]));
    }
}

// ------------------------------------------------------------------ A1f ----
// TODO(A1f) greedy sampling: out[r] = argmax over c of logits[r, c]
// (ties -> the smallest index, like torch.argmax). grid (rows), block 1024.
// Each thread scans a strided subset keeping (best value, best index), then
// reduce across the block. Hint: reduce with shared memory of 1024 (value,
// index) pairs, or with __shfl_xor_sync on both fields.
extern "C" __global__ void argmax_rows(const bf16* __restrict__ logits, int* __restrict__ out, int rows, int cols) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const bf16* x = logits + (long)r * cols;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = to_f(x[c]);
        if (v > bv) { bv = v; bi = c; }
    }
    auto better = [](float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); };
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarps = (blockDim.x + 31) / 32;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        bv = lane < nwarps ? sv[lane] : -INFINITY;
        bi = lane < nwarps ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[r] = bi;
    }
}

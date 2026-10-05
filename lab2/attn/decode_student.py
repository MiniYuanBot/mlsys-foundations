"""Part C2: FlashDecoding (split-KV) in Triton. Fill in every TODO(C2*).

    python -m pytest lab2/tests/test_c_decode.py -v
    python lab2/bench/bench_decode.py

Decode = ONE new query token per sequence attends to a long KV cache:
    q [B, H, D] fp16,  k/v [B, H, S, D] fp16,  out [B, H, D] fp16.
Each (batch, head) is only one row of work. With B * H = 8 rows and S = 32k,
a "one program per row" kernel launches 8 programs on a 24-SM GPU and leaves
most of it idle. FlashDecoding splits the KEYS into NSPLIT chunks, processes
them in parallel, then merges the partial results with their log-sum-exps.
Blog with animations: https://crfm.stanford.edu/2023/10/12/flashdecoding.html
"""
import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # host-side constant
LN2 = tl.constexpr(0.6931471805599453)  # used inside kernels: must be constexpr


# ============================================================== C2a ====
# TODO(C2a) split kernel. Program (bh, split) handles keys
#   [split * chunk, min((split + 1) * chunk, S)).
# A single query row cannot use tl.dot (it needs >= 16 rows), so use
# broadcasting instead:
#   s = sum(q[None, :] * k, axis=1)            # [BN] scores (fp32)
#   ... online softmax over the BN-blocks of this chunk, exactly as in B2 ...
#   acc += sum(p[:, None] * v, axis=0)         # [D]
# Write the chunk's NORMALISED partial output acc / l to
#   Opart[bh, split, :]  (fp32)  and its lse (natural log) to Lpart[bh, split].
# A chunk is never empty (the launcher guarantees chunk * (NSPLIT - 1) < S).
@triton.jit
def decode_split_kernel(Q, K, V, Opart, Lpart, qk_scale, S, chunk, NSPLIT,
                        D: tl.constexpr, BN: tl.constexpr):
    bh = tl.program_id(0)
    split = tl.program_id(1)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + bh * D + offs_d).to(tl.float32) * qk_scale
    kv_base = bh * S * D
    lo = split * chunk
    hi = tl.minimum(lo + chunk, S)
    pass


# ============================================================== C2b ====
# TODO(C2b) combine kernel, one program per bh. With partial outputs o_s and
# log-sum-exps lse_s of NSPLIT chunks:
#   lse   = log(sum_s exp(lse_s))            (compute it stably: subtract max)
#   out   = sum_s exp(lse_s - lse) * o_s
# Write out as fp16 and lse as fp32. Lecture L2 section 4 proves this merge
# is exact (it is the same algebra as the online-softmax rescale).
@triton.jit
def decode_combine_kernel(Opart, Lpart, O, L, NSPLIT, D: tl.constexpr, SPLIT_BLOCK: tl.constexpr):
    bh = tl.program_id(0)
    offs_s = tl.arange(0, SPLIT_BLOCK)
    offs_d = tl.arange(0, D)
    smask = offs_s < NSPLIT
    pass


C2_DONE = False  # set to True when both kernels are written


def default_num_splits(B, H, S, num_sms, block=64):
    """Enough programs to fill the GPU ~2x, but chunks of at least 4 blocks."""
    splits = max(1, (2 * num_sms + B * H - 1) // (B * H))
    return max(1, min(splits, triton.cdiv(S, 4 * block), 128))


def flash_decode(q, k, v, num_splits=None, sm_scale=None, block_n=64):
    """Provided launcher. q [B,H,D], k/v [B,H,S,D] fp16. Returns (out [B,H,D] fp16, lse [B,H] fp32)."""
    if not C2_DONE:
        raise NotImplementedError("TODO C2 decode kernels")
    B, H, S, D = k.shape
    assert q.shape == (B, H, D) and v.shape == k.shape and q.dtype == torch.float16
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if num_splits is None:
        num_splits = default_num_splits(B, H, S, torch.cuda.get_device_properties(q.device).multi_processor_count,
                                        block_n)
    chunk = triton.cdiv(triton.cdiv(S, num_splits), block_n) * block_n
    num_splits = triton.cdiv(S, chunk)  # no empty chunk
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    opart = torch.empty(B * H, num_splits, D, device=q.device, dtype=torch.float32)
    lpart = torch.empty(B * H, num_splits, device=q.device, dtype=torch.float32)
    decode_split_kernel[(B * H, num_splits)](q, k, v, opart, lpart, sm_scale * LOG2E, S, chunk, num_splits, D=D,
                                             BN=block_n, num_warps=4)
    out = torch.empty(B, H, D, device=q.device, dtype=torch.float16)
    lse = torch.empty(B, H, device=q.device, dtype=torch.float32)
    decode_combine_kernel[(B * H,)](opart, lpart, out, lse, num_splits, D=D,
                                    SPLIT_BLOCK=max(2, triton.next_power_of_2(num_splits)), num_warps=4)
    return out, lse

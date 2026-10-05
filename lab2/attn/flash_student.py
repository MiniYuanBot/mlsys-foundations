"""Part B: FlashAttention-2 forward in Triton. Fill in every TODO(B*).

    python -m pytest lab2/tests/test_b_flash.py -v
    python lab2/bench/bench_attention.py

Layout: q, k, v, out are [B, H, S, D] fp16, contiguous; D in {16, 32, 64, 128}.
lse is [B, H, S] fp32, the log-sum-exp of each row of scaled scores (the
backward pass and FlashDecoding both need it).

One program computes BM rows of the output for one (batch, head):
grid = (cdiv(S, BM), B * H). Read lecture L2 section 3 first.

Implementation note: use exp2 instead of exp. Scale the scores by
sm_scale * log2(e) once, then exp2(x) == exp(x * ln 2); exp2 maps to a single
hardware instruction (MUFU.EX2). Convert back at the end: lse = (m + log2 l) * ln 2.
"""
import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # host-side constant
LN2 = tl.constexpr(0.6931471805599453)  # used inside kernels: must be constexpr


# ============================================================== B1 ====
# TODO(B1) single-block attention, the warm-up. Assume S <= BN, so ONE block
# of K/V holds the whole sequence and no online rescaling is needed:
#   q = load [BM, D] rows of Q (rows >= S -> 0)
#   k, v = load [BN, D] (rows >= S -> 0)
#   s = dot(q, trans(k)) * qk_scale         (fp32 [BM, BN])
#   s = where(col < S, s, -inf)              (padding keys must get zero weight)
#   m = max(s, 1); p = exp2(s - m[:, None]); l = sum(p, 1)
#   out = dot(p.to(fp16), v) / l[:, None]; lse = (m + log2(l)) * LN2
# Store out (fp16) and lse for rows < S.
@triton.jit
def attn_single_block_kernel(Q, K, V, O, L, qk_scale, S, stride_bh, stride_s,
                             D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    base = bh * stride_bh
    pass


# ============================================================== B2/B3 ====
# TODO(B2) the real thing: loop over K/V in blocks of BN with ONLINE softmax.
# Keep per row: m_i (running max), l_i (running sum), acc [BM, D] (fp32,
# un-normalised output). For each block:
#   s     = dot(q, trans(k)) * qk_scale, masked to -inf for keys >= S
#   m_new = maximum(m_i, max(s, 1))
#   alpha = exp2(m_i - m_new)               rescale factor for everything so far
#   p     = exp2(s - m_new[:, None])
#   l_i   = l_i * alpha + sum(p, 1)
#   acc   = acc * alpha[:, None] + dot(p.to(fp16), v)
#   m_i   = m_new
# After the loop: out = acc / l_i[:, None]; lse = (m_i + log2(l_i)) * LN2.
# Initialise m_i = -inf, l_i = 0, acc = 0 (tl.full / tl.zeros, fp32).
#
# TODO(B3) CAUSAL (a constexpr flag): query i may only attend to keys j <= i.
#   (a) mask: s = where(offs_m[:, None] >= offs_n[None, :], s, -inf)
#   (b) skip: blocks entirely above the diagonal contribute nothing, so stop
#       the loop at hi = min((pid_m + 1) * BM, S). This halves the work.
# Do (a) first and check correctness, then add (b) and measure the speedup.
@triton.jit
def flash_fwd_kernel(Q, K, V, O, L, qk_scale, S, stride_bh, stride_s,
                     D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    base = bh * stride_bh
    q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :], mask=offs_m[:, None] < S, other=0.0)
    pass


B1_DONE = False  # set to True when attn_single_block_kernel is written
B2_DONE = False  # set to True when flash_fwd_kernel works without causal
B3_DONE = False  # set to True when the CAUSAL branch works


def _check(q, k, v):
    assert q.shape == k.shape == v.shape and q.ndim == 4, "q, k, v must all be [B, H, S, D]"
    assert q.dtype == torch.float16 and q.is_cuda
    assert q.shape[-1] in (16, 32, 64, 128), "head dim must be 16, 32, 64 or 128"
    return (t.contiguous() for t in (q, k, v))


def _block_sizes(D):
    # Fits comfortably in an RTX 40xx SM; tune in the lab (see README Part B).
    return (64, 64, 4) if D <= 64 else (64, 32, 4)


def attention_single_block(q, k, v, sm_scale=None):
    """Provided launcher for B1: requires S <= 64."""
    if not B1_DONE:
        raise NotImplementedError("TODO B1 attn_single_block_kernel")
    q, k, v = _check(q, k, v)
    B, H, S, D = q.shape
    BM = BN = 64
    if S > BN:
        raise ValueError(f"attention_single_block needs S <= {BN}")
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    attn_single_block_kernel[(triton.cdiv(S, BM), B * H)](q, k, v, out, lse, sm_scale * LOG2E, S, S * D, D,
                                                          D=D, BM=BM, BN=BN, num_warps=4)
    return out, lse


def flash_attention(q, k, v, causal=False, sm_scale=None, block_m=None, block_n=None, num_warps=None,
                    num_stages=None):
    """Provided launcher for B2/B3. Returns (out [B,H,S,D] fp16, lse [B,H,S] fp32)."""
    if not B2_DONE:
        raise NotImplementedError("TODO B2 flash_fwd_kernel")
    if causal and not B3_DONE:
        raise NotImplementedError("TODO B3 causal flash_fwd_kernel")
    q, k, v = _check(q, k, v)
    B, H, S, D = q.shape
    BM, BN, warps = _block_sizes(D)
    BM, BN, warps = block_m or BM, block_n or BN, num_warps or warps
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    extra = {"num_stages": num_stages} if num_stages else {}
    flash_fwd_kernel[(triton.cdiv(S, BM), B * H)](q, k, v, out, lse, sm_scale * LOG2E, S, S * D, D, D=D, BM=BM,
                                                  BN=BN, CAUSAL=causal, num_warps=warps, **extra)
    return out, lse

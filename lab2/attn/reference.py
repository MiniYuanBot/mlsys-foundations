"""Plain PyTorch references (fp32 math). Provided: the tests compare against these.

Shapes follow the usual multi-head layout:
    q, k, v : [B, H, S, D]      (prefill / training)
    q       : [B, H, D]         (decode: one new token per sequence)
"""

import math

import torch


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1)


def attention_ref(q, k, v, causal=False, sm_scale=None):
    """Returns (out [B,H,S,D] fp32, lse [B,H,S] fp32), lse = logsumexp of the scaled scores."""
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
    if causal:
        S_q, S_k = s.shape[-2:]
        mask = torch.ones(S_q, S_k, dtype=torch.bool, device=s.device).tril(S_k - S_q)
        s = s.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None])
    return torch.matmul(p, v.float()), lse


def decode_ref(q, k, v, sm_scale=None):
    """q [B,H,D], k/v [B,H,S,D] -> out [B,H,D] fp32."""
    out, _ = attention_ref(q[:, :, None, :], k, v, causal=False, sm_scale=sm_scale)
    return out[:, :, 0, :]


def attention_flops(B, H, S, D, causal):
    """Two matmuls of 2*S*S*D FLOPs each per head; causal does about half."""
    f = 4.0 * B * H * S * S * D
    return f / 2 if causal else f

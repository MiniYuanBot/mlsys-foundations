"""S02: IO-aware forward attention. This module launches GPU code after you implement it."""
import torch


def validate(q, k, v):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V must share [B,H,S,D]")
    if min(q.shape) < 1 or q.shape[-1] not in (32, 64):
        raise ValueError("positive dimensions, head dimension 32 or 64")
    if any(x.dtype != torch.float32 or not x.is_contiguous() or x.device != q.device for x in (q,k,v)):
        raise ValueError("same-device contiguous FP32 required")
    if not q.is_cuda:
        raise ValueError("CUDA required")


def attention(q, k, v, block_q=16, block_k=32):
    validate(q, k, v)
    if block_q not in (16, 32) or block_k not in (32, 64):
        raise ValueError("declared development tile choices only")
    # TODO S02-1: complete kernel.py, allocate only output [B,H,S,D] (no SxS
    # global intermediate), launch ceil_div(S,block_q) x (B*H) programs.
    # Return output; forward only, scale 1/sqrt(D), causal, dropout=0.
    raise NotImplementedError("S02-1: launch your streaming attention kernel")


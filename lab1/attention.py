"""L01: fill only the explicit attention; SDPA is the independent reference."""
import torch
import torch.nn.functional as F


def attention(q, k, v):
    """Contiguous FP32 [B,h,S,d], causal self-attention, no dropout, forward only."""
    # TODO L01-1: scaled QK^T, mask j>i, softmax over last axis, multiply by V.
    raise NotImplementedError("L01-1: implement explicit causal attention")


def reference(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)


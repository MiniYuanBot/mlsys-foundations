"""Shared test inputs."""
import torch


def qkv(B, H, S, D, dtype=torch.float16, scale=1.0):
    """Random q, k, v; `scale` > 1 makes the softmax peaky (a harder numerical case)."""
    g = lambda: (torch.randn(B, H, S, D, device="cuda") * scale).to(dtype)
    return g(), g(), g()

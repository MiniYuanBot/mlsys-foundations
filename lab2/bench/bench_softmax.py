"""Softmax bandwidth: unfused PyTorch ops vs torch.softmax vs your Triton kernels.

    python lab2/bench/bench_softmax.py > results/lab2/softmax.csv
Effective GB/s counts the minimum traffic, 2 * rows * cols * 4 bytes (read + write once).
"""
from common import time_ms, try_time

import torch

from lab2.attn.softmax_student import softmax_fused, softmax_online


def unfused(x):
    e = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def main():
    print("rows,cols,impl,ms,GBps")
    for rows, cols in [(4096, 1024), (4096, 4096), (1024, 16384), (64, 262144)]:
        x = torch.randn(rows, cols, device="cuda")
        nbytes = 2 * x.numel() * 4
        impls = {"torch_unfused": lambda: unfused(x), "torch_softmax": lambda: torch.softmax(x, -1),
                 "online_A2": lambda: softmax_online(x)}
        if cols <= 16384:
            impls["fused_A1"] = lambda: softmax_fused(x)
        for name, fn in impls.items():
            ms = try_time(fn)
            if ms is not None:
                print(f"{rows},{cols},{name},{ms:.4f},{nbytes / ms / 1e6:.1f}", flush=True)


if __name__ == "__main__":
    main()

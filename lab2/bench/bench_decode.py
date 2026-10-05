"""Decode attention (one query token, long KV cache): no split vs FlashDecoding vs SDPA.

    python lab2/bench/bench_decode.py > results/lab2/decode.csv
Effective GB/s counts reading K and V once: 2 * B * H * S * D * 2 bytes.
"""
from common import try_time

import torch
import torch.nn.functional as F

from lab2.attn.decode_student import flash_decode


def main():
    D = 64
    print("B,H,S,impl,splits,ms,GBps")
    for B, H, S in [(1, 8, 4096), (1, 8, 32768), (1, 32, 32768), (16, 32, 4096), (1, 8, 131072)]:
        q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        nbytes = 2 * k.numel() * 2
        rows = [("sdpa", None, lambda: F.scaled_dot_product_attention(q[:, :, None], k, v))]
        rows += [("flash_decode", 1, lambda: flash_decode(q, k, v, num_splits=1)),
                 ("flash_decode", "auto", lambda: flash_decode(q, k, v))]
        for name, splits, fn in rows:
            ms = try_time(fn)
            if ms is not None:
                print(f"{B},{H},{S},{name},{splits or ''},{ms:.4f},{nbytes / ms / 1e6:.1f}", flush=True)


if __name__ == "__main__":
    main()

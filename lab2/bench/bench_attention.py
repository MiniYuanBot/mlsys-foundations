"""Attention forward: PyTorch SDPA (FlashAttention-2 backend and math backend)
vs your Triton (Part B) and CUDA (Part C1) kernels.

    python lab2/bench/bench_attention.py > results/lab2/attention.csv
    python lab2/bench/bench_attention.py --D 128 --causal
TFLOP/s uses 4*B*H*S^2*D FLOPs (halved for causal).
"""
import argparse

from common import try_time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from lab2.attn import attention_flops
from lab2.attn.cuda import flash_attention_cuda, library_path
from lab2.attn.flash_student import flash_attention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--seqs", default="512,1024,2048,4096,8192")
    a = ap.parse_args()
    print("B,H,S,D,causal,impl,ms,tflops,pct_of_sdpa_flash")
    for S in map(int, a.seqs.split(",")):
        q, k, v = (torch.randn(a.B, a.H, S, a.D, device="cuda", dtype=torch.float16) for _ in range(3))

        def sdpa(backend):
            def run():
                with sdpa_kernel(backend):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=a.causal)
            return run

        impls = {"sdpa_flash": sdpa(SDPBackend.FLASH_ATTENTION), "sdpa_math": sdpa(SDPBackend.MATH),
                 "triton_B": lambda: flash_attention(q, k, v, causal=a.causal)}
        if a.D == 64 and S % 64 == 0 and library_path().is_file():
            impls["cuda_C1"] = lambda: flash_attention_cuda(q, k, v, causal=a.causal)
        flops = attention_flops(a.B, a.H, S, a.D, a.causal)
        base = None
        for name, fn in impls.items():
            if name == "sdpa_math" and S > 4096:
                continue  # materialises S x S scores: slow and memory-hungry
            ms = try_time(fn, repeats=10)
            if ms is None:
                continue
            base = base or ms
            print(f"{a.B},{a.H},{S},{a.D},{int(a.causal)},{name},{ms:.4f},{flops / ms / 1e9:.2f},"
                  f"{100 * base / ms:.1f}", flush=True)
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

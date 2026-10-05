import argparse
import torch
from common.benchmark import measure, save_measurements
from common.records import run_directory
from lab2x.reference import fused_reference
from lab2x.fused_add_rmsnorm import fused_add_rmsnorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["eager", "triton4", "triton8"], required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seed = 42 if args.split == "dev" else 2026
    torch.manual_seed(seed)
    fn = fused_reference if args.implementation == "eager" else lambda x, r, g: fused_add_rmsnorm(
        x, r, g, num_warps=int(args.implementation[-1]))
    shapes = [(m, n) for m in (1, 128) for n in (128, 513, 1024)] if args.split == "dev" else [(7, 257), (7, 1537)]
    rows = []
    with run_directory(args.out, {**vars(args), "device": "cuda", "seed": seed,
                                  "dtype": "float32", "warmup": 20, "iterations": 100}) as out:
        for m, n in shapes:
            inputs = (torch.randn(m, n, device="cuda"), torch.randn(m, n, device="cuda"),
                      torch.rand(n, device="cuda") + 0.5)
            records = measure(fn, inputs, fused_reference, "cuda")
            for row in records:
                row.update(M=m, N=n, shape=f"M{m}-N{n}", implementation=args.implementation, split=args.split)
            rows.extend(records)
            save_measurements(out, rows)
            print(m, n, records[0]["max_abs_error"])


if __name__ == "__main__":
    main()


import argparse
import torch
from common.benchmark import measure, save_measurements
from common.records import run_directory
from lab1.attention import attention, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["explicit", "sdpa"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    rows = []
    fn = attention if args.implementation == "explicit" else reference
    oracle = reference if args.implementation == "explicit" else attention
    with run_directory(args.out, {**vars(args), "seed": 42, "warmup": 20, "dtype": "float32",
                                  "backend_note": "SDPA auto; confirm using separate profiler trace"}) as out:
        for batch in (1, 4):
            for sequence in (128, 256, 512):
                inputs = tuple(torch.randn(batch, 4, sequence, 64, device=args.device) for _ in range(3))
                records = measure(fn, inputs, oracle, args.device, iterations=args.iterations)
                for row in records:
                    row.update(implementation=args.implementation, shape=f"B{batch}-S{sequence}",
                               B=batch, h=4, S=sequence, d=64, backend_note="auto/unconfirmed")
                rows.extend(records)
                print(batch, sequence, records[0]["max_abs_error"])
                save_measurements(out, rows)


if __name__ == "__main__":
    main()


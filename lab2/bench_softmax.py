import argparse
import torch
from common.benchmark import measure, save_measurements
from common.records import run_directory
from lab2.reference import explicit_softmax
from lab2.softmax_triton import softmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["explicit", "torch", "triton4", "triton8"], required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    seed = 42 if args.split == "dev" else 2026
    torch.manual_seed(seed)
    functions = {"explicit": explicit_softmax, "torch": lambda x: x.softmax(-1),
                 "triton4": lambda x: softmax(x, 4), "triton8": lambda x: softmax(x, 8)}
    shapes = [(m, n) for m in (128, 1024) for n in (128, 513, 1024)] if args.split == "dev" else [(7, 257), (7, 1537)]
    rows = []
    with run_directory(args.out, {**vars(args), "device": "cuda", "seed": seed, "warmup": 20,
                                  "dtype": "float32"}) as out:
        for m, n in shapes:
            x = torch.randn(m, n, device="cuda")
            records = measure(functions[args.implementation], (x,), lambda x: x.softmax(-1),
                              "cuda", iterations=args.iterations)
            for row in records:
                row.update(M=m, N=n, shape=f"M{m}-N{n}", implementation=args.implementation, split=args.split)
            rows.extend(records)
            save_measurements(out, rows)
            print(m, n, records[0]["max_abs_error"])


if __name__ == "__main__":
    main()


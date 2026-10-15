"""Plot GFLOP/s vs matrix size from one or more bench_gemm / blas_baseline CSVs.

    python lab0/scripts/plot_gemm.py results/lab0/gemm.csv results/lab0/blas_1t.csv \
        --peak 140 --out results/lab0/gemm.png

Each (kernel, threads) pair becomes one line; the value plotted is the median
over repeats, with the min-max range as a shaded band. --peak draws your
computed single-core FP32 peak as a dashed line (see lecture L0 section 3.1).
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--peak", type=float, help="single-core peak GFLOP/s to draw as a reference line"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = defaultdict(lambda: defaultdict(list))
    for path in args.inputs:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(line for line in f if not line.startswith("#")):
                key = f"{row['kernel']} ({row['threads']}t)"
                series[key][int(row["n"])].append(float(row["gflops"]))

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for key, by_n in sorted(series.items()):
        ns = sorted(by_n)
        med = [statistics.median(by_n[n]) for n in ns]
        ax.plot(ns, med, marker="o", label=key)
        ax.fill_between(ns, [min(by_n[n]) for n in ns], [max(by_n[n]) for n in ns], alpha=0.15)
    if args.peak:
        ax.axhline(args.peak, ls="--", color="gray", label=f"1-core peak {args.peak:g}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("matrix size n (n x n x n)")
    ax.set_ylabel("GFLOP/s (median of repeats)")
    ax.set_title("lab0 GEMM ladder")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

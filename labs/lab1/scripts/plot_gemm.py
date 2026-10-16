"""Plot the lab1 GEMM ladder from bench_gemm CSV output.

    python lab1/scripts/plot_gemm.py results/lab1/gemm.csv --out results/lab1/gemm.png \
        --fp32-peak 15.1 --fp16-peak 60

Left panel: TFLOP/s vs n per kernel (median of repeats, min-max band).
Right panel: bar chart of % of cuBLAS at the largest n, the "ladder" view.
Peaks are your computed device peaks (lecture L1 section 1.4); optional.
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
    parser.add_argument("csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fp32-peak", type=float)
    parser.add_argument("--fp16-peak", type=float)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tflops = defaultdict(lambda: defaultdict(list))
    pct = defaultdict(lambda: defaultdict(list))
    order = []
    with open(args.csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(line for line in f if not line.startswith("#")):
            key = f"{row['kernel']} ({row['dtype']})"
            if key not in order:
                order.append(key)
            n = int(row["n"])
            tflops[key][n].append(float(row["tflops"]))
            pct[key][n].append(float(row["pct_of_cublas"]))

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for key in order:
        ns = sorted(tflops[key])
        med = [statistics.median(tflops[key][n]) for n in ns]
        style = "--" if key.startswith("cublas") else "-"
        left.plot(ns, med, style, marker="o", label=key)
        left.fill_between(
            ns, [min(tflops[key][n]) for n in ns], [max(tflops[key][n]) for n in ns], alpha=0.15
        )
    for peak, label in ((args.fp32_peak, "FP32 peak"), (args.fp16_peak, "FP16 tensor peak")):
        if peak:
            left.axhline(peak, color="gray", ls=":", label=f"{label} {peak:g}")
    left.set(
        xscale="log",
        yscale="log",
        xlabel="n (n x n x n)",
        ylabel="TFLOP/s (median)",
        title="GEMM throughput",
    )
    left.set_xscale("log", base=2)
    left.grid(alpha=0.25, which="both")
    left.legend(fontsize=7)

    largest = max(n for key in order for n in pct[key])
    keys = [k for k in order if largest in pct[k]]
    vals = [statistics.median(pct[k][largest]) for k in keys]
    right.barh(keys, vals, color=["gray" if k.startswith("cublas") else "tab:blue" for k in keys])
    for y, v in enumerate(vals):
        right.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=8)
    right.invert_yaxis()
    right.set(xlabel=f"% of cuBLAS (same dtype), n = {largest}", title="The ladder")
    right.grid(alpha=0.25, axis="x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

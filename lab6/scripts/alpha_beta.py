"""Fit the alpha-beta model to ping-pong data and compare it with your allreduce timings.

    mpirun -np 2 build/lab6/bench_coll --pingpong > results/lab6/pingpong.csv
    mpirun -np 4 build/lab6/bench_coll           > results/lab6/allreduce_p4.csv
    python lab6/scripts/alpha_beta.py results/lab6/pingpong.csv results/lab6/allreduce_p4.csv \
        --out results/lab6/alpha_beta.png

Model: sending n bytes point-to-point costs  T(n) = alpha + n * beta.
Predicted allreduce times (lecture L6 section 2.4), P ranks, n bytes:
    star  2 (P - 1) (alpha + n beta)                      (rank 0 serialises everything)
    rd    log2(P) (alpha + n beta)
    ring  2 (P - 1) alpha + 2 (P - 1) / P * n * beta
The reduction arithmetic (gamma term) is ignored on purpose: the table shows
how much it, and caches, matter.
"""
import argparse
import csv
import math


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pingpong")
    ap.add_argument("allreduce", nargs="?")
    ap.add_argument("--max-fit-bytes", type=float, default=4 << 20,
                    help="fit only messages up to this size (larger ones fall out of cache)")
    ap.add_argument("--out")
    a = ap.parse_args()

    pts = [(float(r["bytes"]), float(r["one_way_us"]) * 1e-6) for r in read(a.pingpong)]
    # Fit alpha and beta SEPARATELY. A plain least-squares line through
    # log-spaced sizes is dominated by the largest messages and returns a
    # meaningless alpha (try it: ~1.7 us instead of the measured ~0.3 us).
    #   alpha: the one-way time of tiny messages (median over n <= 256 B)
    #   beta:  least squares through the origin of (t - alpha) vs n, 16 KiB <= n <= max-fit
    small = sorted(t for n, t in pts if n <= 256)
    alpha = small[len(small) // 2]
    big = [(n, t - alpha) for n, t in pts if 16 * 1024 <= n <= a.max_fit_bytes]
    beta = sum(n * t for n, t in big) / sum(n * n for n, _ in big)
    print(f"alpha = {alpha * 1e6:.3f} us    beta = {beta * 1e9:.4f} ns/B  ({1 / beta / 1e9:.1f} GB/s)")
    print(f"half-bandwidth message size n_1/2 = alpha / beta = {alpha / beta / 1024:.1f} KiB")

    rows = []
    if a.allreduce:
        print(f"\n{'algo':5} {'P':>2} {'bytes':>10} {'measured_us':>12} {'model_us':>10} {'ratio':>6}")
        for r in read(a.allreduce):
            P, n, t = int(r["P"]), float(r["bytes"]), float(r["us"]) * 1e-6
            algo = r["algo"]
            if algo == "star":
                model = 2 * (P - 1) * (alpha + n * beta)
            elif algo == "rd":
                model = math.log2(P) * (alpha + n * beta)
            elif algo == "ring":
                model = 2 * (P - 1) * alpha + 2 * (P - 1) / P * n * beta
            else:
                continue
            rows.append((algo, P, n, t, model))
            print(f"{algo:5} {P:>2} {int(n):>10} {t * 1e6:>12.2f} {model * 1e6:>10.2f} {t / model:>6.2f}")

    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (l, r) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        ns = [n for n, _ in pts]
        l.loglog(ns, [t * 1e6 for _, t in pts], "o", label="ping-pong (one way)")
        l.loglog(ns, [(alpha + n * beta) * 1e6 for n in ns], "-", label=f"alpha={alpha*1e6:.2f}us, {1/beta/1e9:.1f} GB/s")
        l.set(xlabel="message bytes", ylabel="us", title="alpha-beta fit")
        l.legend(); l.grid(alpha=0.3, which="both")
        for algo, color in (("ring", "C0"), ("rd", "C1"), ("star", "C2")):
            sel = [x for x in rows if x[0] == algo]
            if not sel:
                continue
            r.loglog([x[2] for x in sel], [x[3] * 1e6 for x in sel], "o", color=color, label=f"{algo} measured")
            r.loglog([x[2] for x in sel], [x[4] * 1e6 for x in sel], "--", color=color, label=f"{algo} model")
        r.set(xlabel="message bytes", ylabel="us", title=f"allreduce, P={rows[0][1] if rows else '?'}")
        r.legend(fontsize=8); r.grid(alpha=0.3, which="both")
        fig.savefig(a.out, dpi=150)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

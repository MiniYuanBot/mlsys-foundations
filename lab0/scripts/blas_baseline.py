"""Strong baseline for Part B: NumPy's BLAS (OpenBLAS / MKL / Apple Accelerate).

Prints CSV in the same format as bench_gemm, so you can concatenate them:

    python lab0/scripts/blas_baseline.py --threads 1 > results/lab0/blas_1t.csv
    python lab0/scripts/blas_baseline.py --threads 8 > results/lab0/blas_8t.csv

The thread count must be set before NumPy is imported, which is why this
script sets the environment variables itself.
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sizes", default="128,256,512,1024")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    for var in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(args.threads)
    import numpy as np

    print(f"# numpy {np.__version__}", file=sys.stderr)
    try:
        info = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        print(f"# blas: {info.get('name')} {info.get('version', '')}", file=sys.stderr)
    except Exception:  # older NumPy
        pass

    print("kernel,threads,n,repeat,ms,gflops")
    rng = np.random.default_rng(0)
    for n in (int(s) for s in args.sizes.split(",")):
        a = rng.uniform(-1, 1, (n, n)).astype(np.float32)
        b = rng.uniform(-1, 1, (n, n)).astype(np.float32)
        a @ b  # warm-up
        for r in range(1, args.repeats + 1):
            iters, elapsed = 1, 0.0
            while True:
                t0 = time.perf_counter()
                for _ in range(iters):
                    a @ b
                elapsed = time.perf_counter() - t0
                if elapsed >= 0.05 or iters >= 1 << 16:
                    break
                iters *= 2
            ms = elapsed * 1000 / iters
            print(f"blas,{args.threads},{n},{r},{ms:.4f},{2 * n**3 / (ms * 1e6):.2f}")


if __name__ == "__main__":
    main()

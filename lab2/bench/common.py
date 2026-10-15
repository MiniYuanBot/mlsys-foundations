"""Timing helpers shared by the lab2 benchmarks."""

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root


def time_ms(fn, warmup=5, repeats=20):
    """Median ms per call over `repeats` CUDA-event-timed calls (after warm-up)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def try_time(fn, **kw):
    """time_ms, or None if the implementation is not done yet / runs out of memory."""
    try:
        fn()
    except NotImplementedError as e:
        print(f"# skip: {e}", file=sys.stderr)
        return None
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("# skip: out of memory", file=sys.stderr)
        return None
    return time_ms(fn, **kw)

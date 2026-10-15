"""Supplied synchronous forward benchmark. Input allocation is outside timing."""

import gc
import time
import torch
from common.io import write_csv, write_json


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def compare(actual, expected):
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    if len(actual) != len(expected):
        raise AssertionError("Output count mismatch")
    errors = []
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
        if not torch.isfinite(left).all():
            raise AssertionError("Output contains non-finite values")
        errors.append((left - right).abs().max().item())
    return errors


@torch.no_grad()
def measure(fn, inputs, oracle, device, warmup=20, iterations=100):
    # First call includes compilation if present; reference is not in this timing.
    synchronize(device)
    start = time.perf_counter()
    actual = fn(*inputs)
    synchronize(device)
    first_call_ms = (time.perf_counter() - start) * 1000
    expected = oracle(*inputs)
    errors = compare(actual, expected)
    del actual, expected
    for _ in range(warmup):
        output = fn(*inputs)
        del output
    synchronize(device)
    rows = []
    for repeat in (1, 2, 3):
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            output = fn(*inputs)
            # Keep each result alive until the next iteration, consistently for all implementations.
        synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000 / iterations
        del output
        gc.collect()
        memory = dict(
            allocated_baseline_bytes=None,
            peak_allocated_bytes=None,
            peak_increment_bytes=None,
            reserved_bytes=None,
        )
        if torch.device(device).type == "cuda":
            synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            output = fn(*inputs)
            synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            memory = dict(
                allocated_baseline_bytes=baseline,
                peak_allocated_bytes=peak,
                peak_increment_bytes=peak - baseline,
                reserved_bytes=torch.cuda.memory_reserved(device),
            )
            del output
        rows.append(
            dict(
                repeat=repeat,
                warmup=warmup,
                iterations=iterations,
                latency_ms=latency_ms,
                first_call_ms=first_call_ms,
                max_abs_error=max(errors),
                output_errors=repr(errors),
                device=str(device),
                dtype="float32",
                **memory,
            )
        )
    return rows


def save_measurements(out, rows):
    write_csv(out / "raw.csv", rows)
    write_json(
        out / "correctness.json",
        [
            {key: row[key] for key in ("shape", "implementation", "max_abs_error", "output_errors")}
            for row in rows
            if row["repeat"] == 1
        ],
    )
    import statistics

    summary = []
    keys = sorted(set((r["shape"], r["implementation"]) for r in rows))
    for shape, implementation in keys:
        values = [
            r["latency_ms"]
            for r in rows
            if (r["shape"], r["implementation"]) == (shape, implementation)
        ]
        summary.append(
            dict(
                shape=shape,
                implementation=implementation,
                median_ms=statistics.median(values),
                min_ms=min(values),
                max_ms=max(values),
            )
        )
    write_csv(out / "summary.csv", summary)

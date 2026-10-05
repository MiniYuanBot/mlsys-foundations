"""Diagnostic trace only. Do not include profiler overhead in performance tables."""
import argparse
from pathlib import Path
import torch
from lab1.attention import attention, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    inputs = tuple(torch.randn(4, 4, 256, 64, device=args.device) for _ in range(3))
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        for _ in range(20):
            attention(*inputs)
            reference(*inputs)
        if args.device == "cuda":
            torch.cuda.synchronize()
        with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
            with torch.profiler.record_function("explicit"):
                attention(*inputs)
            with torch.profiler.record_function("sdpa_auto"):
                reference(*inputs)
            if args.device == "cuda":
                torch.cuda.synchronize()
    profile.export_chrome_trace(str(path))
    print(profile.key_averages().table(row_limit=20))


if __name__ == "__main__":
    main()


"""Deterministic distributed training harness; student work is in student.py."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import time


def make_problem(torch, samples):
    # No downloads, random sampling, or GPU. Non-identical rank distributions
    # expose omitted gradient synchronization and incorrect average-of-means.
    index = torch.arange(samples, dtype=torch.float64)
    x = torch.stack(((index - 3.0) / 5.0, (index.remainder(3) - 1.0) / 2.0), dim=1)
    y = 0.7 * x[:, :1] - 1.2 * x[:, 1:] + 0.3 + 0.01 * index[:, None]
    model = torch.nn.Linear(2, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.1]], dtype=torch.float64))
        model.bias.fill_(-0.2)
    return x, y, model


def flat_parameters(torch, model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).tolist()


def full_batch_oracle(torch, samples, steps, lr):
    x, y, model = make_problem(torch, samples)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        history.append({"step": step, "loss_before_update": loss.item(),
                        "parameters": flat_parameters(torch, model)})
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 1 or args.lr <= 0:
        parser.error("samples, steps and learning rate must be positive")

    import torch
    import torch.distributed as dist
    from .student import synchronize_gradients

    torch.set_num_threads(1)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.samples < world_size:
        parser.error("each rank must have at least one sample")
    if "RANK" not in os.environ:
        parser.error("launch with python -m torch.distributed.run (including world_size=1)")
    args.out.mkdir(parents=True, exist_ok=True)
    # Each rank owns its files; refusing reuse keeps evidence from separate runs
    # from silently being mixed. Use a fresh output directory for every run.
    owned = [args.out / f"rank{rank}.json", args.out / f"rank{rank}.trace.json"]
    if rank == 0:
        owned.append(args.out / "oracle.json")
    if any(path.exists() for path in owned):
        parser.error("output files already exist; choose a fresh --out directory")
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    try:
        x, y, model = make_problem(torch, args.samples)
        indices = torch.arange(rank, args.samples, world_size)
        local_x, local_y = x[indices], y[indices]
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                          record_shapes=True)
                    if args.trace else nullcontext())
        history = []
        # Synchronize the start, outside the measured training region. No extra
        # per-step barriers: collectives/student synchronization determine ordering.
        dist.barrier()
        start = time.perf_counter()
        with profiler as trace:
            for step in range(args.steps):
                with torch.profiler.record_function("local_forward_backward"):
                    optimizer.zero_grad(set_to_none=True)
                    local_loss = (model(local_x) - local_y).square().mean()
                    local_loss.backward()
                before = [p.grad.detach().reshape(-1).tolist() for p in model.parameters()]
                with torch.profiler.record_function("gradient_sync"):
                    synchronize_gradients(list(model.parameters()), len(indices))
                after = [p.grad.detach().reshape(-1).tolist() for p in model.parameters()]
                with torch.profiler.record_function("optimizer_step"):
                    optimizer.step()
                history.append({"step": step, "local_loss": local_loss.item(),
                                "gradients_before_sync": before,
                                "gradients_after_sync": after,
                                "parameters": flat_parameters(torch, model)})
        elapsed = time.perf_counter() - start
        if args.trace:
            trace.export_chrome_trace(str(args.out / f"rank{rank}.trace.json"))
        result = {"rank": rank, "world_size": world_size, "backend": dist.get_backend(),
                  "torch_version": torch.__version__, "pid": os.getpid(),
                  "device": "cpu", "dtype": "float64", "samples": args.samples,
                  "local_samples": len(indices), "indices": indices.tolist(),
                  "lr": args.lr, "steps": args.steps, "elapsed_seconds": elapsed,
                  "timing_scope": "diagnostic loop including Python record collection; not steady-state throughput",
                  "profiled": args.trace, "history": history}
        (args.out / f"rank{rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # Oracle runs after the timed region and never provides training gradients.
        if rank == 0:
            oracle = {"samples": args.samples, "steps": args.steps, "lr": args.lr,
                      "history": full_batch_oracle(torch, args.samples, args.steps, args.lr)}
            (args.out / "oracle.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

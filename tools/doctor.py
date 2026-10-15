"""Read-only smoke checks; never installs packages or downloads a model."""

import argparse
import importlib
import sys
from common.io import write_json
from common.records import environment


def inspect(require):
    report = environment()
    checks = {}
    if require != "none":
        try:
            import torch

            x = torch.randn(8, 8, requires_grad=True)
            (x @ x.T).sum().backward()
            checks["torch"] = bool(torch.isfinite(x.grad).all())
            checks["wheel_cuda"] = torch.version.cuda
            checks["cuda"] = torch.cuda.is_available()
            if require in ("cuda", "triton", "vllm") and checks["cuda"]:
                y = torch.randn(128, 128, device="cuda")
                z = y @ y
                torch.cuda.synchronize()
                checks["cuda"] = bool(torch.isfinite(z).all())
                checks["device"] = torch.cuda.get_device_name(0)
                checks["capability"] = torch.cuda.get_device_capability(0)
        except Exception as exc:
            checks["torch"] = False
            checks["error"] = f"{type(exc).__name__}: {exc}"
        if require in ("triton", "vllm"):
            try:
                importlib.import_module(require)
                checks[require] = True
            except Exception as exc:
                checks[require] = False
                checks[require + "_error"] = str(exc)
    required = {
        "none": [],
        "torch": ["torch"],
        "cuda": ["torch", "cuda"],
        "triton": ["torch", "cuda", "triton"],
        "vllm": ["torch", "cuda", "vllm"],
    }[require]
    report.update(
        checks=checks,
        required=require,
        status="ready" if all(checks.get(k) is True for k in required) else "blocked",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require", choices=["none", "torch", "cuda", "triton", "vllm"], default="none"
    )
    parser.add_argument("--out", default="results/environment/doctor.json")
    args = parser.parse_args()
    result = inspect(args.require)
    write_json(args.out, result)
    print(result["status"], "->", args.out)
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    sys.exit(main())

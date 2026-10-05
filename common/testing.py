import unittest


def require_torch(cuda=False, triton=False):
    try:
        import torch
    except ImportError as exc:
        raise unittest.SkipTest("ENV: PyTorch not installed; follow setup.md") from exc
    if cuda and not torch.cuda.is_available():
        raise unittest.SkipTest("ENV: CUDA unavailable; CPU results do not pass GPU checks")
    if triton:
        try:
            import triton
        except ImportError as exc:
            raise unittest.SkipTest("ENV: Triton not installed") from exc
    return torch


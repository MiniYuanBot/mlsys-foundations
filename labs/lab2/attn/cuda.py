"""ctypes bridge to the CUDA kernel of Part C1 (build/lab2/libflash_mma.so).

Provided. Build first:
    cmake -S lab2 -B build/lab2 && cmake --build build/lab2 -j
or point MLSYS_LAB2_LIB at the shared library.
"""

import ctypes
import math
import os
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_lib = None


def library_path() -> Path:
    return Path(os.environ.get("MLSYS_LAB2_LIB", _ROOT / "build" / "lab2" / "libflash_mma.so"))


def _load():
    global _lib
    if _lib is None:
        path = library_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found: build lab2 with CMake first (see lab2/README.md)"
            )
        _lib = ctypes.CDLL(str(path))
        _lib.flash_mma_forward.restype = ctypes.c_int
        _lib.flash_mma_forward.argtypes = [ctypes.c_void_p] * 5 + [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        ]
        _lib.flash_mma_error.restype = ctypes.c_char_p
    return _lib


HEAD_DIM = 64
BLOCK = 64


def flash_attention_cuda(q, k, v, causal=False, sm_scale=None):
    """q, k, v: [B, H, S, 64] fp16 CUDA tensors, S a multiple of 64. Returns (out fp16, lse fp32)."""
    B, H, S, D = q.shape
    if D != HEAD_DIM or S % BLOCK or q.dtype != torch.float16 or not q.is_cuda:
        raise ValueError(
            f"flash_attention_cuda needs fp16 CUDA tensors, D == {HEAD_DIM}, S % {BLOCK} == 0"
        )
    q, k, v = (t.contiguous() for t in (q, k, v))
    out = torch.empty_like(q)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    lib = _load()
    rc = lib.flash_mma_forward(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        lse.data_ptr(),
        B * H,
        S,
        int(causal),
        sm_scale,
    )
    if rc == 1:
        raise NotImplementedError(lib.flash_mma_error().decode())
    if rc != 0:
        raise RuntimeError(lib.flash_mma_error().decode())
    return out, lse

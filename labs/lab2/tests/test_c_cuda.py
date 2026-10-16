"""Part C1 tests (CUDA, ctypes):  python -m pytest lab2/tests/test_c_cuda.py -v
Build first:  cmake -S lab2 -B build/lab2 && cmake --build build/lab2 -j"""

import pytest
import torch

from lab2.attn import attention_ref
from lab2.attn.cuda import flash_attention_cuda, library_path
from lab2.tests.helpers import qkv

pytestmark = pytest.mark.skipif(not library_path().is_file(), reason=f"{library_path()} not built")
ATOL = RTOL = 2e-3


def check(out, lse, ref_out, ref_lse):
    assert torch.isfinite(out).all(), (
        "non-finite output (an unwritten row, or a NaN in the softmax state)"
    )
    torch.testing.assert_close(out.float(), ref_out, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


def test_C1_one_tile():
    q, k, v = qkv(1, 1, 64, 64)  # one query tile, one key tile: no rescaling across tiles
    check(*flash_attention_cuda(q, k, v), *attention_ref(q, k, v))


@pytest.mark.parametrize("S", [128, 512, 2048])
def test_C1_noncausal(S):
    q, k, v = qkv(2, 3, S, 64)
    check(*flash_attention_cuda(q, k, v), *attention_ref(q, k, v))


def test_C1_peaky():
    q, k, v = qkv(1, 2, 512, 64, scale=3.0)
    check(*flash_attention_cuda(q, k, v), *attention_ref(q, k, v))


@pytest.mark.parametrize("S", [64, 192, 1024])
def test_C1_causal(S):
    q, k, v = qkv(2, 2, S, 64)
    check(*flash_attention_cuda(q, k, v, causal=True), *attention_ref(q, k, v, causal=True))


def test_C1_rejects_bad_shape():
    q, k, v = qkv(1, 1, 100, 64)
    with pytest.raises(ValueError):
        flash_attention_cuda(q, k, v)

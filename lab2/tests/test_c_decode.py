"""Part C2 tests:  python -m pytest lab2/tests/test_c_decode.py -v"""
import pytest
import torch

from lab2.attn import decode_ref
from lab2.attn.decode_student import flash_decode


def make(B, H, S, D, scale=1.0):
    q = (torch.randn(B, H, D, device="cuda") * scale).half()
    k = (torch.randn(B, H, S, D, device="cuda") * scale).half()
    v = torch.randn(B, H, S, D, device="cuda").half()
    return q, k, v


@pytest.mark.parametrize("S", [1, 63, 64, 1000, 8192])
@pytest.mark.parametrize("splits", [1, 2, 7, None])
def test_C2_decode_matches_reference(S, splits):
    q, k, v = make(2, 4, S, 64)
    out, lse = flash_decode(q, k, v, num_splits=splits)
    torch.testing.assert_close(out.float(), decode_ref(q, k, v), atol=2e-3, rtol=2e-3)
    ref_lse = torch.logsumexp((q.float()[:, :, None, :] @ k.float().transpose(-1, -2))[:, :, 0] / 8.0, dim=-1)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("D", [32, 128])
def test_C2_other_head_dims(D):
    q, k, v = make(1, 8, 3000, D)
    out, _ = flash_decode(q, k, v)
    torch.testing.assert_close(out.float(), decode_ref(q, k, v), atol=2e-3, rtol=2e-3)


def test_C2_split_count_does_not_change_result():
    q, k, v = make(1, 2, 4096, 64, scale=2.0)
    a, _ = flash_decode(q, k, v, num_splits=1)
    b, _ = flash_decode(q, k, v, num_splits=32)
    torch.testing.assert_close(a.float(), b.float(), atol=2e-3, rtol=2e-3)

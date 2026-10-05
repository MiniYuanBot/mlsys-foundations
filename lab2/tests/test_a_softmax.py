"""Part A tests:  python -m pytest lab2/tests/test_a_softmax.py -v"""
import pytest
import torch

from lab2.attn import softmax_ref
from lab2.attn.softmax_student import softmax_fused, softmax_online

SHAPES = [(1, 1), (3, 7), (64, 128), (100, 1000), (17, 4097), (8, 16384)]


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_A1_fused_matches_reference(rows, cols):
    x = torch.randn(rows, cols, device="cuda") * 5
    torch.testing.assert_close(softmax_fused(x), softmax_ref(x), atol=1e-6, rtol=1e-5)


def test_A1_large_values_do_not_overflow():
    # exp(1000) overflows fp32: only a max-subtracted softmax survives this.
    x = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -1000.0, -1000.0]], device="cuda")
    y = softmax_fused(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, softmax_ref(x))


def test_A1_strided_rows():
    x = torch.randn(32, 200, device="cuda")[:, :150]  # row stride 200, 150 columns
    torch.testing.assert_close(softmax_fused(x), softmax_ref(x), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("rows,cols", SHAPES + [(4, 100_000), (2, 1 << 20)])
def test_A2_online_matches_reference(rows, cols):
    x = torch.randn(rows, cols, device="cuda") * 5
    y, lse = softmax_online(x)
    torch.testing.assert_close(y, softmax_ref(x), atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(lse, torch.logsumexp(x, dim=-1), atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("block", [16, 128, 4096])
def test_A2_result_independent_of_block(block):
    x = torch.randn(5, 3000, device="cuda") * 10
    y, lse = softmax_online(x, block=block)
    torch.testing.assert_close(y, softmax_ref(x), atol=1e-6, rtol=1e-4)


def test_A2_increasing_row_forces_rescaling():
    # The max grows in every chunk, so every chunk must rescale the old sum.
    x = torch.linspace(0, 50, 10_000, device="cuda")[None, :].repeat(3, 1)
    y, lse = softmax_online(x, block=64)
    torch.testing.assert_close(y, softmax_ref(x), atol=1e-6, rtol=1e-4)

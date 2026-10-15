"""Part B tests:  python -m pytest lab2/tests/test_b_flash.py -v"""

import pytest
import torch

from lab2.attn import attention_ref
from lab2.attn.flash_student import attention_single_block, flash_attention
from lab2.tests.helpers import qkv

ATOL = RTOL = 2e-3  # fp16 inputs and fp16 P, fp32 accumulation


def check(out, lse, ref_out, ref_lse):
    assert out.dtype == torch.float16 and lse.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(out.float(), ref_out, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("S", [1, 7, 33, 64])
@pytest.mark.parametrize("D", [32, 64, 128])
def test_B1_single_block(S, D):
    q, k, v = qkv(2, 3, S, D)
    check(*attention_single_block(q, k, v), *attention_ref(q, k, v))


def test_B1_rejects_long_sequence():
    q, k, v = qkv(1, 1, 65, 64)
    with pytest.raises(ValueError):
        attention_single_block(q, k, v)


@pytest.mark.parametrize("S", [1, 50, 64, 65, 200, 1024])
@pytest.mark.parametrize("D", [16, 64, 128])
def test_B2_flash_noncausal(S, D):
    q, k, v = qkv(2, 4, S, D)
    check(*flash_attention(q, k, v), *attention_ref(q, k, v))


def test_B2_peaky_scores_need_rescaling():
    # Large scores: the running max changes between key blocks, so a missing
    # alpha rescale shows up immediately.
    q, k, v = qkv(1, 2, 512, 64, scale=3.0)
    check(*flash_attention(q, k, v), *attention_ref(q, k, v))


@pytest.mark.parametrize("bm,bn", [(16, 16), (32, 64), (64, 32), (128, 64)])
def test_B2_result_independent_of_tiles(bm, bn):
    q, k, v = qkv(1, 2, 300, 64)
    check(*flash_attention(q, k, v, block_m=bm, block_n=bn), *attention_ref(q, k, v))


@pytest.mark.parametrize("S", [1, 63, 64, 65, 333, 1024])
def test_B3_flash_causal(S):
    q, k, v = qkv(2, 2, S, 64)
    check(*flash_attention(q, k, v, causal=True), *attention_ref(q, k, v, causal=True))


def test_B3_causal_first_row_sees_only_itself():
    q, k, v = qkv(1, 1, 128, 64)
    out, _ = flash_attention(q, k, v, causal=True)
    torch.testing.assert_close(out[0, 0, 0].float(), v[0, 0, 0].float(), atol=1e-3, rtol=0)


def test_B_inputs_not_modified():
    q, k, v = qkv(1, 2, 200, 64)
    copies = [t.clone() for t in (q, k, v)]
    flash_attention(q, k, v, causal=True)
    for a, b in zip((q, k, v), copies):
        assert torch.equal(a, b)

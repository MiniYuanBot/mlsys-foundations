"""Adaptation scaffold, not a working kernel.
Source: https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""
import triton
import triton.language as tl


@triton.jit
def row_softmax(X, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * stride_x + columns, mask=columns < N, other=-float("inf"))
    # TODO L02-1: reduce max, subtract, exp, reduce sum, normalize.
    # Replace the placeholder zeros. Padding contributes zero after exp(-inf).
    probabilities = tl.full((BLOCK,), 0, tl.float32)
    tl.store(Y + row * stride_y + columns, probabilities, mask=columns < N)


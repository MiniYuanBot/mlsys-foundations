"""Part A: softmax in Triton. Fill in every TODO(A*).

    python -m pytest lab2/tests/test_a_softmax.py -v
    python lab2/bench/bench_softmax.py

Read lecture L2 sections 1-2 first. Inputs are 2-D fp32 tensors [rows, cols],
softmax is over the last dim. Triton tutorial for reference:
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
"""
import torch
import triton
import triton.language as tl


# ============================================================== A1 ====
# TODO(A1) fused row softmax: one program per row, the WHOLE row in one block.
#   cols = tl.arange(0, BLOCK); mask = cols < n_cols
#   x = load(row), padding = -inf (so it does not change max or sum)
#   y = exp(x - max(x)) / sum(exp(x - max(x)))    (subtract the max for stability)
# The row is read once and written once: 2 * 4 * n_cols bytes per row, versus
# 5 passes over memory for the unfused torch.exp / sum / div sequence.
@triton.jit
def softmax_row_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pass


A1_DONE = False  # set to True when softmax_row_kernel is written
MAX_FUSED_COLS = 16384  # one row must fit in registers / shared memory


def softmax_fused(x: torch.Tensor) -> torch.Tensor:
    """Provided launcher for A1."""
    if not A1_DONE:
        raise NotImplementedError("TODO A1 softmax_row_kernel")
    assert x.ndim == 2 and x.is_cuda and x.dtype == torch.float32
    rows, cols = x.shape
    if cols > MAX_FUSED_COLS:
        raise ValueError(f"softmax_fused handles at most {MAX_FUSED_COLS} columns; use softmax_online")
    y = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    num_warps = 4 if block <= 2048 else 8 if block <= 8192 else 16
    softmax_row_kernel[(rows,)](x, y, x.stride(0), y.stride(0), cols, BLOCK=block, num_warps=num_warps)
    return y


# ============================================================== A2 ====
# TODO(A2) online softmax: rows of ANY length, read in chunks of BLOCK.
# Pass 1 streams over the row keeping a running max m and running sum l:
#     m_new = max(m, max(chunk))
#     l     = l * exp(m - m_new) + sum(exp(chunk - m_new))      <- rescale old sum
#     m     = m_new
# Pass 2 streams again and writes exp(x - m) / l.
# Also store lse[row] = m + log(l) (the log-sum-exp; Part B reuses this idea).
# Lecture L2 section 2 derives why the rescaling is exact.
# Start with m = -float("inf") and l = 0.0; exp(-inf) = 0 makes the first
# rescale harmless.
@triton.jit
def softmax_online_kernel(x_ptr, y_ptr, lse_ptr, stride_x, stride_y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y
    pass


A2_DONE = False  # set to True when softmax_online_kernel is written


def softmax_online(x: torch.Tensor, block: int = 1024):
    """Provided launcher for A2. Returns (y, lse)."""
    if not A2_DONE:
        raise NotImplementedError("TODO A2 softmax_online_kernel")
    assert x.ndim == 2 and x.is_cuda and x.dtype == torch.float32
    rows, cols = x.shape
    y = torch.empty_like(x)
    lse = torch.empty(rows, device=x.device, dtype=torch.float32)
    softmax_online_kernel[(rows,)](x, y, lse, x.stride(0), y.stride(0), cols, BLOCK=block, num_warps=4)
    return y, lse

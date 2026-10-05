"""Original skeleton. Derive online max/sum/output updates from FlashAttention."""
import triton
import triton.language as tl


@triton.jit
def forward(Q, K, V, O, S: tl.constexpr, D: tl.constexpr,
            BQ: tl.constexpr, BK: tl.constexpr):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    query_rows = query_block * BQ + tl.arange(0, BQ)
    dimensions = tl.arange(0, D)
    base = batch_head * S * D
    q = tl.load(Q + base + query_rows[:, None] * D + dimensions[None, :],
                mask=query_rows[:, None] < S, other=0)
    # TODO S02-2:
    # - initialize per-query running maximum m, normalizer l, FP32 accumulator o;
    # - stream K/V tiles; mask out-of-range keys and causal j>i;
    # - use stable online rescaling of prior state when m changes;
    # - padded rows must not produce invalid stores; avoid -inf - (-inf);
    # - tl.dot input_precision='ieee' for the declared FP32 correctness track;
    # - write normalized o, never write a complete scores/probabilities matrix.
    # The wrapper deliberately refuses to launch until implemented.
    placeholder = tl.full((BQ, D), 0, tl.float32)
    tl.store(O + base + query_rows[:, None] * D + dimensions[None, :],
             placeholder, mask=query_rows[:, None] < S)


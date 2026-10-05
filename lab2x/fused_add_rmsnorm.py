"""L02X: build your kernel from the contract, using L02's launch pattern as a hint."""
def fused_add_rmsnorm(x, residual, gamma, epsilon=1e-5, num_warps=4):
    """Return independent z,y. Inputs contiguous FP32; z=x+residual; no input writes."""
    # TODO L02X-1: validate shape/dtype/device; write and launch your own Triton kernel.
    # Padded squared values contribute 0; divide the sum by actual N, not BLOCK.
    # Both z and y are required outputs. Version C changes only num_warps to 8.
    raise NotImplementedError("L02X-1: design fused residual + RMSNorm kernel")


"""L02: supplied input contract and launch plan; implement the row kernel."""
import torch


def validate(x):
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("expected a nonempty 2D tensor")
    if x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError("expected contiguous FP32")
    if not x.is_cuda:
        raise ValueError("Triton performance path requires CUDA")


def softmax(x, num_warps=4):
    validate(x)
    if num_warps not in (4, 8):
        raise ValueError("this teaching experiment uses num_warps 4 or 8")
    # TODO L02-1: complete kernel.py, then remove this guard.
    raise NotImplementedError("L02-1: fill row softmax arithmetic and enable launch below")
    # Supplied launch reference (uncomment after implementing the kernel):
    # import triton
    # from lab2.kernel import row_softmax
    # output = torch.empty_like(x)
    # block = triton.next_power_of_2(x.shape[1])
    # row_softmax[(x.shape[0],)](x, output, x.stride(0), output.stride(0),
    #                          x.shape[1], BLOCK=block, num_warps=num_warps)
    # return output


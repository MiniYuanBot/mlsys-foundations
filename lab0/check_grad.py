"""L00: centered finite difference, independent of autograd."""
import torch


def finite_difference(x, w, epsilon=1e-6):
    """L=((x@w)**2).mean(); return a tensor with w's shape. Do not use autograd."""
    # TODO L00-3: perturb each coordinate on independent clones.
    raise NotImplementedError("L00-3: implement central differences")


def main():
    torch.manual_seed(42)
    x = torch.randn(2, 2, dtype=torch.float64)
    w = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)
    ((x @ w) ** 2).mean().backward()
    numeric = finite_difference(x, w.detach())
    error = (numeric - w.grad).abs().max().item()
    print({"max_abs_error": error})
    assert error < 1e-5


if __name__ == "__main__":
    main()


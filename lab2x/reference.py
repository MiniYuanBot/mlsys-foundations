"""Independent eager reference; FP32, two outputs, no in-place modifications."""
def fused_reference(x, residual, gamma, epsilon=1e-5):
    z = x + residual
    y = z * (z.square().mean(-1, keepdim=True) + epsilon).rsqrt() * gamma
    return z, y


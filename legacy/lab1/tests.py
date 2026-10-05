import unittest
from common.testing import require_torch


class CPU(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.torch = require_torch(cuda=cls.device == "cuda")

    def test_all_shapes(self):
        from lab1.attention import attention, reference
        torch = self.torch
        torch.manual_seed(42)
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            for batch in (1, 4):
                for sequence in (128, 256, 512):
                    with self.subTest(batch=batch, sequence=sequence):
                        inputs = tuple(torch.randn(batch, 4, sequence, 64, device=self.device) for _ in range(3))
                        torch.testing.assert_close(attention(*inputs), reference(*inputs), rtol=1e-4, atol=1e-5)

    def test_causality(self):
        from lab1.attention import attention
        torch = self.torch
        torch.manual_seed(71)
        q, k, v = [torch.randn(1, 1, 4, 2, device=self.device) for _ in range(3)]
        original = attention(q, k, v)
        torch.testing.assert_close(original[:, :, 0], v[:, :, 0])
        k2, v2 = k.clone(), v.clone()
        k2[:, :, 2:] *= 10
        v2[:, :, 2:] += 100
        torch.testing.assert_close(original[:, :, :2], attention(q, k2, v2)[:, :, :2])


class GPU(CPU):
    device = "cuda"


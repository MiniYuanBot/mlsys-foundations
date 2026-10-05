import unittest
from common.testing import require_torch


class Contract(unittest.TestCase):
    def test_reject_unsupported_input(self):
        torch = require_torch()
        from lab2.softmax_triton import validate
        for x in (torch.ones(3), torch.ones(2, 3, dtype=torch.float64),
                  torch.ones(2, 3).t(), torch.ones(0, 3)):
            with self.assertRaises(ValueError):
                validate(x)


class GPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = require_torch(cuda=True, triton=True)

    def test_shapes_and_extreme(self):
        from lab2.softmax_triton import softmax
        torch = self.torch
        torch.manual_seed(42)
        for m, n in [(m, n) for m in (128, 1024) for n in (128, 513, 1024)] + [(7, 1), (7, 257), (7, 1537)]:
            for scale in (1, 50):
                with self.subTest(m=m, n=n, scale=scale):
                    x = torch.randn(m, n, device="cuda") * scale
                    before = x.clone()
                    y = softmax(x)
                    torch.testing.assert_close(y, x.softmax(-1), rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(y.sum(-1), torch.ones(m, device="cuda"))
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertTrue(torch.equal(x, before))


class Properties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = require_torch(cuda=True, triton=True)

    def test_invariance_and_warps(self):
        from lab2.softmax_triton import softmax
        torch = self.torch
        torch.manual_seed(2026)
        x = torch.randn(7, 257, device="cuda")
        permutation = torch.randperm(257, device="cuda")
        for warps in (4, 8):
            y = softmax(x, warps)
            torch.testing.assert_close(softmax(x + 10, warps), y, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(softmax(x[:, permutation].contiguous(), warps), y[:, permutation], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(softmax(torch.ones_like(x), warps), torch.full_like(x, 1 / 257))
        with self.assertRaises(ValueError):
            softmax(x.t())


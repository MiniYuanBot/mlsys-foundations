import unittest
from common.testing import require_torch


class GPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = require_torch(cuda=True, triton=True)

    def test_shapes_outputs_and_no_mutation(self):
        from lab2x.fused_add_rmsnorm import fused_add_rmsnorm
        from lab2x.reference import fused_reference
        torch = self.torch
        torch.manual_seed(42)
        shapes = [(m, n) for m in (1, 128) for n in (128, 513, 1024)] + [(7, 257), (7, 1537), (3, 1)]
        for m, n in shapes:
            for warps in (4, 8):
                with self.subTest(m=m, n=n, warps=warps):
                    x, r = [torch.randn(m, n, device="cuda") for _ in range(2)]
                    g = torch.rand(n, device="cuda") + 0.5
                    before = [v.clone() for v in (x, r, g)]
                    actual = fused_add_rmsnorm(x, r, g, num_warps=warps)
                    self.assertEqual(len(actual), 2)
                    self.assertNotEqual(actual[0].data_ptr(), actual[1].data_ptr())
                    for out, expected in zip(actual, fused_reference(x, r, g)):
                        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
                        self.assertTrue(torch.isfinite(out).all())
                    for v, copy in zip((x, r, g), before):
                        self.assertTrue(torch.equal(v, copy))

    def test_zero_and_sign(self):
        from lab2x.fused_add_rmsnorm import fused_add_rmsnorm
        torch = self.torch
        torch.manual_seed(73)
        x = torch.randn(7, 513, device="cuda")
        r = torch.randn_like(x)
        g = torch.ones(513, device="cuda")
        z, y = fused_add_rmsnorm(x, -x, g)
        torch.testing.assert_close(z, torch.zeros_like(x))
        torch.testing.assert_close(y, torch.zeros_like(x))
        z, y = fused_add_rmsnorm(x, r, g * 0)
        torch.testing.assert_close(z, x + r)
        torch.testing.assert_close(y, torch.zeros_like(x))
        normal = fused_add_rmsnorm(x, r, g)
        negative = fused_add_rmsnorm(-x, -r, g)
        for a, b in zip(normal, negative):
            torch.testing.assert_close(a, -b, rtol=1e-4, atol=1e-5)


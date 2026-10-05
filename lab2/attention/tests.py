import unittest
from common.testing import require_torch


class GPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = require_torch(cuda=True, triton=True)

    def test_reference_tail_and_tile_invariance(self):
        from lab2.attention.attention import attention
        torch = self.torch
        torch.manual_seed(42)
        torch.set_float32_matmul_precision("highest")
        shapes = [(1,2,s,d) for s in (1,31,64,129,256) for d in (32,64)]
        # Distinguish flattened B*H indexing from an accidentally H-only grid.
        shapes.extend((2,3,65,d) for d in (32,64))
        for batch,heads,sequence,dimension in shapes:
            q, k, v = [torch.randn(batch,heads,sequence,dimension,device="cuda") for _ in range(3)]
            before = [t.clone() for t in (q,k,v)]
            reference = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0)
            for bq in (16,32):
                for bk in (32,64):
                    with self.subTest(B=batch,H=heads,S=sequence,D=dimension,BQ=bq,BK=bk):
                        actual = attention(q,k,v,bq,bk)
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-5)
                        for value, expected in zip((q,k,v),before):
                            self.assertTrue(torch.equal(value,expected))

    def test_causality_and_large_scores(self):
        from lab2.attention.attention import attention
        torch = self.torch
        torch.manual_seed(73)
        q,k,v = [torch.randn(1,1,65,32,device="cuda") for _ in range(3)]
        result = attention(q*10,k*10,v)
        expected = torch.nn.functional.scaled_dot_product_attention(q*10,k*10,v,is_causal=True,dropout_p=0)
        torch.testing.assert_close(result,expected,rtol=1e-4,atol=1e-5)
        original = attention(q,k,v)
        v[:,:,33:]+=100
        torch.testing.assert_close(attention(q,k,v)[:,:,:33],original[:,:,:33],rtol=1e-4,atol=1e-5)

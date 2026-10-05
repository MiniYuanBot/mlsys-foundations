from common.temporary import workspace_temp
import unittest
from pathlib import Path
from common.testing import require_torch


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = require_torch()
        cls.torch.set_num_threads(1)

    def test_update_and_overfit(self):
        from lab0.train_toy import make_data, make_model, train_step, evaluate
        torch = self.torch
        torch.manual_seed(42)
        x, y, _, _ = make_data()
        x, y = x[:32], y[:32]
        model = make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        losses = []
        for _ in range(300):
            loss, correct = train_step(model, optimizer, x, y)
            self.assertTrue(0 <= correct <= 32)
            self.assertTrue(torch.isfinite(torch.tensor(loss)))
            losses.append(loss)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertLess(losses[-1], losses[0])
        self.assertGreaterEqual(evaluate(model, x, y)["accuracy"], 0.95)

    def test_evaluate_partial_batch_and_no_grad(self):
        from lab0.train_toy import make_model, evaluate
        torch = self.torch
        torch.manual_seed(17)
        model = make_model()
        x, y = torch.randn(67, 16), torch.randint(2, (67,))
        with torch.no_grad():
            logits = model(x)
            expected_loss = torch.nn.functional.cross_entropy(logits, y).item()
            expected_acc = (logits.argmax(-1) == y).float().mean().item()
        flags = []
        handle = model.register_forward_hook(lambda *args: flags.append(torch.is_grad_enabled()))
        result = evaluate(model, x, y)
        handle.remove()
        self.assertEqual(result["num_samples"], 67)
        self.assertAlmostEqual(result["loss"], expected_loss, places=6)
        self.assertAlmostEqual(result["accuracy"], expected_acc, places=6)
        self.assertTrue(flags)
        self.assertFalse(any(flags))
        self.assertFalse(model.training)


class Gradient(unittest.TestCase):
    def test_float64_gradient_and_input_unchanged(self):
        torch = require_torch()
        from lab0.check_grad import finite_difference
        for seed in (42, 73):
            torch.manual_seed(seed)
            x = torch.randn(2, 2, dtype=torch.float64)
            w = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)
            before = w.detach().clone()
            ((x @ w) ** 2).mean().backward()
            actual = finite_difference(x, w.detach())
            self.assertLess((actual - w.grad).abs().max().item(), 1e-5)
            self.assertTrue(torch.equal(w, before))


class Checkpoint(unittest.TestCase):
    def test_roundtrip_after_update(self):
        torch = require_torch()
        from lab0.train_toy import make_model, make_data, train_step, save_checkpoint, load_checkpoint
        model = make_model()
        x, y, _, _ = make_data()
        train_step(model, torch.optim.Adam(model.parameters(), lr=0.01), x[:32], y[:32])
        with workspace_temp() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, model, {"seed": 42})
            restored, config = load_checkpoint(path)
        self.assertEqual(config["seed"], 42)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x[:32]), restored(x[:32]), rtol=1e-6, atol=1e-6)

import unittest
from common.io import read_json
from lab3.tests import FIXTURE
from lab4.analyze import slo_metrics
from lab4.plan import build_matrix

DESIGN = dict(low=1.2, high=2.8, num_prompts=120, long_input=1024,
              heldout_input=1536, output_len=64, max_model_len=2048)


class SLO(unittest.TestCase):
    def test_failed_and_single_token_denominator(self):
        result = slo_metrics(read_json(FIXTURE), {"ttft_ms": 300, "tpot_ms": 200})
        self.assertEqual(result["num_satisfied"], 1)
        self.assertEqual(result["slo_attainment"], 0.25)
        self.assertEqual(result["goodput"], 0.5)

    def test_inclusive_thresholds(self):
        run = read_json(FIXTURE)
        result = slo_metrics(run, {"ttft_ms": 100, "tpot_ms": 100})
        self.assertEqual(result["num_satisfied"], 1)


class Matrix(unittest.TestCase):
    def test_complete_finite_matrix(self):
        rows = build_matrix(DESIGN)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["run_id"] for r in rows}), 12)
        expected = {(s, n, rate) for s in (11, 22, 33) for n in (128, 1024) for rate in (1.2, 2.8)}
        self.assertEqual({(r["seed"], r["input_len"], r["request_rate"]) for r in rows}, expected)
        for row in rows:
            self.assertEqual((row["output_len"], row["burstiness"], row["num_prompts"], row["budget"]),
                             (64, 1, 120, 2048))
        order = [(128, 1.2), (1024, 2.8), (128, 2.8), (1024, 1.2)]
        self.assertEqual([(r["input_len"], r["request_rate"]) for r in rows[:4]], order)
        self.assertEqual([(r["input_len"], r["request_rate"]) for r in rows[4:8]], order[::-1])


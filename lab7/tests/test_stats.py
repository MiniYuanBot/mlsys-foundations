"""lab7 tests (CPU only, standard library):  python -m unittest lab7.tests.test_stats -v"""
import json
import random
import statistics
import tempfile
import unittest
from pathlib import Path

from lab7.exp import prereg
from lab7.exp.run import parse_metric
from lab7.exp.stats_student import bootstrap_ci, decide, ratio_ci


class S1Bootstrap(unittest.TestCase):
    def test_constant_sample_has_zero_width(self):
        self.assertEqual(bootstrap_ci([3.0] * 7), (3.0, 3.0))

    def test_interval_contains_median_and_is_reproducible(self):
        xs = [10.2, 9.8, 10.5, 11.0, 9.9, 10.1, 10.3]
        lo, hi = bootstrap_ci(xs, seed=1)
        self.assertLessEqual(lo, statistics.median(xs))
        self.assertGreaterEqual(hi, statistics.median(xs))
        self.assertEqual((lo, hi), bootstrap_ci(xs, seed=1))
        self.assertTrue(min(xs) <= lo <= hi <= max(xs))

    def test_coverage_is_roughly_nominal(self):
        # The 95% CI of the median of N(0,1) samples (n = 15) should contain 0
        # in roughly 95% of experiments (the percentile bootstrap is a bit
        # optimistic for small n; accept 85-99%).
        rng = random.Random(7)
        hits = 0
        trials = 300
        for t in range(trials):
            xs = [rng.gauss(0, 1) for _ in range(15)]
            lo, hi = bootstrap_ci(xs, n_boot=2000, seed=t)
            hits += lo <= 0 <= hi
        self.assertTrue(0.85 <= hits / trials <= 0.99, f"coverage {hits / trials:.3f}")

    def test_other_statistics(self):
        lo, hi = bootstrap_ci([1, 2, 3, 4, 100], stat=max, n_boot=500)
        self.assertLessEqual(hi, 100)
        self.assertRaises(ValueError, bootstrap_ci, [])


class S2Ratio(unittest.TestCase):
    def test_point_estimate_is_ratio_of_medians(self):
        p, lo, hi = ratio_ci([10, 10, 10], [8, 8, 8])
        self.assertAlmostEqual(p, 0.8)
        self.assertAlmostEqual(lo, 0.8)
        self.assertAlmostEqual(hi, 0.8)

    def test_noisy_ratio_brackets_truth(self):
        rng = random.Random(3)
        base = [100 + rng.gauss(0, 3) for _ in range(9)]
        treat = [80 + rng.gauss(0, 3) for _ in range(9)]
        p, lo, hi = ratio_ci(base, treat, seed=2)
        self.assertTrue(lo < p < hi)
        self.assertTrue(lo < 0.8 < hi)
        self.assertLess(hi, 0.9)

    def test_rejects_empty(self):
        self.assertRaises(ValueError, ratio_ci, [], [1.0])


class S3Decide(unittest.TestCase):
    def test_lower_is_better(self):
        # want at least 10% lower: target ratio 0.9
        self.assertEqual(decide(0.70, 0.85, "lower", 0.10), "supported")
        self.assertEqual(decide(0.93, 1.05, "lower", 0.10), "refuted")
        self.assertEqual(decide(0.85, 0.95, "lower", 0.10), "inconclusive")

    def test_higher_is_better(self):
        # want at least 20% higher: target ratio 1.2
        self.assertEqual(decide(1.30, 1.50, "higher", 0.20), "supported")
        self.assertEqual(decide(0.90, 1.10, "higher", 0.20), "refuted")
        self.assertEqual(decide(1.10, 1.30, "higher", 0.20), "inconclusive")

    def test_bad_direction(self):
        self.assertRaises(ValueError, decide, 0.9, 1.0, "faster", 0.1)


class Provided(unittest.TestCase):
    def test_prereg_lock_detects_edits(self):
        spec = json.loads((Path(__file__).parents[1] / "examples" / "bucket_overlap.prereg.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.prereg.json"
            p.write_text(json.dumps(spec), encoding="utf-8")
            self.assertRaises(RuntimeError, prereg.check_locked, p)  # not locked yet
            prereg.lock(p)
            prereg.check_locked(p)
            spec["min_effect"] = 0.01  # moving the goal posts after locking
            p.write_text(json.dumps(spec), encoding="utf-8")
            self.assertRaises(RuntimeError, prereg.check_locked, p)
            self.assertRaises(RuntimeError, prereg.lock, p)

    def test_prereg_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text(json.dumps({"name": "x"}), encoding="utf-8")
            self.assertRaises(ValueError, prereg.load, p)

    def test_parse_metric_reads_last_row(self):
        out = "# comment\nmode,step_ms\nsync,23.5\n"
        self.assertEqual(parse_metric(out, "step_ms"), 23.5)
        self.assertRaises(ValueError, parse_metric, "no csv here\n", "step_ms")


if __name__ == "__main__":
    unittest.main()

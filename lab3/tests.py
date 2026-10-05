import copy
import unittest
from pathlib import Path
from common.io import read_json
from common.service import validate_run
from lab3.metrics import summarize

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.json"


class Metrics(unittest.TestCase):
    def test_hand_computed_fixture(self):
        result = summarize(read_json(FIXTURE))
        self.assertEqual(result["num_requested"], 4)
        self.assertEqual(result["num_success"], 3)
        self.assertEqual(result["num_failed"], 1)
        self.assertEqual(result["total_output_tokens"], 7)
        self.assertAlmostEqual(result["request_throughput"], 1.5)
        self.assertAlmostEqual(result["output_throughput"], 3.5)
        self.assertAlmostEqual(result["p50_ttft_ms"], 200)
        self.assertAlmostEqual(result["p95_ttft_ms"], 380)
        self.assertAlmostEqual(result["p50_tpot_ms"], 200)
        self.assertAlmostEqual(result["p95_tpot_ms"], 290)

    def test_all_failed(self):
        run = read_json(FIXTURE)
        run["requests"] = [dict(request_id="0", success=False, input_tokens=None,
                                output_tokens=0, ttft_s=None, tpot_s=None, error="timeout")]
        run["num_requested"] = 1
        result = summarize(run)
        self.assertEqual(result["request_throughput"], 0)
        self.assertEqual(result["output_throughput"], 0)
        self.assertIsNone(result["p95_ttft_ms"])
        self.assertIsNone(result["p95_tpot_ms"])

    def test_invalid_denominator_and_missing_requests(self):
        run = read_json(FIXTURE)
        for bad in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                summarize({**run, "duration_s": bad})
        with self.assertRaises(ValueError):
            summarize({**run, "num_requested": 3})
        with self.assertRaises(ValueError):
            validate_run(run, allow_synthetic=False)


"""Infrastructure checks, separate from deliberately unfinished student tests."""
import copy
from common.temporary import workspace_temp
import unittest
from pathlib import Path
from common.io import read_json
from common.service import validate_run, validate_design, percentile
from lab3.commands import benchmark_command, capacity_matrix, generate
from tools.grade import assess_manual
from lab3.tests import FIXTURE


class Infrastructure(unittest.TestCase):
    def test_quantile_and_schema_guards(self):
        self.assertEqual(percentile([100, 200, 400], 0.95), 380)
        self.assertIsNone(percentile([], 0.95))
        run = read_json(FIXTURE)
        validate_run(run)
        with self.assertRaises(ValueError):
            validate_run(run, allow_synthetic=False)
        bad = copy.deepcopy(run)
        bad["requests"][2]["tpot_s"] = 0
        with self.assertRaises(ValueError):
            validate_run(bad)
        bad = copy.deepcopy(run)
        bad["requests"][1]["request_id"] = "0"
        with self.assertRaises(ValueError):
            validate_run(bad)

    def test_manual_credit_requires_evidence_and_review(self):
        criterion = dict(id="evidence", points=30)
        with workspace_temp() as tmp:
            root = Path(tmp)
            (root / "report.md").write_text("Example evidence", encoding="utf-8")
            entry = dict(status="reviewed", score=30, evidence=["report.md"], note="Checked each rubric item")
            self.assertEqual(assess_manual(criterion, entry, root)["earned"], 30)
            self.assertEqual(assess_manual(criterion, {**entry, "status": "pending"}, root)["earned"], 0)
            self.assertEqual(assess_manual(criterion, {**entry, "evidence": ["missing.md"]}, root)["earned"], 0)
            self.assertEqual(assess_manual(criterion, {**entry, "evidence": ["../elsewhere"]}, root)["earned"], 0)
            with self.assertRaises(ValueError):
                assess_manual(criterion, {**entry, "score": 100}, root)

    def test_generated_commands_are_pinned_and_nonexecuting(self):
        config = read_json(Path(__file__).resolve().parents[1] / "lab3/service.example.json")
        config.update(model_path="/models/fixed-snapshot", tokenizer_path="/models/fixed-snapshot",
                      model_revision="a" * 40, tokenizer_revision="a" * 40)
        rows = capacity_matrix(config)
        self.assertEqual([r["num_prompts"] for r in rows], [8, 20, 120, 120, 120])
        with workspace_temp() as tmp:
            out = Path(tmp) / "plan"
            generate(config, rows, out)
            text = (out / "commands.txt").read_text(encoding="utf-8")
            self.assertIn("--no-enable-prefix-caching", text)
            self.assertIn("--ignore-eos", text)
            self.assertNotIn("--max-concurrency", text)
            self.assertIn("--tokenizer /models/fixed-snapshot", text)
            self.assertFalse((out / "runs").exists())
            with self.assertRaises(FileExistsError):
                generate(config, rows, out)
        bad = dict(rows[0], request_rate=float("inf"))
        with self.assertRaises(ValueError):
            benchmark_command(config, bad, Path("unused"))

    def test_finite_design_context(self):
        good = dict(low=1.2, high=2.8, num_prompts=120, long_input=1024,
                    heldout_input=1536, output_len=64, max_model_len=2048)
        validate_design(good)
        for patch in ({"high": float("inf")}, {"heldout_input": 2048}, {"long_input": 128}):
            with self.assertRaises(ValueError):
                validate_design({**good, **patch})

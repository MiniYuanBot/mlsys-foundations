import unittest
from common.io import write_json
from common.temporary import workspace_temp
from pathlib import Path
from tools.grade import grade
from tools.run_tests import run


class Grading(unittest.TestCase):
    def test_base_evidence_does_not_complete_research_tier(self):
        # An unrelated base assessment must never flow into advanced grades.
        with workspace_temp() as directory:
            evidence = Path(directory) / "base-assessment.json"
            write_json(evidence, {"analysis": {"score": 100, "status": "reviewed"}})
            report = grade("lab6", evidence, tier="research")
        self.assertEqual(report["tier"], "research")
        self.assertEqual(report["score"], 0)
        self.assertEqual(report["status"], "incomplete")

    def test_invalid_tier_cannot_fall_back_to_base(self):
        with self.assertRaises(ValueError):
            grade("lab0", tier="sytems")
        with self.assertRaises(ValueError):
            grade("lab2x", tier="systems")

    def test_no_false_pass_for_skips_empty_or_todo(self):
        for fixture, expected in [
            ("Pass", "passed"), ("Environment", "environment-blocked"),
            ("Unimplemented", "failed"), ("Skipped", "incomplete"), ("Empty", "incomplete")
        ]:
            with self.subTest(fixture=fixture):
                report = run("tests.grader_fixtures." + fixture)
                self.assertEqual(report["status"], expected)

import unittest
from lab4.tests import DESIGN
from lab6.plan import build_matrix


class Matrix(unittest.TestCase):
    def test_ab_blocks_and_only_budget_changes(self):
        rows = build_matrix(DESIGN)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["run_id"] for r in rows}), 12)
        self.assertEqual([(r["seed"], r["budget"]) for r in rows],
                         [(s, b) for s, budgets in [(11, [2048, 512]), (22, [512, 2048]), (33, [2048, 512])]
                          for b in budgets for _ in range(2)])
        self.assertEqual([r["request_rate"] for r in rows], [1.2, 2.8] * 6)
        for r in rows:
            self.assertEqual((r["input_len"], r["output_len"], r["burstiness"], r["num_prompts"]),
                             (1024, 64, 1, 120))


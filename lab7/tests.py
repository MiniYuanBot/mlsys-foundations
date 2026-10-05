import unittest
from lab4.tests import DESIGN
from lab6.plan import build_matrix as previous
from lab7.plan import build_matrix


class Holdout(unittest.TestCase):
    def test_only_length_and_id_change(self):
        for context, old, new in [(2048, 1024, 1536), (1536, 1024, 1408), (1024, 768, 896)]:
            design = dict(DESIGN, max_model_len=context, long_input=old, heldout_input=new)
            base, rows = previous(design), build_matrix(design)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len({r["run_id"] for r in rows}), 12)
            self.assertFalse({r["run_id"] for r in rows} & {r["run_id"] for r in base})
            for a, b in zip(base, rows):
                self.assertEqual(b["input_len"], new)
                self.assertLessEqual(b["input_len"] + b["output_len"], context)
                self.assertEqual({k: v for k, v in a.items() if k not in ("run_id", "input_len")},
                                 {k: v for k, v in b.items() if k not in ("run_id", "input_len")})

    def test_invalid_context_rejected(self):
        with self.assertRaises(ValueError):
            build_matrix(dict(DESIGN, heldout_input=2048))


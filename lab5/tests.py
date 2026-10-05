import unittest
from lab5.tail_sim import generate_lengths, simulate_barrier, simulate_fcfs, summarize


class Schedulers(unittest.TestCase):
    def test_hand_computed(self):
        self.assertEqual(simulate_fcfs([1, 4, 1, 4], 2, 1), [
            dict(task_id=0, slot=0, length=1, start=0, finish=1),
            dict(task_id=1, slot=1, length=4, start=0, finish=4),
            dict(task_id=2, slot=0, length=1, start=1, finish=2),
            dict(task_id=3, slot=0, length=4, start=2, finish=6)])
        self.assertEqual(summarize(simulate_barrier([1, 4, 1, 4], 2, 1), 2, 1)["makespan"], 8)
        self.assertEqual(summarize(simulate_fcfs([1, 4, 1, 4], 2, 1), 2, 1)["makespan"], 6)

    def test_single_slot_uniform_and_partial_batch(self):
        for fn in (simulate_barrier, simulate_fcfs):
            self.assertEqual(summarize(fn([1, 4, 1, 4], 1, 1), 1, 1)["makespan"], 10)
            for K in (4, 8):
                metrics = summarize(fn([128] * 128, K, 64), K, 64)
                self.assertEqual(metrics["makespan"], 256 / K)
                self.assertEqual(metrics["utilization"], 1)
            rows = fn([3, 1, 2], 8, 1)
            self.assertEqual(len(rows), 3)
            self.assertEqual(summarize(rows, 8, 1)["makespan"], 3)


class Conservation(unittest.TestCase):
    def test_all_inputs_work_and_nonoverlap(self):
        for seed in (11, 22, 33):
            for kind in ("uniform", "long-tail"):
                lengths = generate_lengths(seed, kind)
                self.assertEqual(lengths, generate_lengths(seed, kind))
                for K in (4, 8):
                    for fn in (simulate_barrier, simulate_fcfs):
                        rows = fn(lengths, K, 64)
                        self.assertEqual([r["task_id"] for r in rows], list(range(128)))
                        self.assertEqual([r["length"] for r in rows], lengths)
                        last = [0] * K
                        for row in rows:
                            self.assertTrue(0 <= row["slot"] < K)
                            self.assertGreaterEqual(row["start"], last[row["slot"]])
                            self.assertAlmostEqual(row["finish"] - row["start"], row["length"] / 64)
                            last[row["slot"]] = row["finish"]
                        metrics = summarize(rows, K, 64)
                        self.assertGreaterEqual(metrics["makespan"], max(sum(lengths) / (64 * K), max(lengths) / 64))
                        self.assertTrue(0 < metrics["utilization"] <= 1)
                        self.assertEqual(metrics["total_tokens"], sum(lengths))
                        self.assertEqual(metrics["num_tasks"], 128)
                        self.assertAlmostEqual(metrics["mean_completion_time"], sum(r["finish"] for r in rows) / 128)


"""Real subprocess tests. Expected to fail until student.py is implemented."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from common.temporary import workspace_temp

ROOT = Path(__file__).resolve().parents[2]


def run_bounded(command, cwd, timeout=75):
    """Bound both parent and worker lifetime; preserve useful failure output."""
    proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=(os.name != "nt"))
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10, check=False)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.kill()
        output, _ = proc.communicate(timeout=10)
        raise AssertionError(f"distributed run timed out after {timeout}s\n{output}") from None
    if proc.returncode != 0:
        raise AssertionError(f"distributed workers exited {proc.returncode}\n{output}")
    return output


class RealCollectives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("ENV: install CPU PyTorch yourself; no points awarded")

    def check_case(self, world_size, samples, steps=5):
        with workspace_temp() as directory:
            run_bounded([sys.executable, "-m", "torch.distributed.run", "--standalone",
                         f"--nproc_per_node={world_size}", "--max_restarts=0", "--module",
                         "lab5.distributed.train", "--out", directory,
                         "--samples", str(samples), "--steps", str(steps)], ROOT)
            outputs = [json.loads((Path(directory) / f"rank{r}.json").read_text())
                       for r in range(world_size)]
            oracle = json.loads((Path(directory) / "oracle.json").read_text())
            self.assertEqual(len({out["pid"] for out in outputs}), world_size)
            self.assertEqual(sum(out["local_samples"] for out in outputs), samples)
            self.assertEqual(sorted(i for out in outputs for i in out["indices"]), list(range(samples)))
            for rank, out in enumerate(outputs):
                self.assertEqual(out["backend"], "gloo")
                self.assertEqual(out["rank"], rank)
                self.assertEqual(len(out["history"]), steps)
                for step in range(steps):
                    actual = out["history"][step]["parameters"]
                    expected = oracle["history"][step]["parameters"]
                    for value, target in zip(actual, expected, strict=True):
                        self.assertAlmostEqual(value, target, delta=1e-10)
                    for value, target in zip(actual, outputs[0]["history"][step]["parameters"], strict=True):
                        self.assertAlmostEqual(value, target, delta=1e-12)
            if world_size > 1:
                self.assertNotEqual(outputs[0]["history"][0]["gradients_before_sync"],
                                    outputs[1]["history"][0]["gradients_before_sync"])

    def test_world_size_one(self):
        self.check_case(1, 8)

    def test_two_ranks_equal_shards(self):
        self.check_case(2, 8)

    def test_two_ranks_uneven_shards(self):
        self.check_case(2, 9)


if __name__ == "__main__":
    unittest.main()

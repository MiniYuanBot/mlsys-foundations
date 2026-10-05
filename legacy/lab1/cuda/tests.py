"""Bridge the repository grader to native CUDA tests without importing PyTorch."""

import os
from pathlib import Path
import subprocess
import unittest


class NativeGPU(unittest.TestCase):
    def test_native_gemm(self):
        repo_root = Path(__file__).resolve().parents[2]
        override = os.environ.get("MLSYS_CUDA_EXE")
        if override:
            executable = Path(override).expanduser()
            if not executable.is_absolute():
                executable = repo_root / executable
        else:
            filename = "gemm_lab.exe" if os.name == "nt" else "gemm_lab"
            executable = repo_root / "build" / "lab1-cuda" / filename
        if not executable.is_file():
            self.skipTest(
                "ENV: build lab1/cuda with CMake into build/lab1-cuda, or set "
                f"MLSYS_CUDA_EXE to the compiled gemm_lab executable; missing {executable}"
            )
        try:
            result = subprocess.run(
                [str(executable), "--test"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self.fail(f"Native CUDA test could not complete: {error}")
        output = result.stdout + "\n" + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("# STUDENT_CORRECTNESS_PASSED", result.stdout, output)


if __name__ == "__main__":
    unittest.main()

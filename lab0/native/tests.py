import os
from pathlib import Path
import subprocess
import unittest
from common.records import ROOT


class Native(unittest.TestCase):
    def test_compiled_student_operator(self):
        default = ROOT / "build" / "lab0-native" / ("tensor_lab.exe" if os.name == "nt" else "tensor_lab")
        executable = Path(os.environ.get("MLSYS_NATIVE_EXE", default))
        if not executable.is_file():
            raise unittest.SkipTest("ENV: build lab0/native first; see its setup.md")
        result = subprocess.run([str(executable.resolve()), "--test"], capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("student contract passed", result.stdout)


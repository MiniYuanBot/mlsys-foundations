"""Repository self-checks: lab layout, Python syntax and Markdown links.

Run from the repository root:  python -m unittest discover -s labs/tests -v
"""

import ast
import os
import re
import unittest
from pathlib import Path
from urllib.parse import unquote

LABS_ROOT = Path(__file__).resolve().parents[1]
ROOT = LABS_ROOT.parent
COURSE = ROOT / "course"
SOLUTIONS = ROOT / "solutions"
LABS = ["lab0", "lab1", "lab2", "lab3", "lab4", "lab5", "lab6", "lab7"]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SOURCE_SUFFIXES = {".py", ".cpp", ".cu", ".cuh", ".hpp", ".rs"}
SKIP = {
    ".git",
    ".venv",
    ".venv-format",
    "venv",
    "env",
    "__pycache__",
    "build",
    "results",
    "artifacts",
    "external",
    "target",
    "node_modules",
}


def source_files(suffix):
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [
            name
            for name in dirs
            if name not in SKIP and not name.startswith((".venv", "build-", "cmake-build-"))
        ]
        for name in files:
            if name.endswith(suffix):
                yield Path(folder) / name


class Structure(unittest.TestCase):
    def test_foundations_layout(self):
        for folder in (COURSE, LABS_ROOT, SOLUTIONS):
            with self.subTest(folder=folder.name):
                self.assertTrue((folder / "README.md").is_file())
                self.assertFalse((folder / ".git").exists(), "Nested Git repository")
        self.assertFalse((LABS_ROOT / "solutions").exists())

    def test_each_lab_has_readme_grade_and_build(self):
        for lab in LABS:
            with self.subTest(lab=lab):
                folder = LABS_ROOT / lab
                for name in ("README.md", "grade.md"):
                    self.assertTrue((folder / name).is_file(), f"{lab}/{name}")
                builds = [
                    folder / "CMakeLists.txt",
                    folder / "Cargo.toml",
                    folder / "tests" / "__init__.py",
                ]
                self.assertTrue(
                    any(p.is_file() for p in builds),
                    f"{lab} has no CMake, Cargo or Python test entry point",
                )

    def test_student_files_contain_no_reference_solution_markers(self):
        for lab in LABS:
            for path in (LABS_ROOT / lab).rglob("*_student.*"):
                if path.suffix not in SOURCE_SUFFIXES or "__pycache__" in path.parts:
                    continue
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=str(path.relative_to(ROOT))):
                    self.assertNotIn("BEGIN SOLUTION", text)
                    self.assertNotIn("STUB ", text)

    def test_reference_solutions_match_student_files(self):
        students = {
            path.relative_to(LABS_ROOT)
            for lab in LABS
            for path in (LABS_ROOT / lab).rglob("*_student.*")
            if path.suffix in SOURCE_SUFFIXES
        }
        references = {
            path.relative_to(SOLUTIONS)
            for path in SOLUTIONS.rglob("*_student.*")
            if path.suffix in SOURCE_SUFFIXES
        }
        self.assertTrue(students, "No student files found")
        self.assertSetEqual(students, references)

    def test_python_syntax(self):
        for path in source_files(".py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_markdown_links(self):
        """Check all local links, including links between the three directories."""
        for path in source_files(".md"):
            text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
            for target in LINK.findall(text):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                destination = (path.parent / unquote(target.split("#")[0])).resolve()
                with self.subTest(source=str(path.relative_to(ROOT)), target=target):
                    self.assertTrue(
                        destination.is_relative_to(ROOT),
                        f"Link leaves the repository: {path.relative_to(ROOT)} -> {target}",
                    )
                    self.assertTrue(destination.exists(), f"{path.relative_to(ROOT)} -> {target}")


if __name__ == "__main__":
    unittest.main()

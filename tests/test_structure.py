"""Repository self-checks: lab layout, Python syntax and Markdown links.

Run from the repository root:  python -m unittest discover -s tests -v
"""

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT.parent / "mlsys-research-course"
LABS = ["lab0", "lab1", "lab2", "lab3", "lab4", "lab5", "lab6", "lab7"]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def markdown_files():
    skip = {".git", "build", "results", "external"}
    for path in ROOT.rglob("*.md"):
        if not skip.intersection(path.relative_to(ROOT).parts):
            yield path


class Structure(unittest.TestCase):
    def test_each_lab_has_readme_grade_and_build(self):
        for lab in LABS:
            with self.subTest(lab=lab):
                folder = ROOT / lab
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
            for path in (ROOT / lab).rglob("*_student.*"):
                if (
                    path.suffix not in {".py", ".cpp", ".cu", ".cuh", ".hpp", ".rs"}
                    or "__pycache__" in path.parts
                ):
                    continue
                text = path.read_text(encoding="utf-8")
                with self.subTest(path=str(path.relative_to(ROOT))):
                    self.assertNotIn("BEGIN SOLUTION", text)
                    self.assertNotIn("STUB ", text)

    def test_python_syntax(self):
        for folder in ["common", "tools", "tests", *LABS]:
            for path in (ROOT / folder).rglob("*.py"):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_markdown_links(self):
        """Repository-local links must resolve. Links into the sibling course
        repository are checked too when it is checked out next to this one,
        and the test says so loudly when it is not."""
        if not COURSE.is_dir():
            print(f"\nNOTE: {COURSE} not found; cross-repository links were not checked")
        for path in markdown_files():
            text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
            for target in LINK.findall(text):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                destination = (path.parent / target.split("#")[0]).resolve()
                if destination.is_relative_to(ROOT) or (
                    COURSE.is_dir() and destination.is_relative_to(COURSE.resolve())
                ):
                    with self.subTest(source=str(path.relative_to(ROOT)), target=target):
                        self.assertTrue(
                            destination.exists(), f"{path.relative_to(ROOT)} -> {target}"
                        )


if __name__ == "__main__":
    unittest.main()

"""Audits rubrics, chapter assets, syntax and repository-local Markdown links."""
import ast
import re
import unittest
from pathlib import Path
from common.io import read_json

ROOT = Path(__file__).resolve().parents[1]
LABS = ["lab0", "lab1", "lab2", "lab2x", "lab3", "lab4", "lab5", "lab6", "lab7"]


class Structure(unittest.TestCase):
    def test_advanced_tiers_have_independent_rubrics_and_entrypoints(self):
        for lab in [name for name in LABS if name != "lab2x"]:
            folder = ROOT / lab
            for tier in ("systems", "research"):
                with self.subTest(lab=lab, tier=tier):
                    rubric = read_json(folder / f"{tier}.rubric.json")
                    self.assertEqual(rubric["lab"], lab)
                    self.assertEqual(rubric["tier"], tier)
                    self.assertEqual(rubric["pass_threshold"], 80)
                    self.assertTrue((folder / f"{tier}.md").is_file())
                    criteria = rubric["criteria"]
                    self.assertEqual(sum(c["points"] for c in criteria), 100)
                    self.assertEqual(len({c["id"] for c in criteria}), len(criteria))
                    sample = read_json(folder / f"{tier}.assessment.example.json")
                    self.assertEqual(set(sample), {c["id"] for c in criteria if c["kind"] == "manual"})
                    self.assertTrue(any(c.get("required") for c in criteria))
                    for c in criteria:
                        self.assertGreater(c["points"], 0)
                        self.assertLessEqual(c.get("minimum", c["points"]), c["points"])
                        if c["kind"] == "auto":
                            module, name = c["suite"].rsplit(".", 1)
                            path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
                            tree = ast.parse(path.read_text(encoding="utf-8"))
                            self.assertIn(name, [n.name for n in tree.body if isinstance(n, ast.ClassDef)])

    def test_every_lab_has_consistent_100_point_rubric(self):
        for lab in LABS:
            with self.subTest(lab=lab):
                folder = ROOT / lab
                for filename in ("README.md", "assignment.md", "setup.md", "grade.md", "tests.py", "report.md", "rubric.json", "assessment.example.json"):
                    self.assertTrue((folder / filename).is_file(), f"{lab}/{filename}")
                rubric = read_json(folder / "rubric.json")
                self.assertEqual(sum(c["points"] for c in rubric["criteria"]), 100)
                self.assertEqual(len({c["id"] for c in rubric["criteria"]}), len(rubric["criteria"]))
                sample = read_json(folder / "assessment.example.json")
                self.assertEqual(set(sample), {c["id"] for c in rubric["criteria"] if c["kind"] == "manual"})
                for criterion in rubric["criteria"]:
                    self.assertIn(criterion["id"], (folder / "grade.md").read_text(encoding="utf-8"))
                    if criterion["kind"] == "auto":
                        module, name = criterion["suite"].rsplit(".", 1)
                        module_path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
                        tree = ast.parse(module_path.read_text(encoding="utf-8"))
                        self.assertIn(name, [node.name for node in tree.body if isinstance(node, ast.ClassDef)])

    def test_internal_markdown_links(self):
        course = ROOT.parent / "mlsys-research-course"
        scopes = [ROOT, *(ROOT / name for name in ("docs", "templates", "portfolio", "common", "tools", "tests")), *(ROOT / lab for lab in LABS)]
        paths = list(ROOT.glob("*.md"))
        for folder in scopes[1:]:
            paths.extend(folder.rglob("*.md"))
        if course.is_dir():
            paths.extend(course.glob("*.md"))
            for folder in ("guides", "readings", "labs", "modules", "templates"):
                paths.extend((course / folder).rglob("*.md"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            # Remove code fences: examples are not navigation links.
            text = re.sub(r"```.*?```", "", text, flags=re.S)
            for target in re.findall(r"\[[^\]]+\]\(([^)\s]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                destination = (path.parent / target.split("#")[0]).resolve()
                # A separately distributed labs checkout may lack the course.
                if destination.is_relative_to(ROOT) or (course.is_dir() and destination.is_relative_to(course)):
                    self.assertTrue(destination.exists(), f"{path} -> {target}")

    def test_python_syntax_without_importing_gpu_dependencies(self):
        for folder in ["common", "tools", "tests", *LABS]:
            for path in (ROOT / folder).rglob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

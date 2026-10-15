#!/usr/bin/env python3
# Checks that every git-tracked file matches the pinned formatters.
#   python3 tools/format_check.py
# Changes nothing. Exit code 0 iff every group is formatted.
# Run tools/setup-format.sh (or .ps1) first; the tools are looked up in .venv-format.
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENV_BIN = REPO / ".venv-format" / ("Scripts" if os.name == "nt" else "bin")
CHUNK = 100  # file names per command, to stay under the OS command-line limit

# Same exclusions as .clang-format-ignore, .prettierignore and pyproject.toml.
EXCLUDED_DIRS = (
    "build",
    "build-*",
    "cmake-build-*",
    "CMakeFiles",
    "out",
    "dist",
    "target",
    "_deps",
    "external",
    "third_party",
    "third-party",
    "vendor",
    "node_modules",
    ".venv*",
    "venv",
    "env",
    ".tox",
    ".nox",
    "__pycache__",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "results",
    "artifacts",
    "generated",
)
EXCLUDED_FILES = ("*.generated.*", "*.min.*", "package-lock.json", "Cargo.lock")

# (group name, file-name suffixes, command; the files are appended to the command)
GROUPS = [
    (
        "C/C++/CUDA",
        (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".cu", ".cuh"),
        ["clang-format", "--dry-run", "--Werror"],
    ),
    ("Python", (".py",), ["ruff", "format", "--check"]),
    ("CMake", ("CMakeLists.txt", ".cmake", ".cmake.in"), ["gersemi", "--check"]),
    (
        "Markdown/JSON/YAML",
        (".md", ".json", ".jsonc", ".yaml", ".yml"),
        ["prettier", "--check"],
    ),
    ("Shell", (".sh", ".bash", ".sbatch"), ["shfmt", "--apply-ignore", "-d"]),
    ("Rust", (".rs",), ["rustfmt", "--edition", "2021", "--check"]),
]


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout
    return [f for f in out.decode("utf-8").split("\0") if f]


def excluded(path: str) -> bool:
    *dirs, name = path.split("/")
    if any(fnmatch(d, pat) for d in dirs for pat in EXCLUDED_DIRS):
        return True
    return any(fnmatch(name, pat) for pat in EXCLUDED_FILES)


def resolve(tool: str) -> str | None:
    search = os.pathsep.join([str(VENV_BIN), os.environ.get("PATH", "")])
    return shutil.which(tool, path=search)


def run_group(targets: list[str], cmd: list[str]) -> str | None:
    """Return None if the group is formatted, otherwise a short problem description."""
    exe = resolve(cmd[0])
    if exe is None:
        return f"{cmd[0]} not found; run tools/setup-format.sh (or .ps1)"
    problems = []
    for i in range(0, len(targets), CHUNK):
        chunk = targets[i : i + CHUNK]
        result = subprocess.run(
            [exe, *cmd[1:], *chunk],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            problems.append((result.stdout + result.stderr).strip())
    if not problems:
        return None
    return "\n".join(problems)


def main() -> int:
    files = tracked_files()
    failed = 0
    for name, suffixes, cmd in GROUPS:
        targets = [f for f in files if f.endswith(suffixes) and not excluded(f)]
        if not targets:
            print(f"  [SKIP]  {name:<18} no tracked files")
            continue
        problem = run_group(targets, cmd)
        if problem is None:
            print(f"  [ OK ]  {name:<18} {len(targets)} files")
            continue
        failed += 1
        print(f"  [FAIL]  {name:<18} {len(targets)} files")
        lines = problem.splitlines()
        for line in lines[:40]:
            print(f"          {line}")
        if len(lines) > 40:
            print(f"          ... {len(lines) - 40} more lines")
    if failed:
        print(f"\n{failed} group(s) not formatted. Run the formatter on the listed files.")
        return 1
    print("\nall tracked files match the pinned formatters")
    return 0


if __name__ == "__main__":
    sys.exit(main())

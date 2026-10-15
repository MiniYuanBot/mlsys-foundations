"""Temporary files stay under this project, including in restricted desktop sandboxes."""

from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def workspace_temp():
    root = Path(__file__).resolve().parents[1]
    base = (root / "results" / ".tmp").resolve()
    if not base.is_relative_to(root):
        raise ValueError("Temporary directory escaped the workspace")
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as name:
        if not Path(name).resolve().is_relative_to(base):
            raise ValueError("Temporary path escaped its verified parent")
        yield name

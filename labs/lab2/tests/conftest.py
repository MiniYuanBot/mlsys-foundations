import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # labs/ directory, so `import lab2` works


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="needs a CUDA GPU (run inside WSL with the mlsys venv)")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)

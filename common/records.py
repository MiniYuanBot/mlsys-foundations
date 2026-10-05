"""Run provenance and a tee log; refuses to overwrite a previous run."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from common.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def environment():
    packages = {}
    for name in ("torch", "triton", "vllm", "numpy", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                            stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unversioned", None
    try:
        smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT,
                                      text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        smi = "unavailable: " + str(exc)
    return {"date_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "python_executable": sys.executable, "os": platform.platform(),
            "packages": packages, "code_commit": commit, "code_dirty": dirty,
            "nvidia_smi": smi}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def run_directory(path, config):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    env = environment()
    write_json(path / "config.json", config)
    write_json(path / "environment.json", env)
    (path / "environment.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in env.items()) + "\n", encoding="utf-8")
    # argv.json is authoritative across shells; command.txt is POSIX display syntax.
    write_json(path / "argv.json", [sys.executable, *sys.orig_argv[1:]])
    (path / "command.txt").write_text(shlex.join([sys.executable, *sys.orig_argv[1:]]) + "\n",
                                     encoding="utf-8")
    with (path / "stdout.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                yield path
            except Exception:
                import traceback
                traceback.print_exc()
                raise


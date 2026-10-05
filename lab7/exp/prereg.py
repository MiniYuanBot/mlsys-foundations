"""Pre-registration: write the hypothesis down BEFORE looking at data, and lock it. Provided.

A pre-registration is a JSON file (see lab7/examples/*.prereg.json):

  name            short id, used for the results directory
  question        one sentence
  hypothesis      what you predict, in words
  metric          CSV column to read from the command's output (last CSV row on stdout)
  direction       "lower" or "higher": which way the treatment should move the metric
  min_effect      smallest relative change you would call meaningful (0.10 = 10%)
  repeats         runs per (condition, arm); arms are interleaved in random order
  command         a format string; {placeholders} come from the condition and the arm
  arms            {"baseline": {...}, "treatment": {...}}  (extra arms = ablations / negative controls)
  dev             list of conditions you may look at while developing
  heldout         list of conditions you only run ONCE, after locking
  workdir         directory the command runs in (relative to the repository root)

Locking: `lock(path)` records the SHA-256 of the file. Held-out runs refuse to
start unless the file still has that hash, so the hypothesis, margin and
held-out set cannot be quietly adjusted after seeing results.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ["name", "question", "hypothesis", "metric", "direction", "min_effect", "repeats", "command", "arms", "dev", "heldout"]


def load(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if spec["direction"] not in ("lower", "higher"):
        raise ValueError("direction must be 'lower' or 'higher'")
    if not 0 < float(spec["min_effect"]) < 1:
        raise ValueError("min_effect must be a fraction in (0, 1)")
    if "baseline" not in spec["arms"] or "treatment" not in spec["arms"]:
        raise ValueError("arms must contain 'baseline' and 'treatment'")
    if int(spec["repeats"]) < 3:
        raise ValueError("use at least 3 repeats per arm")
    if not spec["heldout"]:
        raise ValueError("a pre-registration needs at least one held-out condition")
    overlap = [c for c in spec["heldout"] if c in spec["dev"]]
    if overlap:
        raise ValueError(f"held-out conditions also appear in dev: {overlap}")
    return spec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_path(path):
    return Path(str(path) + ".lock")


def lock(path):
    """Freeze the pre-registration (idempotent if unchanged)."""
    p, lp = Path(path), lock_path(path)
    load(p)
    if lp.exists():
        old = json.loads(lp.read_text(encoding="utf-8"))
        if old["sha256"] != digest(p):
            raise RuntimeError(f"{p} changed after it was locked at {old['locked_at']}: make a NEW pre-registration "
                               "(e.g. name-v2) instead of editing a locked one")
        return old
    rec = {"sha256": digest(p), "locked_at": datetime.now(timezone.utc).isoformat()}
    lp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def check_locked(path):
    lp = lock_path(path)
    if not lp.exists():
        raise RuntimeError(f"{path} is not locked: run `python -m lab7.exp.run {path} --lock` before held-out runs")
    rec = json.loads(lp.read_text(encoding="utf-8"))
    if rec["sha256"] != digest(path):
        raise RuntimeError(f"{path} was modified after locking ({rec['locked_at']}); held-out results would not be honest")
    return rec

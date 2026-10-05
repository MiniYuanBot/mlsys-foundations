"""Run a pre-registered experiment matrix. Provided.

    python -m lab7.exp.run lab7/examples/bucket_overlap.prereg.json --phase dev
    python -m lab7.exp.run lab7/examples/bucket_overlap.prereg.json --lock
    python -m lab7.exp.run lab7/examples/bucket_overlap.prereg.json --phase heldout
    python -m lab7.exp.report results/lab7/bucket_overlap

Every (condition, arm, repeat) runs the command once; the order of arms is
shuffled within each repeat (seeded), so slow drift -- a laptop GPU warming
up, a background job -- hits all arms equally instead of biasing one. Each
run's metric is read from the LAST CSV row the command prints (header = the
last line that contains the metric name). Failures are recorded, not dropped.
Output: results/lab7/<name>/<phase>/runs.csv plus provenance (config, environment,
command lines) via common.records.run_directory.
"""
import argparse
import csv
import json
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

from common.records import ROOT, run_directory
from lab7.exp import prereg


def parse_metric(stdout, metric):
    lines = [ln for ln in stdout.splitlines() if ln.strip() and not ln.startswith("#")]
    header_idx = max((i for i, ln in enumerate(lines) if metric in ln.split(",")), default=None)
    if header_idx is None or header_idx + 1 >= len(lines):
        raise ValueError(f"no CSV header with column {metric!r} followed by a data row")
    row = next(csv.DictReader([lines[header_idx], lines[-1]]))
    return float(row[metric])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prereg")
    ap.add_argument("--phase", choices=["dev", "heldout"])
    ap.add_argument("--lock", action="store_true", help="freeze the pre-registration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    spec = prereg.load(a.prereg)
    if a.lock:
        rec = prereg.lock(a.prereg)
        print(f"locked {a.prereg} sha256={rec['sha256'][:16]}... at {rec['locked_at']}")
        if not a.phase:
            return
    if not a.phase:
        ap.error("give --phase dev|heldout (or --lock)")
    if a.phase == "heldout":
        prereg.check_locked(a.prereg)
    conditions = spec[a.phase]
    out = ROOT / "results" / "lab7" / spec["name"] / a.phase
    if out.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = out.with_name(f"{a.phase}-{stamp}")
    workdir = ROOT / spec.get("workdir", ".")
    rng = random.Random(a.seed)
    config = {"prereg": spec, "phase": a.phase, "seed": a.seed, "prereg_sha256": prereg.digest(a.prereg)}
    with run_directory(out, config) as d:
        rows = []
        for ci, cond in enumerate(conditions):
            for rep in range(int(spec["repeats"])):
                arms = list(spec["arms"].items())
                rng.shuffle(arms)
                for arm, params in arms:
                    cmd = spec["command"].format(**{**cond, **params})
                    t0 = time.time()
                    try:
                        p = subprocess.run(shlex.split(cmd), cwd=workdir, capture_output=True, text=True, timeout=a.timeout)
                        value = parse_metric(p.stdout, spec["metric"]) if p.returncode == 0 else None
                        err = "" if p.returncode == 0 else f"exit {p.returncode}: {p.stderr.strip()[-200:]}"
                    except (subprocess.TimeoutExpired, ValueError, OSError) as e:
                        value, err = None, f"{type(e).__name__}: {e}"
                    row = {"condition": ci, "condition_json": json.dumps(cond, sort_keys=True), "arm": arm, "repeat": rep,
                           "value": "" if value is None else value, "seconds": round(time.time() - t0, 2), "error": err,
                           "command": cmd}
                    rows.append(row)
                    print(f"[{a.phase} c{ci} r{rep} {arm:>10}] {spec['metric']} = {row['value']} {err}", flush=True)
        with (d / "runs.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out / 'runs.csv'}")
    failed = sum(1 for r in rows if r["value"] == "")
    if failed:
        print(f"WARNING: {failed} of {len(rows)} runs failed; they are kept in runs.csv", file=sys.stderr)


if __name__ == "__main__":
    main()

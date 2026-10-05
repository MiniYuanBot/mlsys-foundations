"""Summarise a pre-registered experiment. Provided (uses your stats_student).

    python -m lab7.exp.report results/lab7/<name>            # every phase found
    python -m lab7.exp.report results/lab7/<name> --markdown  # paste into your report

For each phase and condition: median and 95% bootstrap CI of every arm, the
treatment / baseline ratio with its CI, and the verdict against the
pre-registered margin. Failed runs are counted and shown, never silently dropped.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from lab7.exp.stats_student import bootstrap_ci, decide, ratio_ci


def summarise(phase_dir):
    cfg = json.loads((phase_dir / "config.json").read_text(encoding="utf-8"))
    spec = cfg["prereg"]
    with (phase_dir / "runs.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by = defaultdict(lambda: defaultdict(list))
    failed = defaultdict(int)
    conds = {}
    for r in rows:
        c = int(r["condition"])
        conds[c] = json.loads(r["condition_json"])
        if r["value"] == "":
            failed[(c, r["arm"])] += 1
        else:
            by[c][r["arm"]].append(float(r["value"]))
    out = []
    for c in sorted(conds):
        arms = by[c]
        entry = {"condition": conds[c], "arms": {}, "verdicts": {}}
        for arm, xs in arms.items():
            lo, hi = bootstrap_ci(xs)
            entry["arms"][arm] = (statistics.median(xs), lo, hi, len(xs), failed[(c, arm)])
        base = arms.get("baseline", [])
        for arm, xs in arms.items():
            if arm == "baseline" or not base:
                continue
            point, lo, hi = ratio_ci(base, xs)
            v = decide(lo, hi, spec["direction"], float(spec["min_effect"])) if arm == "treatment" else "-"
            entry["verdicts"][arm] = (point, lo, hi, v)
        out.append(entry)
    return spec, cfg["phase"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    root = Path(a.result_dir)
    phases = sorted(p for p in root.iterdir() if (p / "runs.csv").exists())
    if not phases:
        raise SystemExit(f"no runs.csv under {root}")
    for pd in phases:
        spec, phase, out = summarise(pd)
        print(f"\n## {spec['name']} / {pd.name}  (metric {spec['metric']}, want {spec['direction']} by >= {float(spec['min_effect']):.0%})\n")
        if a.markdown:
            print("| condition | arm | n (failed) | median [95% CI] | ratio to baseline [95% CI] | verdict |")
            print("|---|---|---|---|---|---|")
        for e in out:
            cond = ", ".join(f"{k}={v}" for k, v in e["condition"].items())
            for arm, (med, lo, hi, n, nf) in sorted(e["arms"].items(), key=lambda kv: kv[0] != "baseline"):
                ratio = verdict = ""
                if arm in e["verdicts"]:
                    p, rl, rh, v = e["verdicts"][arm]
                    ratio, verdict = f"{p:.3f} [{rl:.3f}, {rh:.3f}]", v
                if a.markdown:
                    print(f"| {cond} | {arm} | {n} ({nf}) | {med:.3f} [{lo:.3f}, {hi:.3f}] | {ratio} | {verdict} |")
                else:
                    print(f"{cond:<30} {arm:<12} n={n:<3} failed={nf:<2} median={med:<10.3f} CI=[{lo:.3f}, {hi:.3f}]  {ratio:<28} {verdict}")


if __name__ == "__main__":
    main()

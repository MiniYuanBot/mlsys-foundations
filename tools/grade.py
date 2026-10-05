"""Transparent self-assessment: executable checks + explicitly self-reviewed evidence."""
import argparse
from pathlib import Path
import subprocess
import sys
from common.temporary import workspace_temp
from common.io import read_json, write_json
from common.records import ROOT


def assess_manual(criterion, entry, root=ROOT):
    score = entry.get("score", 0)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= criterion["points"]:
        raise ValueError(f"Invalid score for {criterion['id']}")
    evidence = entry.get("evidence", [])
    valid = bool(evidence) and all(
        isinstance(p, str) and (root / p).resolve().is_relative_to(root.resolve())
        and (root / p).is_file() and (root / p).stat().st_size > 0 for p in evidence)
    reviewed = entry.get("status") == "reviewed" and bool(entry.get("note", "").strip()) and valid
    return {"id": criterion["id"], "kind": "manual", "status": "self-reviewed" if reviewed else "pending",
            "earned": score if reviewed else 0, "possible": criterion["points"],
            "evidence": evidence, "note": entry.get("note", ""),
            "gate_met": reviewed and score >= criterion.get("minimum", criterion["points"])}


def grade(lab, assessment_path=None, tier="base"):
    if tier not in ("base", "systems", "research"):
        raise ValueError("tier must be base, systems or research")
    rubric_name = "rubric.json" if tier == "base" else tier + ".rubric.json"
    rubric_path = ROOT / lab / rubric_name
    if not rubric_path.is_file():
        raise ValueError(f"{lab} has no {tier} rubric; lab2x is an optional standalone systems exercise")
    rubric = read_json(rubric_path)
    assessment = read_json(assessment_path) if assessment_path else {}
    rows = []
    for criterion in rubric["criteria"]:
        if criterion["kind"] == "manual":
            rows.append(assess_manual(criterion, assessment.get(criterion["id"], {})))
            continue
        with workspace_temp() as tmp:
            output = Path(tmp) / "test.json"
            try:
                process = subprocess.run(
                    [sys.executable, "-m", "tools.run_tests", criterion["suite"], "--out", str(output)],
                    cwd=ROOT, capture_output=True, text=True, timeout=criterion.get("timeout_s", 180))
                report = read_json(output) if output.exists() else {
                    "status": "failed", "log": process.stdout + process.stderr}
            except subprocess.TimeoutExpired:
                report = {"status": "failed", "log": "Test timeout; no points awarded"}
        passed = report["status"] == "passed"
        rows.append({"id": criterion["id"], "kind": "auto", "status": report["status"],
                     "earned": criterion["points"] if passed else 0, "possible": criterion["points"],
                     "gate_met": passed, "test_report": report})
    score = sum(row["earned"] for row in rows)
    gates = all(row["gate_met"] for row, criterion in zip(rows, rubric["criteria"]) if criterion.get("required"))
    status = "self-assessed-pass" if score >= rubric.get("pass_threshold", 80) and gates else "incomplete"
    if any(row["status"] == "environment-blocked" for row in rows):
        status = "environment-blocked"
    return {"lab": lab, "tier": tier, "status": status, "score": score, "maximum": 100,
            "automatic_score": sum(r["earned"] for r in rows if r["kind"] == "auto"),
            "manual_score": sum(r["earned"] for r in rows if r["kind"] == "manual"),
            "note": "This score applies ONLY to the selected tier. Manual evidence is self-reviewed. Base pass is not course completion. No speedup required.",
            "criteria": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab", choices=["lab0", "lab1", "lab2", "lab2x", "lab3", "lab4", "lab5", "lab6", "lab7"])
    parser.add_argument("--assessment", help="Filled copy of labN/assessment.example.json")
    parser.add_argument("--tier", choices=["base", "systems", "research"], default="base")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = grade(args.lab, args.assessment, args.tier)
    suffix = "" if args.tier == "base" else "-" + args.tier
    out = args.out or f"results/grades/{args.lab}{suffix}.json"
    write_json(out, report)
    for row in report["criteria"]:
        print(f"{row['id']}: {row['earned']}/{row['possible']} {row['status']}")
    print(f"{report['tier']} | {report['status']}: {report['score']}/100 -> {out}")
    return 0 if report["status"] == "self-assessed-pass" else 2 if report["status"] == "environment-blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())

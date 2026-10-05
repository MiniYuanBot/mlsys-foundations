"""L04: SLO denominator is all issued requests."""
import argparse
from common.io import read_json, write_csv, write_json
from common.service import validate_run, finite, percentile
from lab3.metrics import summarize


def slo_metrics(run, slo):
    """Return num_satisfied, slo_attainment, goodput; thresholds ttft_ms,tpot_ms."""
    validate_run(run)
    for key in ("ttft_ms", "tpot_ms"):
        finite(slo[key], key, positive=True)
    # TODO L04-1: success AND both defined latencies <= thresholds.
    # Undefined TPOT does not satisfy the joint SLO. Failures stay in N.
    raise NotImplementedError("L04-1: implement joint SLO metrics")


def freeze_slo(pilot):
    validate_run(pilot, allow_synthetic=False)
    if any(not r["success"] or r["output_tokens"] != 64 for r in pilot["requests"]):
        raise ValueError("Fix the pilot before freezing SLO")
    return {"ttft_ms": 2000 * percentile([r["ttft_s"] for r in pilot["requests"]], 0.5),
            "tpot_ms": 2000 * percentile([r["tpot_s"] for r in pilot["requests"]], 0.5),
            "source_run": pilot["run_id"], "definition": "twice the pilot request-level medians"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--slo")
    parser.add_argument("--freeze-slo", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    runs = [read_json(p) for p in args.inputs]
    if args.freeze_slo:
        if len(runs) != 1:
            parser.error("freeze from exactly one pilot")
        write_json(args.out, freeze_slo(runs[0]))
    else:
        if not args.slo:
            parser.error("--slo required")
        if len({r["synthetic"] for r in runs}) > 1:
            raise ValueError("Never mix synthetic and measured runs")
        slo = read_json(args.slo)
        write_csv(args.out, [dict(synthetic=r["synthetic"], **summarize(r), **slo_metrics(r, slo)) for r in runs])


if __name__ == "__main__":
    main()


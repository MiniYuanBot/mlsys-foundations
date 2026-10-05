import argparse
import statistics
from common.io import read_json, write_json
from common.service import validate_run
from lab3.metrics import summarize


def calibrate(runs):
    if len(runs) != 3 or len({r["run_id"] for r in runs}) != 3:
        raise ValueError("Three distinct capacity runs required")
    summaries = []
    for run in runs:
        validate_run(run, allow_synthetic=False)
        if any(not r["success"] or r["output_tokens"] != 64 for r in run["requests"]):
            raise ValueError("Capacity baseline needs all-success fixed 64-token outputs")
        summaries.append(summarize(run))
    values = [s["request_throughput"] for s in summaries]
    return {"C": statistics.median(values), "unit": "req/s", "values": values,
            "min": min(values), "max": max(values), "source_runs": [r["run_id"] for r in runs],
            "note": "finite-batch capacity proxy; verify lengths and identical configs manually"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    write_json(args.out, calibrate([read_json(p) for p in args.inputs]))


if __name__ == "__main__":
    main()


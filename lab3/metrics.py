"""L03: metrics from normalized requests; units in output names are explicit."""
import argparse
from common.io import read_json, write_csv
from common.service import validate_run, percentile


def summarize(run):
    """Return run_id, num_requested/success/failed, duration_s, total_output_tokens,
    request_throughput, output_throughput, p50/p95_ttft_ms, p50/p95_tpot_ms.
    Count successful completed output tokens; ignore undefined TPOT in quantiles.
    """
    validate_run(run)
    # TODO L03-2: sums before divisions; convert seconds to ms exactly once.
    raise NotImplementedError("L03-2: implement request-level summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    runs = [read_json(p) for p in args.inputs]
    if len({r["synthetic"] for r in runs}) > 1:
        raise ValueError("Never mix synthetic and measured runs")
    write_csv(args.out, [dict(synthetic=r["synthetic"], **summarize(r)) for r in runs])


if __name__ == "__main__":
    main()


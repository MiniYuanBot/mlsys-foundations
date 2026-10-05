"""L03: adapt the installed vLLM schema only after inspecting its actual output."""
import argparse
from common.io import read_json, write_json
from common.service import validate_run


def normalize(raw, mapping):
    """Return common/service.py schema; mapping records fields, units, and version."""
    # TODO L03-1: inspect one smoke JSON and implement its exact mapping.
    # Keep every issued request. Do not zip only successful arrays with all-request arrays.
    # Derive TPOT only if tool timestamps/counts have a verified matching definition.
    # Missing request-level evidence is a blocker, not a reason to invent values.
    raise NotImplementedError("L03-1: implement and document your installed version's adapter")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    run = normalize(read_json(args.raw), read_json(args.mapping))
    validate_run(run, allow_synthetic=False)
    write_json(args.out, run)


if __name__ == "__main__":
    main()


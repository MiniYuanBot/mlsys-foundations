"""Simple reproducible figure starter; refine grouping/labels in your lab analysis."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--x", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--group", nargs="*", default=[])
    parser.add_argument("--xlabel", required=True)
    parser.add_argument("--ylabel", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups, devices = {}, set()
    for path in args.inputs:
        with open(path, encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("device"):
                    devices.add(row["device"])
                key = tuple(row[column] for column in args.group)
                groups.setdefault(key, []).append((float(row[args.x]), float(row[args.y])))
    if len(devices) > 1:
        raise ValueError("Put measurements from different devices into distinct figures")
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for key, values in groups.items():
        xs, ys = zip(*sorted(values))
        axis.scatter(xs, ys, label="/".join(key) if key else None, alpha=0.75)
    axis.set(xlabel=args.xlabel, ylabel=args.ylabel, title=args.title)
    axis.grid(alpha=0.2)
    if args.group:
        axis.legend()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()

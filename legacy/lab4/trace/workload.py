"""S04: deterministic arrival/prefix identity trace; no server requests are sent here."""
import argparse
import math
import random
from common.io import write_json


def make_trace(seed, count, rate, shared_fraction, input_len=128, output_len=64):
    """Tokens are symbolic IDs, NOT tokenizer vocabulary IDs. Adapter resolves them.
    Prefix identity is shared, suffix identity always unique; same arrivals for A/B.
    """
    if count < 1 or not math.isfinite(rate) or rate <= 0 or not 0 <= shared_fraction <= 1:
        raise ValueError("invalid workload")
    rng = random.Random(seed)
    arrival = 0.0
    rows = []
    for i in range(count):
        arrival += rng.expovariate(rate)
        shared = i < int(count * shared_fraction)
        rows.append({"request_id": f"r{i}", "arrival_s": arrival,
                     "prefix_key": "shared" if shared else f"private-{i}",
                     "suffix_key": f"suffix-{i}", "target_input_tokens": input_len,
                     "target_output_tokens": output_len})
    # Shuffle request identities, not arrival times, to avoid shared-only warm first half.
    identities = [(r["prefix_key"], r["suffix_key"]) for r in rows]
    rng.shuffle(identities)
    for row, (prefix, suffix) in zip(rows, identities):
        row.update(prefix_key=prefix, suffix_key=suffix)
    return {"seed": seed, "rate": rate, "shared_fraction": shared_fraction,
            "kind": "synthetic-arrival-trace-not-measurement", "requests": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[11,22,33], required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--shared", type=float, choices=[0,0.8], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    write_json(args.out, make_trace(args.seed,args.count,args.rate,args.shared))


if __name__ == "__main__":
    main()


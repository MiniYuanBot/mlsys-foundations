"""CPU simulation: independent equal-speed slots, NOT real GPU/RL training."""
import argparse
import random
from common.io import write_csv
from common.records import run_directory


def generate_lengths(seed, kind):
    rng = random.Random(seed)
    if kind == "uniform":
        return [128] * 128
    if kind == "long-tail":
        return [rng.randint(64, 128) if rng.random() < 0.8 else rng.randint(512, 1024)
                for _ in range(128)]
    raise ValueError("kind must be uniform or long-tail")


def validate(lengths, K, rate):
    if not lengths or any(length <= 0 for length in lengths) or K < 1 or rate <= 0:
        raise ValueError("positive lengths, slots and rate required")


def simulate_barrier(lengths, K, rate):
    """Supplied readable reference: finish is individual completion, not barrier release."""
    validate(lengths, K, rate)
    rows, clock = [], 0.0
    for offset in range(0, len(lengths), K):
        batch = lengths[offset:offset + K]
        for slot, length in enumerate(batch):
            rows.append(dict(task_id=offset + slot, slot=slot, length=length,
                             start=clock, finish=clock + length / rate))
        clock += max(batch) / rate
    return rows


def simulate_fcfs(lengths, K, rate):
    """Return same row schema, in task_id order. Tie-break by slot id."""
    validate(lengths, K, rate)
    # TODO L05-1: heap entries (available_time, slot_id); pop, assign, push.
    raise NotImplementedError("L05-1: implement work-conserving FCFS")


def summarize(rows, K, rate):
    """Return makespan, mean_completion_time, utilization, num_tasks, total_tokens."""
    # TODO L05-2: all arrivals t=0; useful busy time is sum(length)/rate.
    raise NotImplementedError("L05-2: implement simulation metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summaries, timelines, saved_lengths = [], [], []
    with run_directory(args.out, {"kind": "simulation", "model": "independent equal-speed slots",
                                  "rate": 64, "K": [4, 8], "seeds": [11, 22, 33]}) as out:
        for kind in ("uniform", "long-tail"):
            for seed in (11, 22, 33):
                lengths = generate_lengths(seed, kind)
                saved_lengths.extend(dict(kind=kind, seed=seed, task_id=i, length=n) for i, n in enumerate(lengths))
                for K in (4, 8):
                    for name, simulator in (("barrier", simulate_barrier), ("fcfs", simulate_fcfs)):
                        rows = simulator(lengths, K, 64)
                        tags = dict(kind=kind, seed=seed, K=K, strategy=name)
                        summaries.append({**tags, **summarize(rows, K, 64)})
                        timelines.extend({**tags, **row} for row in rows)
        write_csv(out / "lengths.csv", saved_lengths)
        write_csv(out / "timeline.csv", timelines)
        write_csv(out / "raw.csv", timelines)
        write_csv(out / "summary.csv", summaries)
        print("24 summary rows; inputs are paired across strategies")


if __name__ == "__main__":
    main()


"""Trace the allocator and its accounting. All results are CPU metadata only."""

import argparse
import json

from .allocator import CapacityError, PagedKV
from .reference import contiguous_reserved_slots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps({
        "evidence_kind": "cpu_metadata_accounting",
        "contiguous_two_private_reservations": contiguous_reserved_slots((5, 5), (12, 12)),
    }))
    if args.baseline_only:
        return 0
    kv = PagedKV(num_blocks=6, block_size=4)
    steps = (
        ("cache prefix", lambda: kv.allocate("prefix", (1, 2, 3, 4, 5))),
        ("fork a", lambda: kv.fork_prefix("prefix", "a")),
        ("fork b", lambda: kv.fork_prefix("prefix", "b")),
        ("release prefix owner", lambda: kv.release("prefix")),
        ("decode a; COW tail", lambda: kv.append("a", (6, 7))),
        ("decode b; own tail", lambda: kv.append("b", (8, 9, 10, 11))),
        ("release a", lambda: kv.release("a")),
        ("release b", lambda: kv.release("b")),
    )
    for label, operation in steps:
        try:
            operation()
            kv.audit()
        except (NotImplementedError, CapacityError) as exc:
            print(json.dumps({"step": label, "status": "incomplete_or_failed", "error": str(exc)}))
            return 1
        print(json.dumps({"step": label, "state": kv.snapshot(), "accounting": kv.accounting()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

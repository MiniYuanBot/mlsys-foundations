"""Run one unittest suite and preserve machine-readable outcomes."""
import argparse
import io
import unittest
from common.io import write_json


def run(suite_name):
    suite = unittest.defaultTestLoader.loadTestsFromName(suite_name)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    missing = any("ModuleNotFoundError" in detail for _, detail in result.errors)
    status = "passed"
    if result.failures or result.errors:
        status = "environment-blocked" if missing and not result.failures else "failed"
    elif result.skipped:
        status = "environment-blocked" if all(reason.startswith("ENV:") for _, reason in result.skipped) else "incomplete"
    elif result.testsRun == 0:
        status = "incomplete"
    return {"suite": suite_name, "status": status, "run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
            "log": log.getvalue()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run(args.suite)
    write_json(args.out, report)
    print(report["log"], end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


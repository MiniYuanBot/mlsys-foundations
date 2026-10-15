"""Execute a user-selected command with configuration, environment and combined output."""

import argparse
import subprocess
from common.io import read_json
from common.records import run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    with run_directory(args.out, read_json(args.config)) as out:
        from common.io import write_json

        write_json(out / "executed_argv.json", command)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
        write_json(out / "exit.json", {"returncode": code})
    return code


if __name__ == "__main__":
    raise SystemExit(main())

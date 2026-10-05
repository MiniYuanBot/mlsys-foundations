"""Generate reviewable Bash commands; never start servers, install or download anything."""
import argparse
import importlib
from pathlib import Path
import re
import shlex
from common.io import read_json, write_json, write_csv
from common.service import finite


def validate_config(config):
    for field in ("model_path", "tokenizer_path", "model_revision", "tokenizer_revision"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip() or "TODO" in value:
            raise ValueError(f"Fill {field}; pin downloaded immutable snapshots before generating commands")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", config["model_revision"]) or not re.fullmatch(
            r"[0-9a-fA-F]{40}", config["tokenizer_revision"]):
        raise ValueError("Record full 40-character model and tokenizer commit hashes")
    if config["host"] != "127.0.0.1":
        raise ValueError("Course commands use localhost")
    if not (config["enforce_eager"] and config["chunked_prefill"] and not config["prefix_caching"]):
        raise ValueError("Preserve the course eager/chunked-prefill/no-prefix-cache controls")
    if config["long_input"] + config["output_len"] > config["max_model_len"]:
        raise ValueError("Input + output exceeds context")


def serve_command(config, budget):
    return ["vllm", "serve", config["model_path"], "--served-model-name", config["model_id"],
            "--tokenizer", config["tokenizer_path"], "--host", config["host"], "--port", str(config["port"]),
            "--dtype", config["dtype"], "--max-model-len", str(config["max_model_len"]),
            "--max-num-seqs", str(config["max_num_seqs"]),
            "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--enforce-eager", "--enable-chunked-prefill", "--no-enable-prefix-caching",
            "--max-num-batched-tokens", str(budget)]


def benchmark_command(config, row, out):
    if row["input_len"] + row["output_len"] > config["max_model_len"]:
        raise ValueError("Planned input + output exceeds server context")
    if row["request_rate"] != "inf":
        finite(row["request_rate"], "request_rate", positive=True)
    return ["vllm", "bench", "serve", "--backend", "vllm", "--model", config["model_id"],
            "--tokenizer", config["tokenizer_path"], "--host", config["host"], "--port", str(config["port"]),
            "--endpoint", "/v1/completions", "--dataset-name", "random",
            "--random-input-len", str(row["input_len"]), "--random-output-len", str(row["output_len"]),
            "--random-range-ratio", "0", "--num-prompts", str(row["num_prompts"]),
            "--request-rate", str(row["request_rate"]), "--burstiness", str(row["burstiness"]),
            "--ignore-eos", "--seed", str(row["seed"]), "--save-result", "--save-detailed",
            "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "50,95",
            "--result-dir", out.as_posix(), "--result-filename", "raw.json"]


def capacity_matrix(config):
    rows = []
    for name, count, seed in [("warmup", 8, 11), ("smoke", 20, 11),
                              ("capacity-r1", 120, 11), ("capacity-r2", 120, 22), ("capacity-r3", 120, 33)]:
        rows.append(dict(run_id="L03-" + name, input_len=config["long_input"], output_len=config["output_len"],
                         request_rate="inf", burstiness=1, num_prompts=count, seed=seed, budget=config["budget"]))
    return rows


def capture_line(out, config_path, command):
    return shlex.join(["python", "-m", "tools.capture", "--out", out.as_posix(),
                       "--config", config_path.as_posix(), "--", *command])


def generate(config, rows, out):
    validate_config(config)
    out = Path(out)
    # Validate all rows before creating output.
    if not rows or len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("Run IDs must be unique")
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["run_id"]):
            raise ValueError("Run IDs may only use letters, digits, underscore and hyphen")
        benchmark_command(config, row, out / "runs" / row["run_id"])
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "run_matrix.csv", rows)
    lines = ["# Generated plan. Run ONE line at a time from the repository root in WSL.",
             "# Check installed vllm --help. Start matching server budget in another terminal.",
             "# Wait for readiness; after every restart run 8 warmup requests (not measured).",
             "# Every output directory is new; change IDs for repeats, never overwrite data."]
    for budget in sorted({r["budget"] for r in rows}, reverse=True):
        server_config = out / f"server-{budget}.json"
        write_json(server_config, {**config, "budget": budget})
        lines.extend([f"\n# SERVER budget={budget}; rename output for each restart:",
                      "# " + capture_line(out / "servers" / f"budget-{budget}-restart1", server_config,
                                          serve_command(config, budget))])
        warmup = dict(rows[0], run_id=f"warmup-{budget}", budget=budget, num_prompts=8)
        warm_config = out / f"warmup-{budget}.json"
        write_json(warm_config, {**config, "workload": warmup, "phase": "warmup"})
        lines.extend(["# WARMUP for this server; rename output each restart:",
                      "# " + capture_line(out / "warmups" / f"budget-{budget}-restart1", warm_config,
                                          benchmark_command(config, warmup, out / "warmups" / f"budget-{budget}-restart1"))])
    for index, row in enumerate(rows, 1):
        cfg_path = out / "configs" / f"{row['run_id']}.json"
        write_json(cfg_path, {**config, "budget": row["budget"], "workload": row})
        result = out / "runs" / row["run_id"]
        lines.extend([f"\n# Step {index}: ensure budget={row['budget']}; previous run must drain first.",
                      capture_line(result, cfg_path, benchmark_command(config, row, result))])
    # Text, not executable: users inspect, run readiness checks and control restarts themselves.
    (out / "commands.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lab", choices=["lab3", "lab4", "lab6", "lab7"], default="lab3")
    parser.add_argument("--design", help="Frozen L04 design, reused unchanged for L06/L07")
    parser.add_argument("--pilot", action="store_true", help="L04 long-low, 30 requests")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    if args.lab == "lab3":
        rows = capacity_matrix(config)
    else:
        if not args.design:
            parser.error("--design required")
        design = read_json(args.design)
        if (design["max_model_len"], design["long_input"], design["output_len"]) != (
                config["max_model_len"], config["long_input"], config["output_len"]):
            raise ValueError("Frozen design and server context/length controls disagree")
        if args.pilot:
            if args.lab != "lab4":
                parser.error("--pilot applies only to lab4")
            rows = [dict(run_id="L04-pilot", input_len=design["long_input"], output_len=64,
                         request_rate=design["low"], burstiness=1, num_prompts=30, seed=11, budget=2048)]
        else:
            rows = importlib.import_module(args.lab + ".plan").build_matrix(design)
    generate(config, rows, args.out)
    print("Review commands and help before running:", Path(args.out) / "commands.txt")


if __name__ == "__main__":
    main()


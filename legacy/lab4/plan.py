"""L04: turn a real calibration into a frozen design, then fill the run matrix."""
import argparse
from common.io import read_json, write_json
from common.service import validate_design


def build_matrix(design):
    """12 dict rows. Keys: run_id,input_len,output_len,request_rate,burstiness,
    num_prompts,seed,budget. Seed 11/33: short-low,long-high,short-high,long-low;
    seed 22: reverse this order. Budget remains 2048.
    """
    validate_design(design)
    # TODO L04-2: three seeds x two lengths x two finite arrival rates.
    raise NotImplementedError("L04-2: construct the pre-registered 12-run matrix")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--num-prompts", type=int, default=120)
    parser.add_argument("--max-model-len", type=int, choices=[1024, 1536, 2048], default=2048)
    parser.add_argument("--out", required=True, help="Frozen design JSON (not a measurements file)")
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    calibration = read_json(args.calibration)
    C = calibration["C"]
    design = {"low": 0.3 * C, "high": 0.7 * C, "num_prompts": args.num_prompts,
              "long_input": 768 if args.max_model_len == 1024 else 1024,
              "heldout_input": {1024: 896, 1536: 1408, 2048: 1536}[args.max_model_len],
              "output_len": 64, "max_model_len": args.max_model_len,
              "calibration": args.calibration}
    validate_design(design)
    write_json(args.out, design)
    print("Design saved; verify N, preserve it through L04/L06/L07:", design)


if __name__ == "__main__":
    main()


"""Version-independent request schema. Native vLLM JSON is always preserved separately."""

import math


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if value <= 0 if positive else value < 0:
        raise ValueError(f"{name} has invalid sign")


def validate_run(run, allow_synthetic=True):
    if type(run.get("synthetic")) is not bool:
        raise ValueError("synthetic must be explicitly true or false")
    if run["synthetic"] and not allow_synthetic:
        raise ValueError("Synthetic fixtures cannot be used as measured evidence")
    if not isinstance(run.get("run_id"), str) or not run["run_id"]:
        raise ValueError("run_id required")
    finite(run["duration_s"], "duration_s", positive=True)
    requests = run["requests"]
    n = run["num_requested"]
    if type(n) is not int or n < 1 or len(requests) != n:
        raise ValueError("Every issued request, including failures, needs one row")
    ids = [r["request_id"] for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate request ids")
    for request in requests:
        if type(request["success"]) is not bool:
            raise ValueError("success must be a boolean")
        tokens = request["output_tokens"]
        if type(tokens) is not int or tokens < 0:
            raise ValueError("output_tokens must be a nonnegative integer")
        if request["success"]:
            if tokens < 1:
                raise ValueError("successful generation must have at least one output token")
            if type(request["input_tokens"]) is not int or request["input_tokens"] < 1:
                raise ValueError("actual input length required for successful requests")
            finite(request["ttft_s"], "ttft_s")
            if tokens > 1:
                finite(request["tpot_s"], "tpot_s")
            elif request["tpot_s"] is not None:
                raise ValueError("single-token TPOT must be null, never zero")
        elif not request.get("error"):
            raise ValueError("Failure reason required; use 'not provided by tool' if absent")
    return run


def percentile(values, q):
    """Supplied linear interpolation, equivalent to the usual inclusive linear quantile."""
    if not 0 <= q <= 1:
        raise ValueError("q outside [0,1]")
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def validate_design(design):
    for key in ("low", "high"):
        finite(design[key], key, positive=True)
    if design["low"] >= design["high"]:
        raise ValueError("low must be smaller than high")
    if type(design["num_prompts"]) is not int or design["num_prompts"] < 1:
        raise ValueError("positive integer num_prompts required")
    if not 512 < design["long_input"] < design["heldout_input"]:
        raise ValueError("long input >512; heldout input must be strictly longer")
    if design["output_len"] != 64 or design["heldout_input"] + 64 > design["max_model_len"]:
        raise ValueError("check output length and context limit")

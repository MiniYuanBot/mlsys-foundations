"""S04: connect a pinned real engine's client to the workload contract."""
def materialize(trace, tokenizer, prefix_tokens):
    """Map symbolic prefix/suffix identities to exact tokenized prompts.
    Preserve target lengths and exact shared prefix tokens. Save token IDs & tokenizer commit.
    """
    # TODO S04-2: implement for the chosen pinned tokenizer/engine API.
    raise NotImplementedError("S04-2: materialize and verify actual tokenized inputs")


def replay(trace, tokenized_requests, endpoint, output_directory):
    """Send at absolute monotonic offsets, without per-request blocking or a concurrency cap.
    Retain raw streaming events, usage/length, errors and scheduling/send timestamps.
    Check server protocol & installed-version benchmark client before implementing.
    """
    # TODO S04-3: adapt an upstream client; implement timeout/cancel/drain and diagnostics.
    # Do not treat network chunks as individual tokens. Retain failed requests.
    raise NotImplementedError("S04-3: implement a real controlled arrival client")


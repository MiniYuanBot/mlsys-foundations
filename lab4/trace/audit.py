"""Student contract for auditing a real replay client."""
def audit(trace, observations, jitter_limit_s=0.05):
    """observations: request_id, scheduled_s, actual_send_s, completed, error.
    Return num_requested, num_completed, num_failed, missing_ids, duplicate_ids,
    max_send_lag_s, arrival_valid. Failed requests count; never silently cap concurrency.
    Arrival_valid iff all IDs exactly once and all send delays in [0,jitter_limit_s].
    Negative delay is an early send and invalid, not 'good low latency'.
    """
    # TODO S04-1: reject mismatched scheduled times; audit actual send lag and coverage.
    raise NotImplementedError("S04-1: implement open-loop replay audit")


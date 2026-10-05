"""Provided analytical baseline, independent of the paged student allocator.

This is reservation accounting, not a timing simulator or a model of every
contiguous allocator. Equal capacity must be used when comparing policies.
"""


def contiguous_reserved_slots(lengths: tuple[int, ...], max_lengths: tuple[int, ...]) -> dict:
    """One private contiguous reservation per request; no prefix sharing."""
    if len(lengths) != len(max_lengths):
        raise ValueError("length vectors must have equal size")
    if any(n < 0 or cap < n for n, cap in zip(lengths, max_lengths)):
        raise ValueError("require 0 <= current length <= reserved max length")
    reserved = sum(max_lengths)
    used = sum(lengths)
    return {
        "logical_tokens": used,
        "reserved_token_slots": reserved,
        "unused_reserved_slots": reserved - used,
    }

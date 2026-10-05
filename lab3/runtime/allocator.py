"""Student implementation of a paged KV *metadata* state machine.

Token IDs stand in for cached token positions so tests can detect aliasing.
They are NOT K/V tensors. See README.md for the deliberately limited model.
"""

from collections import Counter
from dataclasses import dataclass
from math import ceil


class CapacityError(RuntimeError):
    """An operation cannot fit; the entire state must remain unchanged."""


@dataclass
class Block:
    tokens: tuple[int, ...] = ()
    ref_count: int = 0


@dataclass
class Sequence:
    block_ids: tuple[int, ...] = ()
    length: int = 0


class PagedKV:
    """Fixed-size page pool; each live sequence owns references to its pages.

    Public operations form a synchronous, single-threaded state machine.
    IDs and page size are abstract; no claim is made about vLLM's exact APIs.
    Implement only the four TODO methods. Helpers below are supplied for
    observation and contract validation, not as hidden reference solutions.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = {i: Block() for i in range(num_blocks)}
        self.free_ids = set(self.blocks)
        self.sequences: dict[str, Sequence] = {}

    def allocate(self, request_id: str, tokens: tuple[int, ...]) -> None:
        """Create a sequence with exactly these token IDs.

        Existing ID: ValueError. Insufficient free pages: CapacityError.
        Either error must preserve the full snapshot. Empty input is legal
        and owns zero pages. Prefer lowest free IDs for readable traces;
        correctness tests do not require a particular allocation order.

        TODO A: validate/preflight -> build pages/table -> commit. Do not
        reserve output-length capacity yet: pages are allocated on demand.
        """
        raise NotImplementedError("TODO A: allocate pages and a sequence table")

    def fork_prefix(self, source_id: str, new_id: str) -> None:
        """New sequence shares the source's entire currently cached prefix.

        This models prefix reuse after the prefix has already been computed.
        The two sequences initially have equal length and identical page IDs.
        Do not allocate/copy pages; increment each shared page's ref_count.
        A partial last page may be shared until one sequence appends to it.
        Missing source: KeyError; existing new_id: ValueError. Atomic errors.

        TODO B: retain page references and create an independent page table.
        Prefix hashing, arbitrary-prefix lookup, and eviction are extensions.
        """
        raise NotImplementedError("TODO B: share cached prefix by reference")

    def append(self, request_id: str, tokens: tuple[int, ...]) -> None:
        """Append all tokens, or change nothing if capacity is insufficient.

        Append to an unshared partial tail in place. If that partial tail is
        shared, allocate a private copy before changing it (copy-on-write).
        A full shared tail does NOT need copying: append on new pages.
        Empty input is a no-op for an existing ID. Missing ID: KeyError.

        TODO C: calculate all required pages (including a possible COW page)
        before mutating. Preserve other sequences' tokens and page tables.
        """
        raise NotImplementedError("TODO C: atomic append with copy-on-write")

    def release(self, request_id: str) -> None:
        """Remove the sequence and decrement each page reference.

        Reclaim pages whose ref_count reaches zero; reset their token payload.
        Missing/already released ID: KeyError with no state change.

        TODO D: free the sequence's references, not every referenced page.
        """
        raise NotImplementedError("TODO D: release references and reclaim pages")

    def read(self, request_id: str) -> tuple[int, ...]:
        """Inspect logical token IDs; supplied so tests can detect aliasing."""
        seq = self.sequences[request_id]
        return tuple(t for i in seq.block_ids for t in self.blocks[i].tokens)

    def snapshot(self) -> dict:
        """Deterministic JSON-serializable state, including free-page contents."""
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "free_ids": sorted(self.free_ids),
            "sequences": {
                rid: {"block_ids": list(seq.block_ids), "length": seq.length}
                for rid, seq in sorted(self.sequences.items())
            },
            "blocks": {
                str(i): {"tokens": list(b.tokens), "ref_count": b.ref_count}
                for i, b in sorted(self.blocks.items())
            },
        }

    def audit(self) -> None:
        """Assert structural invariants; this does not establish performance."""
        assert set(self.blocks) == set(range(self.num_blocks)), "pool changed"
        assert self.free_ids <= set(self.blocks), "unknown free page"
        refs: Counter[int] = Counter()
        for seq in self.sequences.values():
            assert seq.length >= 0, "negative sequence length"
            assert len(seq.block_ids) == ceil(seq.length / self.block_size)
            assert len(set(seq.block_ids)) == len(seq.block_ids), "self-alias"
            assert all(i in self.blocks for i in seq.block_ids), "unknown page"
            assert sum(len(self.blocks[i].tokens) for i in seq.block_ids) == seq.length
            for i in seq.block_ids[:-1]:
                assert len(self.blocks[i].tokens) == self.block_size, "partial interior"
            refs.update(seq.block_ids)
        for i, page in self.blocks.items():
            assert page.ref_count == refs[i], f"refcount mismatch on page {i}"
            assert (i in self.free_ids) == (refs[i] == 0), "free/live disagreement"
            if refs[i]:
                assert 0 < len(page.tokens) <= self.block_size
            else:
                assert page.tokens == (), "stale free-page contents"
        assert len(self.free_ids) + sum(v > 0 for v in refs.values()) == self.num_blocks

    def accounting(self) -> dict[str, int]:
        """Distinguish logical request tokens from physical shared payload."""
        live = [b for b in self.blocks.values() if b.ref_count]
        physical = sum(len(b.tokens) for b in live)
        capacity = len(live) * self.block_size
        return {
            "logical_tokens": sum(s.length for s in self.sequences.values()),
            "physical_tokens": physical,
            "live_blocks": len(live),
            "free_blocks": len(self.free_ids),
            "allocated_token_slots": capacity,
            "internal_fragmentation_slots": capacity - physical,
        }

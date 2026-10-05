"""Hand-computable contract tests, independent of allocation order.

Run from repository root: python -m unittest lab3.runtime.tests -v
TODOs intentionally raise errors. Passing here means CPU metadata semantics,
not GPU correctness, runtime integration, or the research portion of lab3.
"""

import unittest

from .allocator import CapacityError, PagedKV
from .reference import contiguous_reserved_slots


class TestProvidedInfrastructure(unittest.TestCase):
    def test_initial_state_and_invalid_geometry(self):
        kv = PagedKV(3, 4)
        kv.audit()
        self.assertEqual(kv.accounting()["free_blocks"], 3)
        self.assertEqual(kv.accounting()["logical_tokens"], 0)
        for geometry in ((0, 4), (2, 0), (-1, 4)):
            with self.assertRaises(ValueError):
                PagedKV(*geometry)

    def test_contiguous_baseline(self):
        self.assertEqual(
            contiguous_reserved_slots((5, 5), (12, 12)),
            {"logical_tokens": 10, "reserved_token_slots": 24,
             "unused_reserved_slots": 14},
        )
        with self.assertRaises(ValueError):
            contiguous_reserved_slots((5,), (4,))


class TestPagedKVContract(unittest.TestCase):
    def test_allocate_append_and_memory_conservation(self):
        kv = PagedKV(4, 4)
        kv.allocate("a", (1, 2, 3, 4, 5))
        kv.audit()
        self.assertEqual(kv.read("a"), (1, 2, 3, 4, 5))
        self.assertEqual(kv.accounting(), {
            "logical_tokens": 5, "physical_tokens": 5, "live_blocks": 2,
            "free_blocks": 2, "allocated_token_slots": 8,
            "internal_fragmentation_slots": 3,
        })
        kv.append("a", (6, 7, 8, 9))
        kv.audit()
        self.assertEqual(kv.read("a"), tuple(range(1, 10)))
        self.assertEqual(kv.accounting()["live_blocks"], 3)

    def test_shared_prefix_refcounts_and_partial_tail_cow(self):
        kv = PagedKV(4, 4)
        kv.allocate("a", (1, 2, 3, 4, 5))
        kv.fork_prefix("a", "b")
        before_a = kv.sequences["a"].block_ids
        self.assertEqual(kv.sequences["b"].block_ids, before_a)
        self.assertEqual([kv.blocks[i].ref_count for i in before_a], [2, 2])
        kv.append("b", (6,))
        kv.audit()
        after_b = kv.sequences["b"].block_ids
        self.assertEqual(before_a[0], after_b[0])  # full page stays shared
        self.assertNotEqual(before_a[1], after_b[1])  # partial page is copied
        self.assertEqual(kv.read("a"), (1, 2, 3, 4, 5))
        self.assertEqual(kv.read("b"), (1, 2, 3, 4, 5, 6))
        self.assertEqual(kv.accounting()["physical_tokens"], 7)
        self.assertEqual(kv.accounting()["logical_tokens"], 11)
        self.assertEqual(kv.blocks[before_a[0]].ref_count, 2)
        self.assertEqual(kv.blocks[before_a[1]].ref_count, 1)
        self.assertEqual(kv.blocks[after_b[1]].ref_count, 1)

    def test_full_shared_tail_append_does_not_copy(self):
        kv = PagedKV(2, 4)
        kv.allocate("a", (1, 2, 3, 4))
        kv.fork_prefix("a", "b")
        shared = kv.sequences["a"].block_ids[0]
        kv.append("b", (5,))  # one free page is enough; an extra COW is wrong
        kv.audit()
        self.assertEqual(kv.sequences["b"].block_ids[0], shared)
        self.assertEqual(kv.read("a"), (1, 2, 3, 4))
        self.assertEqual(kv.read("b"), (1, 2, 3, 4, 5))
        self.assertEqual(kv.accounting()["free_blocks"], 0)

    def test_release_shared_prefix_and_reuse(self):
        kv = PagedKV(2, 4)
        kv.allocate("a", (1, 2, 3, 4, 5))
        kv.fork_prefix("a", "b")
        kv.release("a")
        kv.audit()
        self.assertEqual(kv.accounting()["free_blocks"], 0)
        self.assertEqual(kv.read("b"), (1, 2, 3, 4, 5))
        self.assertTrue(all(kv.blocks[i].ref_count == 1 for i in kv.sequences["b"].block_ids))
        kv.release("b")
        kv.audit()
        self.assertEqual(kv.accounting()["free_blocks"], 2)
        kv.allocate("c", (7, 8, 9, 10, 11, 12, 13, 14))
        kv.audit()
        self.assertEqual(kv.read("c"), (7, 8, 9, 10, 11, 12, 13, 14))

    def test_capacity_failure_is_atomic_for_allocate(self):
        kv = PagedKV(2, 4)
        kv.allocate("a", (1, 2, 3, 4))
        before = kv.snapshot()
        with self.assertRaises(CapacityError):
            kv.allocate("b", (5, 6, 7, 8, 9))
        self.assertEqual(kv.snapshot(), before)
        kv.audit()

    def test_capacity_failure_is_atomic_after_partial_tail(self):
        kv = PagedKV(2, 4)
        kv.allocate("a", (1, 2, 3))
        before = kv.snapshot()
        with self.assertRaises(CapacityError):
            kv.append("a", (4, 5, 6, 7, 8, 9))  # needs 3 total pages
        self.assertEqual(kv.snapshot(), before)  # token 4 must not leak in
        kv.audit()

    def test_capacity_failure_counts_cow_page(self):
        kv = PagedKV(2, 4)
        kv.allocate("a", (1, 2, 3))
        kv.fork_prefix("a", "b")
        before = kv.snapshot()
        with self.assertRaises(CapacityError):
            kv.append("b", (4, 5))  # private tail + overflow page = 2 free needed
        self.assertEqual(kv.snapshot(), before)
        kv.audit()

    def test_repeated_release_and_duplicate_id_rejected_atomically(self):
        kv = PagedKV(3, 4)
        kv.allocate("a", (1,))
        kv.allocate("b", (2,))
        before = kv.snapshot()
        with self.assertRaises(ValueError):
            kv.allocate("a", (3,))
        self.assertEqual(kv.snapshot(), before)
        with self.assertRaises(ValueError):
            kv.fork_prefix("a", "b")
        self.assertEqual(kv.snapshot(), before)
        kv.release("a")
        after = kv.snapshot()
        with self.assertRaises(KeyError):
            kv.release("a")
        self.assertEqual(kv.snapshot(), after)
        kv.audit()

    def test_empty_sequence_and_empty_append(self):
        kv = PagedKV(1, 4)
        kv.allocate("a", ())
        kv.fork_prefix("a", "b")
        kv.audit()
        before = kv.snapshot()
        kv.append("a", ())
        self.assertEqual(kv.snapshot(), before)
        kv.append("a", (10,))
        self.assertEqual(kv.read("b"), ())
        kv.release("a")
        kv.release("b")
        kv.audit()
        self.assertEqual(kv.accounting()["free_blocks"], 1)

    def test_missing_ids_preserve_state(self):
        kv = PagedKV(2, 4)
        before = kv.snapshot()
        for operation in (
            lambda: kv.append("missing", (1,)),
            lambda: kv.fork_prefix("missing", "new"),
            lambda: kv.release("missing"),
        ):
            with self.assertRaises(KeyError):
                operation()
            self.assertEqual(kv.snapshot(), before)


if __name__ == "__main__":
    unittest.main()

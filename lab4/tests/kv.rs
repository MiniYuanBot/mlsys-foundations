//! Block manager tests (CPU only).   cargo test --release --test kv b1
use minillm::kv_student::BlockManager;
use minillm::sequence::Sequence;

fn seq(id: usize, tokens: Vec<u32>) -> Sequence {
    Sequence::new(id, tokens, 16, vec![])
}

#[test]
fn b1a_allocate_until_empty() {
    let mut kv = BlockManager::new(4, 16, false);
    let got: Vec<u32> = (0..4).map(|_| kv.allocate().unwrap()).collect();
    assert_eq!(got, vec![0, 1, 2, 3]);
    assert!(kv.allocate().is_none());
    assert_eq!(kv.num_free_blocks(), 0);
    for b in got {
        assert_eq!(kv.refcount(b), 1);
    }
}

#[test]
fn b1b_release_returns_to_free_list() {
    let mut kv = BlockManager::new(2, 16, false);
    let a = kv.allocate().unwrap();
    kv.release(a);
    assert_eq!(kv.num_free_blocks(), 2);
    assert_eq!(kv.refcount(a), 0);
}

#[test]
#[should_panic(expected = "double free")]
fn b1b_double_free_panics() {
    let mut kv = BlockManager::new(2, 16, false);
    let a = kv.allocate().unwrap();
    kv.release(a);
    kv.release(a);
}

#[test]
fn b1c_ensure_capacity_is_all_or_nothing() {
    let mut kv = BlockManager::new(5, 16, false);
    let mut s = seq(0, (0..40).collect());
    assert!(kv.ensure_capacity(&mut s, 40)); // 3 blocks
    assert_eq!(s.block_table.len(), 3);
    assert!(kv.ensure_capacity(&mut s, 48)); // still 3
    assert_eq!(s.block_table.len(), 3);
    let mut t = seq(1, (0..100).collect());
    assert!(!kv.ensure_capacity(&mut t, 100)); // needs 7, only 2 free
    assert!(t.block_table.is_empty(), "a failed ensure_capacity must not keep partial allocations");
    assert_eq!(kv.num_free_blocks(), 2);
    assert!(kv.ensure_capacity(&mut s, 49)); // 4th block
}

#[test]
fn b1d_free_sequence() {
    let mut kv = BlockManager::new(8, 16, false);
    let mut s = seq(0, (0..50).collect());
    assert!(kv.ensure_capacity(&mut s, 50));
    s.num_computed = 50;
    kv.free_sequence(&mut s);
    assert!(s.block_table.is_empty());
    assert_eq!(s.num_computed, 0);
    assert_eq!(kv.num_free_blocks(), 8);
}

/// Simulate computing a sequence: allocate, mark computed, register.
fn compute(kv: &mut BlockManager, s: &mut Sequence) {
    assert!(kv.ensure_capacity(s, s.len()));
    s.num_computed = s.len();
    kv.register_full_blocks(s);
}

#[test]
fn c2a_prefix_hit_shares_blocks() {
    let mut kv = BlockManager::new(16, 4, true);
    let shared: Vec<u32> = (100..112).collect(); // 3 full blocks
    let mut a = seq(0, [shared.clone(), vec![1, 2]].concat());
    compute(&mut kv, &mut a);
    let mut b = seq(1, [shared.clone(), vec![7, 8, 9]].concat());
    assert_eq!(kv.match_prefix(&mut b), 12);
    assert_eq!(b.num_computed, 12);
    assert_eq!(&b.block_table[..], &a.block_table[..3]);
    for &blk in &b.block_table {
        assert_eq!(kv.refcount(blk), 2);
    }
    assert_eq!(kv.stats.prefix_hits, 3);
}

#[test]
fn c2a_different_first_block_misses_everything() {
    let mut kv = BlockManager::new(16, 4, true);
    let mut a = seq(0, (0..12).collect());
    compute(&mut kv, &mut a);
    // Same tokens 4..12 but a different first block: chained hashes differ.
    let mut b = seq(1, [vec![99, 98, 97, 96], (4..12).collect()].concat());
    assert_eq!(kv.match_prefix(&mut b), 0);
}

#[test]
fn c2a_never_reuses_the_last_token() {
    let mut kv = BlockManager::new(16, 4, true);
    let mut a = seq(0, (0..8).collect());
    compute(&mut kv, &mut a);
    let mut b = seq(1, (0..8).collect()); // identical prompt, exactly 2 blocks
    assert_eq!(kv.match_prefix(&mut b), 4, "the last block must be recomputed to get logits");
    assert_eq!(b.block_table.len(), 1);
}

#[test]
fn c2b_freed_cached_blocks_survive_until_evicted() {
    let mut kv = BlockManager::new(4, 4, true);
    let mut a = seq(0, (0..9).collect()); // 2 full blocks + 1 partial
    compute(&mut kv, &mut a);
    let full = a.block_table[..2].to_vec();
    kv.free_sequence(&mut a);
    assert_eq!(kv.num_free_blocks(), 4);
    assert!(kv.is_cached(full[0]) && kv.is_cached(full[1]));
    // Reuse after free:
    let mut b = seq(1, (0..10).collect());
    assert_eq!(kv.match_prefix(&mut b), 8);
    kv.free_sequence(&mut b);
    // Exhaust the pool: the free list goes first, then cached blocks get evicted.
    let mut c = seq(2, (1000..1016).collect());
    assert!(kv.ensure_capacity(&mut c, 16));
    assert!(kv.stats.evictions >= 2);
    assert!(!kv.is_cached(full[0]) || !kv.is_cached(full[1]));
}

#[test]
fn c2_disabled_means_no_matching() {
    let mut kv = BlockManager::new(16, 4, false);
    let mut a = seq(0, (0..12).collect());
    compute(&mut kv, &mut a);
    let mut b = seq(1, (0..12).collect());
    assert_eq!(kv.match_prefix(&mut b), 0);
    kv.free_sequence(&mut a);
    assert_eq!(kv.num_free_blocks(), 16);
}

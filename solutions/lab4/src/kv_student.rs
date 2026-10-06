//! Part B1 + C2: the block manager (CPU side of PagedAttention).
//!
//!   cargo test --release --test kv            (pure CPU, no GPU needed)
//!
//! The KV cache is a pool of `num_blocks` fixed-size blocks of `block_size`
//! token slots. A sequence's block_table maps logical block i (positions
//! i*B .. i*B+B-1) to a physical block. Blocks are reference counted so that
//! several sequences can share a block (C2 prefix caching).
//!
//! Part C2 (prefix caching, vLLM's "automatic prefix caching"): every FULL
//! block gets a content hash  H(parent hash, its B tokens).  A table maps
//! hash -> physical block. When a new request arrives we walk its prompt
//! block by block, and while the hash is in the table we reuse that block
//! instead of recomputing it. A cached block whose refcount drops to 0 is
//! not freed right away: it becomes EVICTABLE and is only reused (and its
//! hash forgotten) when the free list is empty, least recently freed first.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use crate::sequence::Sequence;

#[derive(Debug, Default, Clone)]
pub struct KvStats {
    pub prefix_queries: usize,  // blocks looked up in the prefix table
    pub prefix_hits: usize,     // of which were found
    pub evictions: usize,
}

pub struct BlockManager {
    pub block_size: usize,
    pub num_blocks: usize,
    pub prefix_caching: bool,
    pub stats: KvStats,
    free: VecDeque<u32>,         // never-used or forgotten blocks, FIFO
    refcount: Vec<u32>,
    // ---- C2 ----
    block_hash: Vec<Option<u64>>, // hash of the content of a full, registered block
    #[allow(dead_code)] // used once you implement C2
    cached: HashMap<u64, u32>,    // hash -> block
    evictable: VecDeque<u32>,     // refcount 0 but still cached; front = least recently freed
}

/// Provided: content hash of one full block, chained to the previous block's hash.
pub fn block_hash(parent: Option<u64>, tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    parent.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize, prefix_caching: bool) -> Self {
        Self {
            block_size,
            num_blocks,
            prefix_caching,
            stats: KvStats::default(),
            free: (0..num_blocks as u32).collect(),
            refcount: vec![0; num_blocks],
            block_hash: vec![None; num_blocks],
            cached: HashMap::new(),
            evictable: VecDeque::new(),
        }
    }

    /// Blocks that can be handed out right now (free + evictable).
    pub fn num_free_blocks(&self) -> usize {
        self.free.len() + self.evictable.len()
    }

    pub fn refcount(&self, block: u32) -> u32 {
        self.refcount[block as usize]
    }

    pub fn is_cached(&self, block: u32) -> bool {
        self.block_hash[block as usize].is_some()
    }

    /// Blocks needed to hold `n` tokens.
    pub fn blocks_for(&self, n: usize) -> usize {
        n.div_ceil(self.block_size)
    }

    // ------------------------------------------------------------- B1 ----
    // TODO(B1a) allocate one block with refcount 1, or None if none left.
    //   1. take the front of `free` if any;
    //   2. else (C2) take the front of `evictable`: forget its hash (remove it
    //      from `cached`, set block_hash to None), count an eviction;
    //   3. else None.
    pub fn allocate(&mut self) -> Option<u32> {
        let b = if let Some(b) = self.free.pop_front() {
            b
        } else if let Some(b) = self.evictable.pop_front() {
            if let Some(h) = self.block_hash[b as usize].take() {
                self.cached.remove(&h);
            }
            self.stats.evictions += 1;
            b
        } else {
            return None;
        };
        debug_assert_eq!(self.refcount[b as usize], 0);
        self.refcount[b as usize] = 1;
        Some(b)
    }

    // TODO(B1b) drop one reference. At refcount 0:
    //   cached block (block_hash is Some) -> push_back to `evictable`
    //   otherwise                         -> push_back to `free`
    // Panic on a double free (refcount already 0): it is always a bug.
    pub fn release(&mut self, block: u32) {
        let rc = &mut self.refcount[block as usize];
        assert!(*rc > 0, "double free of block {block}");
        *rc -= 1;
        if *rc == 0 {
            if self.block_hash[block as usize].is_some() {
                self.evictable.push_back(block);
            } else {
                self.free.push_back(block);
            }
        }
    }

    // TODO(B1c) grow seq.block_table so it covers `num_tokens` tokens.
    // ALL OR NOTHING: if not enough blocks are available, allocate none and
    // return false (the scheduler then preempts or waits). Return true otherwise.
    pub fn ensure_capacity(&mut self, seq: &mut Sequence, num_tokens: usize) -> bool {
        let need = self.blocks_for(num_tokens).saturating_sub(seq.block_table.len());
        if need > self.num_free_blocks() {
            return false;
        }
        for _ in 0..need {
            let b = self.allocate().expect("checked above");
            seq.block_table.push(b);
        }
        true
    }

    // TODO(B1d) release every block of the sequence (in REVERSE order, so the
    // tail of a chain is evicted before its head) and clear its table and
    // num_computed (used for finished and for preempted sequences).
    pub fn free_sequence(&mut self, seq: &mut Sequence) {
        for &b in seq.block_table.iter().rev() {
            self.release(b);
        }
        seq.block_table.clear();
        seq.num_computed = 0;
    }

    // ------------------------------------------------------------- C2 ----
    // TODO(C2a) reuse cached blocks for a NEW sequence (empty block table,
    // num_computed == 0). Only when prefix_caching is on.
    //   walk full blocks of seq.tokens: for i in 0.., tokens[i*B..(i+1)*B],
    //     h = block_hash(previous h, those tokens); count a prefix query;
    //     if cached has h -> block b: count a hit; take a reference (if its
    //        refcount was 0 it is in `evictable`: remove it from there);
    //        push b to the block table
    //     else stop.
    //   Never reuse the LAST token: at least one token must be computed to
    //   produce logits. So if every block matched and len % B == 0, give back
    //   (release) the last matched block.
    //   Set seq.num_computed = seq.num_cached_tokens = matched_blocks * B.
    // Returns the number of reused tokens.
    pub fn match_prefix(&mut self, seq: &mut Sequence) -> usize {
        if !self.prefix_caching || !seq.block_table.is_empty() {
            return 0;
        }
        let bs = self.block_size;
        let mut parent = None;
        for chunk in seq.tokens.chunks_exact(bs) {
            let h = block_hash(parent, chunk);
            self.stats.prefix_queries += 1;
            let Some(&b) = self.cached.get(&h) else { break };
            self.stats.prefix_hits += 1;
            if self.refcount[b as usize] == 0 {
                let pos = self.evictable.iter().position(|&x| x == b).expect("cached block with refcount 0 is evictable");
                self.evictable.remove(pos);
            }
            self.refcount[b as usize] += 1;
            seq.block_table.push(b);
            parent = Some(h);
        }
        if seq.block_table.len() * bs >= seq.tokens.len() {
            let b = seq.block_table.pop().expect("at least one block matched");
            self.release(b);
        }
        seq.num_computed = seq.block_table.len() * bs;
        seq.num_cached_tokens = seq.num_computed;
        seq.num_computed
    }

    // TODO(C2b) after a step, register every block of `seq` that is now FULL
    // of computed tokens (index i with (i+1)*B <= seq.num_computed) and not
    // yet hashed: compute its chained hash (recompute the chain from block 0;
    // blocks are few), set block_hash[b] = Some(h) and cached[h] = b.
    // If another block already holds the same hash (two requests computed the
    // same prefix at the same time), leave this one unregistered.
    // Only when prefix_caching is on.
    pub fn register_full_blocks(&mut self, seq: &Sequence) {
        if !self.prefix_caching {
            return;
        }
        let bs = self.block_size;
        let mut parent = None;
        for (i, &b) in seq.block_table.iter().enumerate() {
            if (i + 1) * bs > seq.num_computed {
                break;
            }
            let h = block_hash(parent, &seq.tokens[i * bs..(i + 1) * bs]);
            parent = Some(h);
            if self.block_hash[b as usize].is_some() {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = self.cached.entry(h) {
                e.insert(b);
                self.block_hash[b as usize] = Some(h);
            }
        }
    }
}

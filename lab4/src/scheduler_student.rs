//! Part B3 + C1: the continuous-batching scheduler.
//!
//!   cargo test --release --test scheduler      (pure CPU, no GPU needed)
//!
//! Every engine step asks `schedule()` which sequences run and how many of
//! their uncomputed tokens to process. Policy (that of vLLM's v1 scheduler):
//!
//!   budget = max_batched_tokens  (tokens per step, the GPU's work limit)
//!
//!   1. RUNNING sequences first, in admission order. Each gets
//!        n = seq.num_uncomputed()     (1 when decoding)
//!        (C1, chunked prefill: n = min(n, budget); without chunking a running
//!         sequence always has n == 1 because its prompt was scheduled whole)
//!      and enough blocks for num_computed + n tokens. If the block manager
//!      cannot provide them, PREEMPT the most recently admitted running
//!      sequence (the back of `running`): free all its blocks, num_computed
//!      = 0, status Waiting, push it to the FRONT of `waiting` (it will be
//!      recomputed, prompt + generated so far). Repeat until the allocation
//!      succeeds; if the victim is the sequence itself, stop scheduling.
//!   2. Then admit WAITING sequences in FIFO order while
//!        running.len() < max_num_seqs  and  budget > 0:
//!        a. (C2) kv.match_prefix(seq) if its block table is empty
//!        b. n = seq.num_uncomputed(); if n > budget:
//!             chunked prefill: n = budget
//!             no chunking: stop admitting, UNLESS nothing at all was
//!               scheduled in this step (then run the whole prompt anyway,
//!               otherwise a long prompt would wait forever)
//!        c. admission control: the free blocks must cover the sequence's WHOLE
//!           remaining prompt (blocks_for(seq.len()) - block_table.len()), not
//!           just this chunk. Otherwise chunked prefill admits many half-done
//!           prompts that later evict each other (preemption thrashing).
//!           Then kv.ensure_capacity(seq, num_computed + n). If either fails,
//!           undo the prefix match (kv.free_sequence) and stop admitting
//!        d. move it to `running` (status Running) and schedule it.
//!      Static batching (the baseline): admit only if `running` was empty at
//!      the START of this call; then the whole batch runs to completion.
//!   Each scheduled sequence consumes n tokens of budget.

use std::collections::{HashMap, VecDeque};

use crate::kv_student::BlockManager;
use crate::sequence::{Sequence, Status};

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_batched_tokens: usize,
    pub chunked_prefill: bool,
    pub static_batching: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self { max_num_seqs: 64, max_batched_tokens: 2048, chunked_prefill: true, static_batching: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scheduled {
    pub seq_id: usize,
    pub num_tokens: usize,
}

#[derive(Debug, Default, Clone)]
pub struct SchedulerOutput {
    pub items: Vec<Scheduled>,
    pub preempted: Vec<usize>,
}

pub struct Scheduler {
    pub cfg: SchedulerConfig,
    pub waiting: VecDeque<usize>,
    pub running: Vec<usize>, // admission order
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig) -> Self {
        Self { cfg, waiting: VecDeque::new(), running: Vec::new() }
    }

    pub fn add(&mut self, id: usize) {
        self.waiting.push_back(id);
    }

    /// Provided: called by the engine when a sequence finishes.
    pub fn finish(&mut self, id: usize) {
        self.running.retain(|&x| x != id);
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    /// Provided: preempt `id` (recompute policy).
    #[allow(dead_code)]
    fn preempt(&mut self, id: usize, seqs: &mut HashMap<usize, Sequence>, kv: &mut BlockManager, out: &mut SchedulerOutput) {
        let seq = seqs.get_mut(&id).expect("unknown sequence");
        kv.free_sequence(seq);
        seq.status = Status::Waiting;
        seq.num_preemptions += 1;
        self.running.retain(|&x| x != id);
        self.waiting.push_front(id);
        out.preempted.push(id);
    }

    // TODO(B3 + C1) implement the policy described at the top of this file.
    #[allow(unreachable_code, unused_mut)]
    pub fn schedule(&mut self, seqs: &mut HashMap<usize, Sequence>, kv: &mut BlockManager) -> SchedulerOutput {
        let mut out = SchedulerOutput::default();
        let _ = (seqs, kv, &mut out);
        todo!("B3 schedule");
        out
    }
}

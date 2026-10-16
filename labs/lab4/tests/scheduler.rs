//! Scheduler tests (CPU only): we play the engine's role by hand.
//!   cargo test --release --test scheduler b3
use std::collections::HashMap;

use minillm::kv_student::BlockManager;
use minillm::scheduler_student::{Scheduled, Scheduler, SchedulerConfig};
use minillm::sequence::{Sequence, Status};

struct Sim {
    sched: Scheduler,
    kv: BlockManager,
    seqs: HashMap<usize, Sequence>,
}

impl Sim {
    fn new(cfg: SchedulerConfig, blocks: usize, prefix: bool) -> Self {
        Self {
            sched: Scheduler::new(cfg),
            kv: BlockManager::new(blocks, 4, prefix),
            seqs: HashMap::new(),
        }
    }
    fn add(&mut self, id: usize, prompt_len: usize, max_new: usize) {
        let toks = (0..prompt_len as u32)
            .map(|x| x + 1000 * id as u32)
            .collect();
        self.seqs
            .insert(id, Sequence::new(id, toks, max_new, vec![]));
        self.sched.add(id);
    }
    /// One step as the engine would do it: advance num_computed, sample token 7
    /// for sequences that reached their end, finish those at max_new_tokens.
    fn step(&mut self) -> Vec<Scheduled> {
        let out = self.sched.schedule(&mut self.seqs, &mut self.kv);
        for it in &out.items {
            let s = self.seqs.get_mut(&it.seq_id).unwrap();
            assert!(it.num_tokens >= 1 && s.num_computed + it.num_tokens <= s.len());
            assert!(
                s.block_table.len() * 4 >= s.num_computed + it.num_tokens,
                "blocks must cover the scheduled tokens"
            );
            s.num_computed += it.num_tokens;
            self.kv.register_full_blocks(s);
            if s.num_computed == s.len() {
                s.tokens.push(7);
                if s.should_stop(7) {
                    s.status = Status::Finished;
                    self.kv.free_sequence(s);
                    self.sched.finish(it.seq_id);
                }
            }
        }
        out.items
    }
    fn run(&mut self) -> usize {
        let mut steps = 0;
        while self.sched.has_work() {
            assert!(!self.step().is_empty(), "scheduler made no progress");
            steps += 1;
            assert!(steps < 10_000);
        }
        steps
    }
}

fn cfg(max_seqs: usize, budget: usize, chunked: bool) -> SchedulerConfig {
    SchedulerConfig {
        max_num_seqs: max_seqs,
        max_batched_tokens: budget,
        chunked_prefill: chunked,
        static_batching: false,
    }
}

#[test]
fn b3_prefill_then_decode() {
    let mut s = Sim::new(cfg(8, 64, false), 64, false);
    s.add(0, 10, 3);
    assert_eq!(
        s.step(),
        vec![Scheduled {
            seq_id: 0,
            num_tokens: 10
        }]
    );
    assert_eq!(
        s.step(),
        vec![Scheduled {
            seq_id: 0,
            num_tokens: 1
        }]
    );
    assert_eq!(
        s.step(),
        vec![Scheduled {
            seq_id: 0,
            num_tokens: 1
        }]
    );
    assert!(!s.sched.has_work());
    assert_eq!(s.seqs[&0].output(), &[7, 7, 7]);
}

#[test]
fn b3_continuous_batching_admits_while_others_decode() {
    let mut s = Sim::new(cfg(8, 64, false), 64, false);
    s.add(0, 10, 5);
    s.step();
    s.add(1, 20, 5);
    let items = s.step();
    // running decode first, then the new prefill joins the SAME step
    assert_eq!(
        items,
        vec![
            Scheduled {
                seq_id: 0,
                num_tokens: 1
            },
            Scheduled {
                seq_id: 1,
                num_tokens: 20
            }
        ]
    );
}

#[test]
fn b3_respects_max_num_seqs_and_budget() {
    let mut s = Sim::new(cfg(2, 25, false), 64, false);
    for i in 0..4 {
        s.add(i, 10, 2);
    }
    let items = s.step();
    assert_eq!(items.len(), 2, "max_num_seqs = 2");
    assert_eq!(items.iter().map(|x| x.num_tokens).sum::<usize>(), 20);
    s.run();
    assert!(s.seqs.values().all(|q| q.status == Status::Finished));
}

#[test]
fn b3_long_prompt_without_chunking_still_runs() {
    let mut s = Sim::new(cfg(4, 16, false), 64, false);
    s.add(0, 40, 1); // longer than the budget: runs alone, whole
    assert_eq!(
        s.step(),
        vec![Scheduled {
            seq_id: 0,
            num_tokens: 40
        }]
    );
}

#[test]
fn b3_preemption_when_out_of_blocks() {
    // 6 blocks of 4 tokens. Two sequences of 8 prompt tokens each fit (4 blocks),
    // but as they decode they need more; the later one must be preempted.
    let mut s = Sim::new(cfg(4, 64, false), 6, false);
    s.add(0, 8, 12);
    s.add(1, 8, 12);
    let mut preempted = 0;
    let mut steps = 0;
    while s.sched.has_work() {
        let out = s.sched.schedule(&mut s.seqs, &mut s.kv);
        preempted += out.preempted.len();
        for p in &out.preempted {
            assert_eq!(
                *p, 1,
                "the most recently admitted sequence is preempted first"
            );
            assert_eq!(s.seqs[p].num_computed, 0);
            assert!(s.seqs[p].block_table.is_empty());
        }
        for it in &out.items {
            let q = s.seqs.get_mut(&it.seq_id).unwrap();
            q.num_computed += it.num_tokens;
            if q.num_computed == q.len() {
                q.tokens.push(7);
                if q.should_stop(7) {
                    q.status = Status::Finished;
                    s.kv.free_sequence(q);
                    s.sched.finish(it.seq_id);
                }
            }
        }
        steps += 1;
        assert!(steps < 1000);
    }
    assert!(preempted >= 1);
    assert!(s
        .seqs
        .values()
        .all(|q| q.status == Status::Finished && q.num_generated() == 12));
}

#[test]
fn b3_static_batching_waits_for_the_whole_batch() {
    let mut c = cfg(2, 64, false);
    c.static_batching = true;
    let mut s = Sim::new(c, 64, false);
    s.add(0, 4, 1); // finishes after its prefill
    s.add(1, 4, 6);
    s.add(2, 4, 1);
    s.step(); // 0 and 1 prefill; 0 finishes
    for _ in 0..4 {
        let items = s.step();
        assert!(
            items.iter().all(|x| x.seq_id == 1),
            "static batching must not admit 2 while 1 runs"
        );
    }
    s.run();
    assert_eq!(s.seqs[&2].status, Status::Finished);
}

#[test]
fn c1_chunked_prefill_splits_long_prompts() {
    let mut s = Sim::new(cfg(4, 16, true), 64, false);
    s.add(0, 40, 2);
    let chunks: Vec<usize> = (0..3).map(|_| s.step()[0].num_tokens).collect();
    assert_eq!(chunks, vec![16, 16, 8]);
    assert_eq!(s.seqs[&0].num_generated(), 1); // sampled after the last chunk
}

#[test]
fn c1_decode_is_not_stalled_by_a_long_prefill() {
    let mut s = Sim::new(cfg(4, 16, true), 64, false);
    s.add(0, 4, 20);
    s.step();
    s.add(1, 100, 1);
    for _ in 0..5 {
        let items = s.step();
        assert_eq!(
            items[0],
            Scheduled {
                seq_id: 0,
                num_tokens: 1
            },
            "the decode runs every step"
        );
        assert_eq!(
            items.iter().map(|x| x.num_tokens).sum::<usize>(),
            16,
            "the prefill fills the rest of the budget"
        );
    }
}

#[test]
fn c2_scheduler_uses_the_prefix_cache() {
    let mut s = Sim::new(cfg(4, 64, true), 64, true);
    // Same 12 first tokens for both (ids are offset by 1000*id, so build by hand).
    let shared: Vec<u32> = (500..512).collect();
    for id in 0..2 {
        let toks = [shared.clone(), vec![id as u32 + 1, 9]].concat();
        s.seqs.insert(id, Sequence::new(id, toks, 1, vec![]));
    }
    s.sched.add(0);
    s.step();
    s.sched.add(1);
    let items = s.step();
    assert_eq!(
        items,
        vec![Scheduled {
            seq_id: 1,
            num_tokens: 2
        }],
        "12 of 14 tokens come from the cache"
    );
    assert_eq!(s.seqs[&1].num_cached_tokens, 12);
}

//! The engine loop: schedule -> build batch -> forward -> sample -> update. Provided.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;

use crate::batch::{BatchInput, DeviceBatch};
use crate::gpu::Gpu;
use crate::kv_student::BlockManager;
use crate::kvcache::KvCache;
use crate::model_student::Qwen3;
use crate::ops::{self, AttnImpl};
use crate::scheduler_student::{Scheduler, SchedulerConfig};
use crate::sequence::{Sequence, Status};

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub block_size: usize,
    /// Bytes of GPU memory for the KV cache.
    pub kv_cache_bytes: usize,
    pub sched: SchedulerConfig,
    pub prefix_caching: bool,
    pub attention: AttnImpl,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            kv_cache_bytes: 2 << 30,
            sched: SchedulerConfig::default(),
            prefix_caching: true,
            attention: AttnImpl::Fast,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct EngineStats {
    pub steps: usize,
    pub tokens_processed: usize, // prompt + decode tokens through the model
    pub forward_secs: f64,
    pub peak_blocks_used: usize,
    pub preemptions: usize,
}

pub struct Engine {
    pub gpu: Gpu,
    pub model: Qwen3,
    pub cache: KvCache,
    pub kv: BlockManager,
    pub sched: Scheduler,
    pub seqs: HashMap<usize, Sequence>,
    pub stats: EngineStats,
    next_id: usize,
}

impl Engine {
    pub fn new(model_dir: &Path, cfg: EngineConfig) -> Result<Self> {
        let gpu = Gpu::new()?;
        let mut model = Qwen3::load(&gpu, model_dir)?;
        model.attn = cfg.attention;
        Self::with_model(gpu, model, cfg)
    }

    pub fn with_model(gpu: Gpu, model: Qwen3, cfg: EngineConfig) -> Result<Self> {
        let num_blocks = KvCache::blocks_for_bytes(&model.cfg, cfg.block_size, cfg.kv_cache_bytes);
        anyhow::ensure!(num_blocks > 0, "KV cache too small");
        let cache = KvCache::new(&gpu, &model.cfg, num_blocks, cfg.block_size)?;
        Ok(Self {
            gpu,
            model,
            cache,
            kv: BlockManager::new(num_blocks, cfg.block_size, cfg.prefix_caching),
            sched: Scheduler::new(cfg.sched),
            seqs: HashMap::new(),
            stats: EngineStats::default(),
            next_id: 0,
        })
    }

    pub fn add_request(
        &mut self,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        stop_tokens: Vec<u32>,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.seqs
            .insert(id, Sequence::new(id, prompt, max_new_tokens, stop_tokens));
        self.sched.add(id);
        id
    }

    pub fn has_work(&self) -> bool {
        self.sched.has_work()
    }

    /// One engine step. Returns the (sequence id, new token) pairs produced.
    pub fn step(&mut self) -> Result<Vec<(usize, u32)>> {
        let out = self.sched.schedule(&mut self.seqs, &mut self.kv);
        self.stats.preemptions += out.preempted.len();
        if out.items.is_empty() {
            anyhow::ensure!(
                !self.sched.has_work(),
                "nothing could be scheduled: is the KV cache too small for the next request?"
            );
            return Ok(Vec::new());
        }
        let bs = self.cache.block_size;
        let items: Vec<(&[u32], usize, &[u32], bool)> = out
            .items
            .iter()
            .map(|it| {
                let s = &self.seqs[&it.seq_id];
                let end = s.num_computed + it.num_tokens;
                (
                    &s.tokens[s.num_computed..end],
                    s.num_computed,
                    s.block_table.as_slice(),
                    end == s.len(),
                )
            })
            .collect();
        let batch = BatchInput::build(&items, bs);
        let sampled_seqs: Vec<usize> = out
            .items
            .iter()
            .filter(|it| {
                let s = &self.seqs[&it.seq_id];
                s.num_computed + it.num_tokens == s.len()
            })
            .map(|it| it.seq_id)
            .collect();
        let t0 = Instant::now();
        let db = DeviceBatch::upload(&self.gpu, &batch)?;
        let logits = self.model.forward(&self.gpu, &db, &mut self.cache)?;
        let mut next = self.gpu.alloc::<i32>(db.num_samples)?;
        if db.num_samples > 0 {
            ops::argmax_rows(
                &self.gpu,
                &logits,
                &mut next,
                db.num_samples,
                self.model.cfg.vocab_size,
            )?;
        }
        let next = self.gpu.download(&next)?;
        self.stats.forward_secs += t0.elapsed().as_secs_f64();
        self.stats.steps += 1;
        self.stats.tokens_processed += batch.num_tokens();

        for it in &out.items {
            let s = self.seqs.get_mut(&it.seq_id).unwrap();
            s.num_computed += it.num_tokens;
            self.kv.register_full_blocks(s);
        }
        self.stats.peak_blocks_used = self
            .stats
            .peak_blocks_used
            .max(self.kv.num_blocks - self.kv.num_free_blocks());
        let now = Instant::now();
        let mut produced = Vec::new();
        for (&id, &tok) in sampled_seqs.iter().zip(next.iter().take(db.num_samples)) {
            let tok = tok as u32;
            let s = self.seqs.get_mut(&id).unwrap();
            s.tokens.push(tok);
            s.first_token_at.get_or_insert(now);
            produced.push((id, tok));
            if s.should_stop(tok) {
                s.status = Status::Finished;
                s.finished_at = Some(now);
                self.kv.free_sequence(s);
                self.sched.finish(id);
            }
        }
        Ok(produced)
    }

    /// Run until every request has finished.
    pub fn run_to_completion(&mut self) -> Result<()> {
        while self.has_work() {
            self.step()?;
        }
        Ok(())
    }

    /// Greedy generation for one prompt, from scratch (convenience for tests and the CLI).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: &[u32],
    ) -> Result<Vec<u32>> {
        let id = self.add_request(prompt.to_vec(), max_new_tokens, stop.to_vec());
        while self.seqs[&id].status != Status::Finished {
            self.step()?;
        }
        Ok(self.seqs[&id].output().to_vec())
    }
}

//! One sequence, one model, one KV cache: the building block of speculative
//! decoding. Provided.
//!
//! Invariant between rounds:   num_computed == tokens.len() - 1
//! (the last token has been sampled but not yet fed through the model).
//! The cache uses blocks 0, 1, 2, ... in order, so cache position p is slot p.

use anyhow::Result;
use half::bf16;
use minillm::batch::{BatchInput, DeviceBatch};
use minillm::gpu::Gpu;
use minillm::kvcache::KvCache;
use minillm::model_student::Qwen3;
use minillm::ops::{self, Buf};

pub struct Session<'m> {
    pub model: &'m Qwen3,
    pub cache: KvCache,
    pub tokens: Vec<u32>,
    pub num_computed: usize,
    pub forward_calls: usize,
    pub tokens_fed: usize,
    table: Vec<u32>,
}

/// Logits of the rows a forward call returned.
pub struct Logits {
    pub buf: Buf,
    pub rows: usize,
    pub vocab: usize,
}

impl Logits {
    /// argmax of every row (on the GPU).
    pub fn argmax(&self, gpu: &Gpu) -> Result<Vec<u32>> {
        let mut out = gpu.alloc::<i32>(self.rows)?;
        ops::argmax_rows(gpu, &self.buf, &mut out, self.rows, self.vocab)?;
        Ok(gpu.download(&out)?.into_iter().map(|x| x as u32).collect())
    }

    /// Row-wise softmax(logits / temperature) on the host, f32. Used by sampling.
    pub fn probs(&self, gpu: &Gpu, temperature: f32) -> Result<Vec<Vec<f32>>> {
        let n = self.rows * self.vocab;
        let mut f = gpu.alloc::<f32>(n)?;
        ops::bf16_to_f32(gpu, &self.buf, &mut f, n)?;
        let host = gpu.download(&f)?;
        Ok(host
            .chunks(self.vocab)
            .map(|row| {
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut p: Vec<f32> = row.iter().map(|x| ((x - m) / temperature).exp()).collect();
                let s: f32 = p.iter().sum();
                p.iter_mut().for_each(|x| *x /= s);
                p
            })
            .collect())
    }

    /// Keep only the given rows.
    pub fn select(&self, gpu: &Gpu, rows: &[usize]) -> Result<Logits> {
        let mut buf = gpu.alloc::<bf16>(rows.len() * self.vocab)?;
        let idx = gpu.upload(&rows.iter().map(|&r| r as i32).collect::<Vec<_>>())?;
        ops::gather_rows(gpu, &self.buf, &idx, &mut buf, rows.len(), self.vocab)?;
        Ok(Logits { buf, rows: rows.len(), vocab: self.vocab })
    }

    /// Indices of the k largest logits of every row (host side), largest first.
    pub fn topk(&self, gpu: &Gpu, k: usize) -> Result<Vec<Vec<u32>>> {
        let n = self.rows * self.vocab;
        let mut f = gpu.alloc::<f32>(n)?;
        ops::bf16_to_f32(gpu, &self.buf, &mut f, n)?;
        let host = gpu.download(&f)?;
        Ok(host
            .chunks(self.vocab)
            .map(|row| {
                let mut idx: Vec<u32> = (0..self.vocab as u32).collect();
                idx.select_nth_unstable_by(k - 1, |&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap());
                let mut top = idx[..k].to_vec();
                top.sort_by(|&a, &b| row[b as usize].partial_cmp(&row[a as usize]).unwrap());
                top
            })
            .collect())
    }
}

impl<'m> Session<'m> {
    pub fn new(gpu: &Gpu, model: &'m Qwen3, prompt: &[u32], max_len: usize) -> Result<Self> {
        let block_size = 16;
        let blocks = max_len.div_ceil(block_size);
        let cache = KvCache::new(gpu, &model.cfg, blocks, block_size)?;
        Ok(Self {
            model,
            cache,
            tokens: prompt.to_vec(),
            num_computed: 0,
            forward_calls: 0,
            tokens_fed: 0,
            table: (0..blocks as u32).collect(),
        })
    }

    pub fn capacity(&self) -> usize {
        self.table.len() * self.cache.block_size
    }

    /// Feed `new` at cache positions num_computed.. and return the logits of
    /// every fed position. Does NOT touch `tokens`; advances num_computed.
    pub fn feed(&mut self, gpu: &Gpu, new: &[u32]) -> Result<Logits> {
        let start = self.num_computed;
        anyhow::ensure!(start + new.len() <= self.capacity(), "session capacity exceeded");
        let mut b = BatchInput::build(&[(new, start, &self.table, true)], self.cache.block_size);
        b.sample_rows = (0..new.len() as i32).collect();
        let db = DeviceBatch::upload(gpu, &b)?;
        let buf = self.model.forward(gpu, &db, &mut self.cache)?;
        self.num_computed += new.len();
        self.forward_calls += 1;
        self.tokens_fed += new.len();
        Ok(Logits { buf, rows: new.len(), vocab: self.model.cfg.vocab_size })
    }

    /// Feed every committed token except the last one (no-op if up to date).
    pub fn sync_prefix(&mut self, gpu: &Gpu) -> Result<()> {
        let end = self.tokens.len() - 1;
        if self.num_computed < end {
            let pending = self.tokens[self.num_computed..end].to_vec();
            self.feed(gpu, &pending)?;
        }
        Ok(())
    }

    /// Bring the cache up to date: feed tokens[num_computed..] and return the
    /// logits of the LAST token (the distribution of the next token).
    pub fn catch_up(&mut self, gpu: &Gpu) -> Result<Logits> {
        anyhow::ensure!(self.num_computed < self.tokens.len(), "nothing to feed");
        let pending = self.tokens[self.num_computed..].to_vec();
        let all = self.feed(gpu, &pending)?;
        // keep only the last row
        let mut last = gpu.alloc::<bf16>(all.vocab)?;
        let rows = gpu.upload(&[(all.rows - 1) as i32])?;
        ops::gather_rows(gpu, &all.buf, &rows, &mut last, 1, all.vocab)?;
        Ok(Logits { buf: last, rows: 1, vocab: all.vocab })
    }

    /// Forget cache contents beyond position `n` (they will be overwritten).
    pub fn truncate(&mut self, n: usize) {
        self.num_computed = self.num_computed.min(n);
    }

    /// Feed a token tree in ONE forward pass (lab5 Part C).
    ///   tokens[0] is the root = the last committed token (tokens.last()),
    ///   parents[i] < i for i >= 1 (parents[0] is ignored), at most 64 nodes.
    /// Node i is stored at cache position num_computed + i and gets RoPE
    /// position num_computed + depth(i). Returns logits for every node.
    /// Does not advance num_computed: call `commit_tree_path` afterwards.
    pub fn feed_tree(&mut self, gpu: &Gpu, tokens: &[u32], parents: &[usize]) -> Result<Logits> {
        let n = tokens.len();
        anyhow::ensure!(n <= 64 && parents.len() == n, "tree must have at most 64 nodes");
        let base = self.num_computed;
        anyhow::ensure!(base + n <= self.capacity(), "session capacity exceeded");
        let mut mask = vec![0u64; n];
        let mut depth = vec![0usize; n];
        for i in 0..n {
            mask[i] = 1u64 << i;
            if i > 0 {
                anyhow::ensure!(parents[i] < i, "parents must point to earlier nodes");
                mask[i] |= mask[parents[i]];
                depth[i] = depth[parents[i]] + 1;
            }
        }
        let bs = self.cache.block_size;
        let b = BatchInput {
            tokens: tokens.iter().map(|&t| t as i32).collect(),
            positions: depth.iter().map(|d| (base + d) as i32).collect(),
            token_seq: vec![0; n],
            slots: (0..n).map(|i| (base + i) as i32).collect(), // table is the identity
            block_tables: self.table.iter().map(|&x| x as i32).collect(),
            max_blocks: self.table.len(),
            sample_rows: (0..n as i32).collect(),
            tree_mask: mask,
            prefix_len: base,
        };
        debug_assert_eq!(bs, 16);
        let db = DeviceBatch::upload(gpu, &b)?;
        let buf = self.model.forward(gpu, &db, &mut self.cache)?;
        self.forward_calls += 1;
        self.tokens_fed += n;
        Ok(Logits { buf, rows: n, vocab: self.model.cfg.vocab_size })
    }

    /// After `feed_tree`: keep the root-to-node path `path` (node indices,
    /// path[0] == 0 is the root, path[d] has depth d). Moves their K/V to cache
    /// positions num_computed + d and advances num_computed past the path.
    pub fn commit_tree_path(&mut self, gpu: &Gpu, path: &[usize]) -> Result<()> {
        anyhow::ensure!(path.first() == Some(&0), "path must start at the root");
        let base = self.num_computed;
        let moves: Vec<(i32, i32)> =
            path.iter().enumerate().filter(|(d, &j)| *d != j).map(|(d, &j)| ((base + j) as i32, (base + d) as i32)).collect();
        if !moves.is_empty() {
            let src = gpu.upload(&moves.iter().map(|m| m.0).collect::<Vec<_>>())?;
            let dst = gpu.upload(&moves.iter().map(|m| m.1).collect::<Vec<_>>())?;
            let row = self.model.cfg.kv_dim();
            let mut tmp = gpu.alloc::<bf16>(moves.len() * row)?;
            for l in 0..self.cache.k.len() {
                for cache in [&mut self.cache.k[l], &mut self.cache.v[l]] {
                    crate::ops5::move_slots(gpu, cache, &src, &dst, &mut tmp, moves.len(), row)?;
                }
            }
        }
        self.num_computed = base + path.len();
        Ok(())
    }
}

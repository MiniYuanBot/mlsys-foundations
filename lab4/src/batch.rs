//! One forward step's inputs, "flattened" across sequences (like vLLM's
//! ModelRunner). Provided.
//!
//! A step processes T tokens from S sequences, e.g. a 5-token prefill chunk of
//! sequence 0 and one decode token each for sequences 1 and 2:
//!
//!   tokens     [t0 t1 t2 t3 t4 | u7 | w3]          T = 7
//!   positions  [ 0  1  2  3  4 |  7 |  3]
//!   token_seq  [ 0  0  0  0  0 |  1 |  2]          row of block_tables
//!   slots      where each token's K/V is written: block_tables[seq][pos / B] * B + pos % B
//!   sample_rows = [4, 5, 6]                        tokens whose logits we need
//!
//! Attention for token i covers positions 0..=positions[i] of its sequence,
//! reading earlier tokens' K/V from the cache: this is what lets prefill
//! chunks and decode tokens share one kernel launch (continuous batching).

use anyhow::Result;
use cudarc::driver::CudaSlice;

use crate::gpu::Gpu;

#[derive(Debug, Clone, Default)]
pub struct BatchInput {
    pub tokens: Vec<i32>,
    pub positions: Vec<i32>,
    pub token_seq: Vec<i32>,
    pub slots: Vec<i32>,
    /// [num_seqs * max_blocks], padded with 0.
    pub block_tables: Vec<i32>,
    pub max_blocks: usize,
    pub sample_rows: Vec<i32>,
    /// Tree attention (lab5 Part C). Empty = normal causal attention. Otherwise
    /// one sequence only: token i is tree node i, its K/V sit at cache position
    /// `prefix_len + i`, and it attends to cache positions 0..prefix_len plus
    /// the tree nodes j whose bit is set in tree_mask[i] (its ancestors and itself).
    pub tree_mask: Vec<u64>,
    pub prefix_len: usize,
}

impl BatchInput {
    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Build a batch from (tokens of this step, first position, block table) per sequence.
    /// The last token of each entry is sampled if `sample[i]` is true.
    pub fn build(items: &[(&[u32], usize, &[u32], bool)], block_size: usize) -> Self {
        let max_blocks = items.iter().map(|(_, _, bt, _)| bt.len()).max().unwrap_or(1).max(1);
        let mut b = BatchInput { max_blocks, ..Default::default() };
        for (s, (toks, start, table, sample)) in items.iter().enumerate() {
            for (k, &tok) in toks.iter().enumerate() {
                let pos = start + k;
                b.tokens.push(tok as i32);
                b.positions.push(pos as i32);
                b.token_seq.push(s as i32);
                let block = table[pos / block_size] as usize;
                b.slots.push((block * block_size + pos % block_size) as i32);
            }
            if *sample && !toks.is_empty() {
                b.sample_rows.push((b.tokens.len() - 1) as i32);
            }
            let mut row = vec![0i32; max_blocks];
            for (i, &blk) in table.iter().enumerate() {
                row[i] = blk as i32;
            }
            b.block_tables.extend(row);
        }
        b
    }

    /// A single sequence whose cache uses blocks 0, 1, 2, ... (for tests that
    /// do not involve the block manager). Every position is sampled.
    pub fn single_sequence(tokens: &[u32], block_size: usize) -> Self {
        let blocks: Vec<u32> = (0..tokens.len().div_ceil(block_size) as u32).collect();
        let mut b = Self::build(&[(tokens, 0, &blocks, true)], block_size);
        b.sample_rows = (0..tokens.len() as i32).collect();
        b
    }
}

/// The same arrays on the GPU.
pub struct DeviceBatch {
    pub tokens: CudaSlice<i32>,
    pub positions: CudaSlice<i32>,
    pub token_seq: CudaSlice<i32>,
    pub slots: CudaSlice<i32>,
    pub block_tables: CudaSlice<i32>,
    pub sample_rows: CudaSlice<i32>,
    pub num_tokens: usize,
    pub num_samples: usize,
    pub max_blocks: usize,
    pub tree_mask: Option<CudaSlice<u64>>,
    pub prefix_len: usize,
}

impl DeviceBatch {
    pub fn upload(gpu: &Gpu, b: &BatchInput) -> Result<Self> {
        Ok(Self {
            tokens: gpu.upload(&b.tokens)?,
            positions: gpu.upload(&b.positions)?,
            token_seq: gpu.upload(&b.token_seq)?,
            slots: gpu.upload(&b.slots)?,
            block_tables: gpu.upload(&b.block_tables)?,
            sample_rows: gpu.upload(&b.sample_rows)?,
            num_tokens: b.tokens.len(),
            num_samples: b.sample_rows.len(),
            max_blocks: b.max_blocks,
            tree_mask: if b.tree_mask.is_empty() { None } else { Some(gpu.upload(&b.tree_mask)?) },
            prefix_len: b.prefix_len,
        })
    }
}

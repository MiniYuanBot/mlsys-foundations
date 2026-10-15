//! GPU memory for the paged KV cache. Provided.
//! Which block belongs to which sequence is decided by the BlockManager
//! (kv_student.rs); this struct only owns the storage.

use anyhow::Result;
use half::bf16;

use crate::config::Qwen3Config;
use crate::gpu::Gpu;
use crate::ops::Buf;

pub struct KvCache {
    pub k: Vec<Buf>, // per layer: [num_blocks, block_size, kv_heads, head_dim]
    pub v: Vec<Buf>,
    pub num_blocks: usize,
    pub block_size: usize,
}

impl KvCache {
    pub fn new(gpu: &Gpu, cfg: &Qwen3Config, num_blocks: usize, block_size: usize) -> Result<Self> {
        let per_layer = num_blocks * block_size * cfg.kv_dim();
        let mut k = Vec::new();
        let mut v = Vec::new();
        for _ in 0..cfg.num_hidden_layers {
            k.push(gpu.zeros::<bf16>(per_layer)?);
            v.push(gpu.zeros::<bf16>(per_layer)?);
        }
        Ok(Self {
            k,
            v,
            num_blocks,
            block_size,
        })
    }

    /// How many blocks fit in `bytes` of memory.
    pub fn blocks_for_bytes(cfg: &Qwen3Config, block_size: usize, bytes: usize) -> usize {
        bytes / (block_size * cfg.kv_bytes_per_token())
    }
}

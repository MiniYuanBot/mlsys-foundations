//! Part A2: the Qwen3 forward pass. Fill in TODO(A2).
//!
//!   cargo test --release --test model        (compares with Hugging Face, see README)
//!
//! You compose the ops from `ops.rs` (cuBLAS GEMMs + your A1 kernels). The
//! structure of one decoder layer, exactly as in HF `modeling_qwen3.py`:
//!
//!   x, residual = fused_add_rmsnorm(h, residual, input_layernorm)   (first layer: residual = h)
//!   q = x Wq^T  [T, 16*128]   k = x Wk^T  [T, 8*128]   v = x Wv^T  [T, 8*128]
//!   q = rmsnorm per head (q_norm)      k = rmsnorm per head (k_norm)   <- Qwen3-specific
//!   q, k = rope(q, k, positions)
//!   write k, v into the paged cache at this step's slots
//!   o = paged_attention(q, cache)       [T, 16*128]
//!   h = o Wo^T                          [T, hidden]
//!   x, residual = fused_add_rmsnorm(h, residual, post_attention_layernorm)
//!   h = (silu(x Wg^T) * (x Wu^T)) Wd^T
//!
//! After the last layer: x = fused_add_rmsnorm(h, residual, norm); keep only the
//! rows in `sample_rows`; logits = x Wlm^T  [num_samples, vocab].

use std::path::Path;

use anyhow::Result;
use half::bf16;

use crate::batch::DeviceBatch;
use crate::config::Qwen3Config;
use crate::gpu::Gpu;
use crate::kvcache::KvCache;
use crate::ops::{self, AttnArgs, AttnImpl, Buf};
use crate::weights::Weights;

pub struct Qwen3 {
    pub cfg: Qwen3Config,
    pub w: Weights,
    pub attn: AttnImpl,
}

impl Qwen3 {
    pub fn load(gpu: &Gpu, dir: &Path) -> Result<Self> {
        let cfg = Qwen3Config::load(dir)?;
        let w = Weights::load(gpu, dir, &cfg)?;
        Ok(Self { cfg, w, attn: AttnImpl::Fast })
    }

    /// Run one step. Returns logits [num_samples, vocab] (bf16, on the GPU).
    #[allow(unreachable_code, unused_mut, unused_variables)]
    pub fn forward(&self, gpu: &Gpu, b: &DeviceBatch, cache: &mut KvCache) -> Result<Buf> {
        let c = &self.cfg;
        let t = b.num_tokens;
        let (h, inter, qd, kvd) = (c.hidden_size, c.intermediate_size, c.q_dim(), c.kv_dim());
        let (heads, kv_heads, eps) = (c.num_attention_heads, c.num_key_value_heads, c.rms_norm_eps);
        // Activation buffers for this step (stream-ordered allocations, cheap).
        let mut hidden = gpu.alloc::<bf16>(t * h)?;   // h: the sub-layer output / next input
        let mut residual = gpu.alloc::<bf16>(t * h)?; // the residual stream
        let mut q_raw = gpu.alloc::<bf16>(t * qd)?;
        let mut k_raw = gpu.alloc::<bf16>(t * kvd)?;
        let mut q = gpu.alloc::<bf16>(t * qd)?;
        let mut k = gpu.alloc::<bf16>(t * kvd)?;
        let mut v = gpu.alloc::<bf16>(t * kvd)?;
        let mut attn_out = gpu.alloc::<bf16>(t * qd)?;
        let mut gate = gpu.alloc::<bf16>(t * inter)?;
        let mut up = gpu.alloc::<bf16>(t * inter)?;
        let mut act = gpu.alloc::<bf16>(t * inter)?;
        let attn_args = AttnArgs {
            block_tables: &b.block_tables,
            token_seq: &b.token_seq,
            token_pos: &b.positions,
            t,
            heads,
            kv_heads,
            max_blocks: b.max_blocks,
            block_size: cache.block_size,
            tree: b.tree_mask.as_ref().map(|m| (m, b.prefix_len)),
        };

        ops::embedding(gpu, &self.w.embed_tokens, &b.tokens, &mut hidden, t, h)?;
        // Before layer 0 the residual stream is zero: then fused_add_rmsnorm
        // gives residual = h and x = rmsnorm(h), as HF does.
        gpu.stream.memset_zeros(&mut residual)?;

        // TODO(A2) the 28 decoder layers. `hidden` holds h on entry to each
        // layer; use the buffers above (and `hidden` again for o Wo^T and for
        // the MLP output). Write K/V with ops::reshape_and_cache(.., &mut
        // cache.k[l], &mut cache.v[l], &b.slots, ..) BEFORE attention, then
        // ops::paged_attention(gpu, self.attn, &q, &cache.k[l], &cache.v[l], &mut attn_out, &attn_args).
        let _ = (&mut q_raw, &mut k_raw, &mut q, &mut k, &mut v, &mut attn_out, &mut gate, &mut up, &mut act, &attn_args, inter, qd, kvd, heads, kv_heads);
        anyhow::bail!("TODO A2: decoder layers in model_student.rs");

        // Final norm, keep the sampled rows, project to the vocabulary (provided).
        ops::fused_add_rmsnorm(gpu, &mut hidden, &mut residual, &self.w.norm, t, h, eps)?;
        let n = b.num_samples;
        let mut picked = gpu.alloc::<bf16>(n * h)?;
        ops::gather_rows(gpu, &hidden, &b.sample_rows, &mut picked, n, h)?;
        let mut logits = gpu.alloc::<bf16>(n * c.vocab_size)?;
        ops::linear(gpu, &picked, &self.w.lm_head, &mut logits, n, h, c.vocab_size)?;
        Ok(logits)
    }
}

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
    /// Attention kernel. Defaults to the provided `paged_attention_naive`, so
    /// that A2 does not depend on your B2 kernel (`generate` / `bench` pick Fast).
    pub attn: AttnImpl,
}

impl Qwen3 {
    pub fn load(gpu: &Gpu, dir: &Path) -> Result<Self> {
        let cfg = Qwen3Config::load(dir)?;
        let w = Weights::load(gpu, dir, &cfg)?;
        Ok(Self {
            cfg,
            w,
            attn: AttnImpl::Naive,
        })
    }

    /// Run one step. Returns logits [num_samples, vocab] (bf16, on the GPU).
    #[allow(unreachable_code, unused_mut, unused_variables)]
    pub fn forward(&self, gpu: &Gpu, b: &DeviceBatch, cache: &mut KvCache) -> Result<Buf> {
        let c = &self.cfg;
        let t = b.num_tokens;
        let (h, inter, qd, kvd) = (c.hidden_size, c.intermediate_size, c.q_dim(), c.kv_dim());
        let (heads, kv_heads, eps) = (c.num_attention_heads, c.num_key_value_heads, c.rms_norm_eps);
        // Activation buffers for this step (stream-ordered allocations, cheap).
        let mut hidden = gpu.alloc::<bf16>(t * h)?; // h: the sub-layer output / next input
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
        for (l, lw) in self.w.layers.iter().enumerate() {
            ops::fused_add_rmsnorm(
                gpu,
                &mut hidden,
                &mut residual,
                &lw.input_layernorm,
                t,
                h,
                eps,
            )?;
            ops::linear(gpu, &hidden, &lw.q_proj, &mut q_raw, t, h, qd)?;
            ops::linear(gpu, &hidden, &lw.k_proj, &mut k_raw, t, h, kvd)?;
            ops::linear(gpu, &hidden, &lw.v_proj, &mut v, t, h, kvd)?;
            ops::rmsnorm(gpu, &q_raw, &lw.q_norm, &mut q, t * heads, c.head_dim, eps)?;
            ops::rmsnorm(
                gpu,
                &k_raw,
                &lw.k_norm,
                &mut k,
                t * kv_heads,
                c.head_dim,
                eps,
            )?;
            ops::rope(
                gpu,
                &mut q,
                &mut k,
                &b.positions,
                t,
                heads,
                kv_heads,
                c.rope_theta,
            )?;
            ops::reshape_and_cache(
                gpu,
                &k,
                &v,
                &mut cache.k[l],
                &mut cache.v[l],
                &b.slots,
                t,
                kv_heads,
            )?;
            ops::paged_attention(
                gpu,
                self.attn,
                &q,
                &cache.k[l],
                &cache.v[l],
                &mut attn_out,
                &attn_args,
            )?;
            ops::linear(gpu, &attn_out, &lw.o_proj, &mut hidden, t, qd, h)?;
            ops::fused_add_rmsnorm(
                gpu,
                &mut hidden,
                &mut residual,
                &lw.post_attention_layernorm,
                t,
                h,
                eps,
            )?;
            ops::linear(gpu, &hidden, &lw.gate_proj, &mut gate, t, h, inter)?;
            ops::linear(gpu, &hidden, &lw.up_proj, &mut up, t, h, inter)?;
            ops::silu_mul(gpu, &gate, &up, &mut act, t * inter)?;
            ops::linear(gpu, &act, &lw.down_proj, &mut hidden, t, inter, h)?;
        }

        // Final norm, keep the sampled rows, project to the vocabulary (provided).
        ops::fused_add_rmsnorm(gpu, &mut hidden, &mut residual, &self.w.norm, t, h, eps)?;
        let n = b.num_samples;
        let mut picked = gpu.alloc::<bf16>(n * h)?;
        ops::gather_rows(gpu, &hidden, &b.sample_rows, &mut picked, n, h)?;
        let mut logits = gpu.alloc::<bf16>(n * c.vocab_size)?;
        ops::linear(
            gpu,
            &picked,
            &self.w.lm_head,
            &mut logits,
            n,
            h,
            c.vocab_size,
        )?;
        Ok(logits)
    }
}

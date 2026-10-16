//! Load Qwen3 safetensors (bf16) straight into GPU memory. Provided.
//!
//! Every Linear weight keeps PyTorch's layout [out_features, in_features],
//! row-major. `ops::linear` knows how to multiply by it without a transpose.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use cudarc::driver::CudaSlice;
use half::bf16;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::config::Qwen3Config;
use crate::gpu::Gpu;

pub struct LayerWeights {
    pub input_layernorm: CudaSlice<bf16>, // [hidden]
    pub q_proj: CudaSlice<bf16>,          // [q_dim, hidden]
    pub k_proj: CudaSlice<bf16>,          // [kv_dim, hidden]
    pub v_proj: CudaSlice<bf16>,          // [kv_dim, hidden]
    pub o_proj: CudaSlice<bf16>,          // [hidden, q_dim]
    pub q_norm: CudaSlice<bf16>,          // [head_dim]
    pub k_norm: CudaSlice<bf16>,          // [head_dim]
    pub post_attention_layernorm: CudaSlice<bf16>,
    pub gate_proj: CudaSlice<bf16>, // [intermediate, hidden]
    pub up_proj: CudaSlice<bf16>,   // [intermediate, hidden]
    pub down_proj: CudaSlice<bf16>, // [hidden, intermediate]
}

pub struct Weights {
    pub embed_tokens: CudaSlice<bf16>, // [vocab, hidden]
    pub lm_head: CudaSlice<bf16>,      // [vocab, hidden] (a copy of embed_tokens when tied)
    pub norm: CudaSlice<bf16>,         // [hidden]
    pub layers: Vec<LayerWeights>,
    pub bytes: usize,
}

struct Files {
    maps: Vec<Mmap>,
}

impl Files {
    fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| {
                format!(
                    "model dir {} not found (run tools/setup_wsl.sh)",
                    dir.display()
                )
            })?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        anyhow::ensure!(
            !paths.is_empty(),
            "no .safetensors files in {}",
            dir.display()
        );
        let maps = paths
            .iter()
            .map(|p| unsafe { Mmap::map(&std::fs::File::open(p)?) }.map_err(anyhow::Error::from))
            .collect::<Result<_>>()?;
        Ok(Self { maps })
    }

    fn find(&self, name: &str) -> Result<Option<(Vec<usize>, Vec<bf16>)>> {
        for m in &self.maps {
            let st = SafeTensors::deserialize(m)?;
            if let Ok(t) = st.tensor(name) {
                anyhow::ensure!(
                    t.dtype() == Dtype::BF16,
                    "{name}: expected BF16, got {:?}",
                    t.dtype()
                );
                let data: Vec<bf16> = t
                    .data()
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]))
                    .collect();
                return Ok(Some((t.shape().to_vec(), data)));
            }
        }
        Ok(None)
    }
}

impl Weights {
    pub fn load(gpu: &Gpu, dir: &Path, cfg: &Qwen3Config) -> Result<Self> {
        let files = Files::open(dir)?;
        let mut bytes = 0usize;
        let mut get = |name: &str, shape: &[usize]| -> Result<CudaSlice<bf16>> {
            let (s, data) = files
                .find(name)?
                .ok_or_else(|| anyhow!("tensor {name} not found"))?;
            anyhow::ensure!(s == shape, "{name}: shape {s:?}, expected {shape:?}");
            bytes += data.len() * 2;
            gpu.upload(&data)
        };
        let (h, i, q, kv, d) = (
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.q_dim(),
            cfg.kv_dim(),
            cfg.head_dim,
        );
        let embed_tokens = get("model.embed_tokens.weight", &[cfg.vocab_size, h])?;
        let norm = get("model.norm.weight", &[h])?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{l}.");
            layers.push(LayerWeights {
                input_layernorm: get(&(p.clone() + "input_layernorm.weight"), &[h])?,
                q_proj: get(&(p.clone() + "self_attn.q_proj.weight"), &[q, h])?,
                k_proj: get(&(p.clone() + "self_attn.k_proj.weight"), &[kv, h])?,
                v_proj: get(&(p.clone() + "self_attn.v_proj.weight"), &[kv, h])?,
                o_proj: get(&(p.clone() + "self_attn.o_proj.weight"), &[h, q])?,
                q_norm: get(&(p.clone() + "self_attn.q_norm.weight"), &[d])?,
                k_norm: get(&(p.clone() + "self_attn.k_norm.weight"), &[d])?,
                post_attention_layernorm: get(
                    &(p.clone() + "post_attention_layernorm.weight"),
                    &[h],
                )?,
                gate_proj: get(&(p.clone() + "mlp.gate_proj.weight"), &[i, h])?,
                up_proj: get(&(p.clone() + "mlp.up_proj.weight"), &[i, h])?,
                down_proj: get(&(p + "mlp.down_proj.weight"), &[h, i])?,
            });
        }
        let lm_head = match files.find("lm_head.weight")? {
            Some((s, data)) => {
                anyhow::ensure!(s == [cfg.vocab_size, h], "lm_head shape {s:?}");
                bytes += data.len() * 2;
                gpu.upload(&data)?
            }
            None => {
                anyhow::ensure!(
                    cfg.tie_word_embeddings,
                    "lm_head.weight missing and embeddings are not tied"
                );
                let mut copy = gpu.alloc::<bf16>(cfg.vocab_size * h)?;
                gpu.stream.memcpy_dtod(&embed_tokens, &mut copy)?;
                copy
            }
        };
        Ok(Self {
            embed_tokens,
            lm_head,
            norm,
            layers,
            bytes,
        })
    }
}

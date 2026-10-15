//! Model config (the subset of Hugging Face `config.json` that Qwen3 needs). Provided.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)?;
        anyhow::ensure!(
            cfg.head_dim == 128,
            "kernels assume head_dim == 128, got {}",
            cfg.head_dim
        );
        anyhow::ensure!(
            cfg.num_attention_heads % cfg.num_key_value_heads == 0,
            "bad GQA head counts"
        );
        Ok(cfg)
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// Bytes of K + V cache per token over all layers (bf16).
    pub fn kv_bytes_per_token(&self) -> usize {
        2 * self.num_hidden_layers * self.kv_dim() * 2
    }
}

/// Default model directory: $MLSYS_MODELS/<name> (set by tools/setup_wsl.sh).
pub fn model_dir(name: &str) -> std::path::PathBuf {
    let root = std::env::var("MLSYS_MODELS")
        .unwrap_or_else(|_| format!("{}/models", std::env::var("HOME").unwrap_or_default()));
    std::path::PathBuf::from(root).join(name)
}

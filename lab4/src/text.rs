//! Tokenizer and the Qwen3 chat template. Provided.

use std::path::Path;

use anyhow::{anyhow, Result};
use tokenizers::Tokenizer;

pub struct Text {
    pub tok: Tokenizer,
    pub eos: Vec<u32>,
}

impl Text {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tok =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        // <|im_end|> ends an assistant turn, <|endoftext|> ends a document.
        let eos = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tok.token_to_id(t))
            .collect();
        Ok(Self { tok, eos })
    }

    pub fn encode(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tok
            .encode(s, false)
            .map_err(|e| anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, true).map_err(|e| anyhow!("{e}"))
    }

    /// One user turn in Qwen3's chat format, with thinking switched off.
    pub fn chat_prompt(&self, user: &str) -> Result<Vec<u32>> {
        self.encode(&format!(
            "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ))
    }
}

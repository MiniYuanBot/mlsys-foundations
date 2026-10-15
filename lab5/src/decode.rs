//! The decoding loops for every method. Provided.

use std::time::Instant;

use anyhow::Result;
use minillm::gpu::Gpu;
use minillm::model_student::Qwen3;

use crate::rng::Rng;
use crate::session::Session;
use crate::spec_student::{draft_greedy, greedy_round, sampling_round, sync_draft, RoundStats};
use crate::tree::tree_round;
use crate::verify_student::ngram_propose;

#[derive(Debug, Clone)]
pub enum Method {
    /// Plain autoregressive decoding with the target (the baseline).
    Plain,
    /// Prompt lookup: draft up to k tokens by n-gram matching (A1 + A3).
    Ngram { n: usize, k: usize },
    /// Greedy draft model chain of length k (A3).
    Draft { k: usize },
    /// Sampling at temperature `temp` with rejection sampling (B1 + B2).
    Sample { k: usize, temp: f32, seed: u64 },
    /// SpecInfer token tree with the given level widths (C1 + C2).
    Tree { widths: Vec<usize> },
}

#[derive(Debug, Clone, Default)]
pub struct Report {
    pub output: Vec<u32>,
    pub secs: f64,
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub target_calls: usize,
    pub draft_calls: usize,
}

impl Report {
    pub fn tokens_per_sec(&self) -> f64 {
        self.output.len() as f64 / self.secs
    }
    /// Average number of tokens produced per target forward pass.
    pub fn tokens_per_round(&self) -> f64 {
        self.output.len() as f64 / self.rounds.max(1) as f64
    }
    pub fn acceptance_rate(&self) -> f64 {
        self.accepted as f64 / self.proposed.max(1) as f64
    }
}

/// Generate `max_new` tokens after `prompt` (never stops early, so every
/// method does the same amount of work). `draft` is needed for Draft, Sample, Tree.
pub fn generate(
    gpu: &Gpu,
    target: &Qwen3,
    draft: Option<&Qwen3>,
    prompt: &[u32],
    max_new: usize,
    method: &Method,
) -> Result<Report> {
    let cap = prompt.len() + max_new + 80;
    let mut t = Session::new(gpu, target, prompt, cap)?;
    let mut d = match draft {
        Some(m) => Some(Session::new(gpu, m, prompt, cap)?),
        None => None,
    };
    let need_draft = !matches!(method, Method::Plain | Method::Ngram { .. });
    anyhow::ensure!(!need_draft || d.is_some(), "{method:?} needs a draft model");
    // Prefill both models (not timed: TTFT is the same for every method).
    let first = t.catch_up(gpu)?.argmax(gpu)?[0];
    t.tokens.push(first);
    if let Some(d) = d.as_mut() {
        d.tokens.clone_from(&t.tokens);
        d.sync_prefix(gpu)?;
    }
    gpu.sync()?;
    let (t_calls0, d_calls0) = (t.forward_calls, d.as_ref().map_or(0, |d| d.forward_calls));
    let mut rng = Rng::new(match method {
        Method::Sample { seed, .. } => *seed,
        _ => 0,
    });
    let start = Instant::now();
    let mut rep = Report::default();
    while t.tokens.len() < prompt.len() + max_new {
        let st: RoundStats = match method {
            Method::Plain => greedy_round(gpu, &mut t, &[])?,
            Method::Ngram { n, k } => {
                let prop = ngram_propose(&t.tokens, *n, *k);
                greedy_round(gpu, &mut t, &prop)?
            }
            Method::Draft { k } => {
                let d = d.as_mut().unwrap();
                let prop = draft_greedy(gpu, d, *k)?;
                let st = greedy_round(gpu, &mut t, &prop)?;
                sync_draft(d, &t);
                st
            }
            Method::Sample { k, temp, .. } => {
                sampling_round(gpu, &mut t, d.as_mut().unwrap(), *k, *temp, &mut rng)?
            }
            Method::Tree { widths } => tree_round(gpu, &mut t, d.as_mut().unwrap(), widths)?,
        };
        rep.rounds += 1;
        rep.proposed += st.proposed;
        rep.accepted += st.accepted;
    }
    gpu.sync()?;
    rep.secs = start.elapsed().as_secs_f64();
    t.tokens.truncate(prompt.len() + max_new);
    rep.output = t.tokens[prompt.len()..].to_vec();
    rep.target_calls = t.forward_calls - t_calls0;
    rep.draft_calls = d.as_ref().map_or(0, |d| d.forward_calls) - d_calls0;
    Ok(rep)
}

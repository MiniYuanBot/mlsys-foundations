//! Per-request state, shared by the block manager, scheduler and engine. Provided.

use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Waiting,
    Running,
    Finished,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: usize,
    /// prompt followed by every generated token
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub max_new_tokens: usize,
    pub stop_tokens: Vec<u32>,
    /// How many leading tokens already have their K/V in the cache.
    /// Invariant: num_computed <= tokens.len(). The scheduler asks for
    /// tokens[num_computed..num_computed + n] to be processed in a step.
    pub num_computed: usize,
    /// Physical cache blocks; block i holds positions i*B .. i*B+B-1.
    pub block_table: Vec<u32>,
    pub status: Status,
    /// Tokens that were served from the prefix cache when the request was admitted (stats).
    pub num_cached_tokens: usize,
    pub num_preemptions: usize,
    pub arrival: Instant,
    pub first_token_at: Option<Instant>,
    pub finished_at: Option<Instant>,
}

impl Sequence {
    pub fn new(id: usize, prompt: Vec<u32>, max_new_tokens: usize, stop_tokens: Vec<u32>) -> Self {
        Self {
            id,
            prompt_len: prompt.len(),
            tokens: prompt,
            max_new_tokens,
            stop_tokens,
            num_computed: 0,
            block_table: Vec::new(),
            status: Status::Waiting,
            num_cached_tokens: 0,
            num_preemptions: 0,
            arrival: Instant::now(),
            first_token_at: None,
            finished_at: None,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn num_generated(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    pub fn output(&self) -> &[u32] {
        &self.tokens[self.prompt_len..]
    }

    /// Tokens not yet in the cache.
    pub fn num_uncomputed(&self) -> usize {
        self.tokens.len() - self.num_computed
    }

    /// Done after appending `tok`? (max_new_tokens reached or a stop token).
    pub fn should_stop(&self, tok: u32) -> bool {
        self.num_generated() >= self.max_new_tokens || self.stop_tokens.contains(&tok)
    }
}

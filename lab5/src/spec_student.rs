//! Speculative rounds on the GPU. Fill in TODO(A3, B2).
//!
//!   cargo test --release --test spec
//!
//! A round turns the session state "tokens = [..., last], num_computed =
//! len - 1" into the same state with 1 + (accepted drafts) more tokens.
//! The hard part is the KV cache: the target computes K/V for the drafted
//! positions too, and the rejected ones must be forgotten (Session::truncate).

use anyhow::Result;
use minillm::gpu::Gpu;

use crate::rng::Rng;
use crate::session::Session;
use crate::verify_student::{greedy_accept, speculative_sample};

/// What one round did (for statistics).
#[derive(Debug, Clone, Copy, Default)]
pub struct RoundStats {
    pub proposed: usize,
    pub accepted: usize,
}

// ------------------------------------------------------------------- A3 ----
// TODO(A3) one greedy round with a given proposal (from A1 or a draft model).
//   1. let n = target.tokens.len(); the target has computed n - 1 tokens.
//   2. feed [last token, proposal...] in ONE call: logits for k + 1 positions
//      (positions n-1 .. n-1+k), then argmax -> target_next (k + 1 tokens)
//   3. (a, bonus) = greedy_accept(proposal, target_next)
//   4. tokens += proposal[..a] + [bonus]
//   5. the KV for positions n-1 .. n-1+a is valid (last token + accepted drafts);
//      the rest is not: target.truncate(n + a)   -> num_computed == new len - 1
// Empty proposal = a normal decode step (feed just [last]).
pub fn greedy_round(gpu: &Gpu, target: &mut Session, proposal: &[u32]) -> Result<RoundStats> {
    let _ = (gpu, target, proposal, greedy_accept as fn(&[u32], &[u32]) -> (usize, u32));
    todo!("A3 greedy_round")
}

/// Provided: k greedy draft tokens from the draft model. The draft session
/// first catches up on any committed tokens it has not seen.
pub fn draft_greedy(gpu: &Gpu, draft: &mut Session, k: usize) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(k);
    let mut tok = draft.catch_up(gpu)?.argmax(gpu)?[0];
    for i in 0..k {
        out.push(tok);
        if i + 1 < k {
            tok = draft.feed(gpu, &[tok])?.argmax(gpu)?[0];
        }
    }
    // The draft cache now also holds out[..k-1]: record them so that
    // sync_draft can tell which cached positions stay valid.
    draft.tokens.extend_from_slice(&out[..k.saturating_sub(1)]);
    Ok(out)
}

/// Provided: keep the draft session consistent with the target after a round.
/// The draft cache holds [committed prefix, its own proposals...]; only the
/// part that matches the committed tokens stays valid.
pub fn sync_draft(draft: &mut Session, target: &Session) {
    let valid = draft.tokens.iter().zip(&target.tokens).take_while(|(a, b)| a == b).count();
    draft.tokens.clone_from(&target.tokens);
    draft.truncate(valid.min(target.tokens.len() - 1));
}

// ------------------------------------------------------------------- B2 ----
// TODO(B2) one SAMPLING round with a draft model at temperature `temp`.
//   1. draft k tokens: d_i ~ q_i, keeping each q_i (the draft's probabilities):
//        q_0 = draft.catch_up(gpu)?.probs(gpu, temp)?[0];  d_0 = rng.sample(&q_0)
//        then for i >= 1: q_i = draft.feed(gpu, &[d_{i-1}])?.probs(..)[0]; d_i = rng.sample(&q_i)
//      (catch_up feeds the committed tokens the draft has not seen yet,
//       including the last one, and returns the next-token distribution)
//   2. target: feed [last, d_0, ..., d_{k-1}] once, p = probs(gpu, temp) (k + 1 rows)
//   3. emitted = speculative_sample(d, q, p, rng); a = emitted.len() - 1 accepted
//   4. tokens += emitted; target.truncate(n + a); sync_draft(draft, target)
pub fn sampling_round(gpu: &Gpu, target: &mut Session, draft: &mut Session, k: usize, temp: f32, rng: &mut Rng) -> Result<RoundStats> {
    let _ = (gpu, target, draft, k, temp, rng, speculative_sample as fn(&[u32], &[Vec<f32>], &[Vec<f32>], &mut Rng) -> Vec<u32>);
    todo!("B2 sampling_round")
}

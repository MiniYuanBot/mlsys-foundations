//! The math of speculative decoding (pure CPU). Fill in TODO(A1, A2, B1, C2).
//!
//!   cargo test --release --test verify        (no GPU needed)
//!
//! A1 proposes draft tokens from the text itself (prompt lookup / n-gram).
//! A2 and C2 decide which draft tokens a GREEDY target accepts.
//! B1 is the rejection-sampling rule that keeps the output distribution
//!    EXACTLY that of the target model (Leviathan et al. / Chen et al. 2023).

use crate::rng::Rng;

// ------------------------------------------------------------------- A1 ----
// TODO(A1) prompt-lookup drafting. Take the last `n` tokens of `tokens` as a
// pattern; find its most recent EARLIER occurrence (one that does not end at
// the last token); propose the up-to-`k` tokens that followed it.
// If the pattern is not found, retry with n-1, n-2, ..., 1; if nothing
// matches, return an empty proposal.
//   tokens = [5 6 7 8 | 5 6], n = 2, k = 3  ->  [7, 8, 5]
// Lookahead / prompt-lookup decoding use exactly this; it costs no model calls
// and works very well on code edits, RAG, and anything that copies the context.
pub fn ngram_propose(tokens: &[u32], n: usize, k: usize) -> Vec<u32> {
    let _ = (tokens, n, k);
    todo!("A1 ngram_propose")
}

// ------------------------------------------------------------------- A2 ----
// TODO(A2) greedy verification of a CHAIN of draft tokens.
// The target was fed [last, d_1, ..., d_k] in one forward pass;
// target_next[i] = argmax of the target's logits at fed position i, i.e. the
// token the target wants AFTER last, d_1, ..., d_i.
// Accept d_1, d_2, ... while d_i == target_next[i-1]. Return
//   (number of accepted drafts a, the bonus token target_next[a]).
// The bonus is free: even if every draft is rejected we still make 1 token.
//   draft [7, 8, 9], target_next [7, 8, 4, 2]  ->  (2, 4)
pub fn greedy_accept(draft: &[u32], target_next: &[u32]) -> (usize, u32) {
    let _ = (draft, target_next);
    todo!("A2 greedy_accept")
}

// ------------------------------------------------------------------- B1 ----
// TODO(B1) speculative sampling.
//   draft[i]      the i-th drafted token (i = 0..k), sampled from q[i]
//   q[i]          the DRAFT model's distribution it was sampled from
//   p[i]          the TARGET's distribution at the same position (k+1 rows:
//                 p[k] is the distribution after all k drafts)
// For i = 0..k:
//   x = draft[i];  accept with probability min(1, p[i][x] / q[i][x])
//     (draw u = rng.uniform(); accept iff u < p[i][x] / q[i][x])
//   if accepted: emit x and continue
//   else: emit a sample from the RESIDUAL  r = max(0, p[i] - q[i]) (renormalised,
//         rng.sample does that) and STOP
// If all k were accepted, emit one more sample from p[k].
// Return the emitted tokens (length a + 1 for a accepted drafts).
// Lecture L5 section 3 proves the output is distributed exactly as the target.
pub fn speculative_sample(draft: &[u32], q: &[Vec<f32>], p: &[Vec<f32>], rng: &mut Rng) -> Vec<u32> {
    let _ = (draft, q, p, rng);
    todo!("B1 speculative_sample")
}

// ------------------------------------------------------------------- C2 ----
// TODO(C2) greedy verification of a TOKEN TREE (SpecInfer).
//   tree_tokens[i], parents[i]: node i (node 0 = root = last committed token;
//                  parents[i] < i)
//   target_next[i]: target argmax after the path root -> ... -> i
// Walk from the root: at node v, the target wants t = target_next[v]; if some
// child c of v has tree_tokens[c] == t, move to c; otherwise stop.
// Return (path of node indices starting with 0, bonus = target_next[last node]).
// The accepted tokens are tree_tokens[path[1..]], then the bonus.
// (If two children carry the same token, take the first one.)
pub fn tree_accept(tree_tokens: &[u32], parents: &[usize], target_next: &[u32]) -> (Vec<usize>, u32) {
    let _ = (tree_tokens, parents, target_next);
    todo!("C2 tree_accept")
}

//! CPU tests of the speculative-decoding math.   cargo test --release --test verify
use specdec::rng::Rng;
use specdec::verify_student::{greedy_accept, ngram_propose, speculative_sample, tree_accept};

#[test]
fn a1_ngram_finds_the_latest_earlier_match() {
    assert_eq!(ngram_propose(&[5, 6, 7, 8, 5, 6], 2, 3), vec![7, 8, 5]);
    // two earlier matches of [1, 2]: the latest one wins
    assert_eq!(ngram_propose(&[1, 2, 9, 1, 2, 4, 4, 1, 2], 2, 2), vec![4, 4]);
    // the proposal is cut at the end of the text
    assert_eq!(ngram_propose(&[3, 1, 2, 3], 1, 5), vec![1, 2, 3]);
}

#[test]
fn a1_ngram_backs_off_and_can_fail() {
    // [8, 9] never occurred before, [9] did
    assert_eq!(ngram_propose(&[9, 4, 4, 8, 9], 2, 2), vec![4, 4]);
    assert!(ngram_propose(&[1, 2, 3, 4], 3, 4).is_empty());
    assert!(ngram_propose(&[7], 3, 4).is_empty());
}

#[test]
fn a2_greedy_accept() {
    assert_eq!(greedy_accept(&[7, 8, 9], &[7, 8, 4, 2]), (2, 4));
    assert_eq!(greedy_accept(&[7, 8, 9], &[7, 8, 9, 2]), (3, 2));
    assert_eq!(greedy_accept(&[7, 8, 9], &[1, 8, 9, 2]), (0, 1));
    assert_eq!(greedy_accept(&[], &[5]), (0, 5));
}

#[test]
fn c2_tree_accept_follows_matching_children() {
    //        0 (root)
    //      /   |   \
    //    1:a  2:b  3:c
    //    / \         |
    //  4:x 5:y      6:z
    let toks = [0, 10, 11, 12, 20, 21, 22];
    let par = [0, 0, 0, 0, 1, 1, 3];
    // after the root the target wants c (12); after node 3 it wants z (22); after 6 it wants 99
    assert_eq!(tree_accept(&toks, &par, &[12, 0, 0, 22, 0, 0, 99]), (vec![0, 3, 6], 99));
    // the target wants something that is not in the tree
    assert_eq!(tree_accept(&toks, &par, &[42, 0, 0, 0, 0, 0, 0]), (vec![0], 42));
    // a chain-shaped tree behaves like greedy_accept
    assert_eq!(tree_accept(&[0, 7, 8, 9], &[0, 0, 1, 2], &[7, 8, 4, 0]), (vec![0, 1, 2], 4));
}

/// B1: speculative sampling must reproduce the TARGET distribution exactly.
/// Small vocabulary, Markov "models": target P[prev][next], draft Q[prev][next].
/// We generate 3-token continuations with speculative rounds and compare the
/// empirical distribution of (t1, t2, t3) with the exact target probability.
#[test]
fn b1_speculative_sampling_is_exact() {
    const V: usize = 4;
    let p = [[0.1f32, 0.6, 0.2, 0.1], [0.5, 0.1, 0.1, 0.3], [0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]];
    let q = [[0.4f32, 0.2, 0.2, 0.2], [0.1, 0.1, 0.7, 0.1], [0.6, 0.2, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]];
    let mut rng = Rng::new(42);
    let trials = 200_000;
    let mut counts = vec![0usize; V * V * V];
    let (mut proposed, mut accepted) = (0usize, 0usize);
    for _ in 0..trials {
        let mut seq = vec![0u32]; // start token
        while seq.len() < 4 {
            let k = 2;
            let (mut draft, mut qs) = (Vec::new(), Vec::new());
            let mut prev = *seq.last().unwrap() as usize;
            for _ in 0..k {
                let d = rng.sample(&q[prev]) as u32;
                qs.push(q[prev].to_vec());
                draft.push(d);
                prev = d as usize;
            }
            // the target's distribution after each drafted prefix
            let mut ps = vec![p[*seq.last().unwrap() as usize].to_vec()];
            for &d in &draft {
                ps.push(p[d as usize].to_vec());
            }
            let out = speculative_sample(&draft, &qs, &ps, &mut rng);
            assert!(!out.is_empty() && out.len() <= k + 1);
            assert_eq!(&out[..out.len() - 1], &draft[..out.len() - 1], "accepted tokens must be a prefix of the draft");
            proposed += k;
            accepted += out.len() - 1;
            seq.extend(out);
        }
        let (a, b, c) = (seq[1] as usize, seq[2] as usize, seq[3] as usize);
        counts[(a * V + b) * V + c] += 1;
    }
    let mut worst = 0f64;
    for a in 0..V {
        for b in 0..V {
            for c in 0..V {
                let exact = (p[0][a] * p[a][b] * p[b][c]) as f64;
                let got = counts[(a * V + b) * V + c] as f64 / trials as f64;
                worst = worst.max((got - exact).abs());
            }
        }
    }
    eprintln!("max |empirical - exact| = {worst:.4}, acceptance rate {:.3}", accepted as f64 / proposed as f64);
    assert!(worst < 0.006, "output distribution differs from the target's (max error {worst})");
}

#[test]
fn b1_identical_models_accept_everything() {
    let p = vec![vec![0.2f32, 0.3, 0.5]; 4];
    let mut rng = Rng::new(7);
    for _ in 0..1000 {
        let draft = vec![rng.sample(&p[0]) as u32, rng.sample(&p[1]) as u32, rng.sample(&p[2]) as u32];
        let out = speculative_sample(&draft, &p[..3], &p, &mut rng);
        assert_eq!(out.len(), 4);
        assert_eq!(&out[..3], &draft[..]);
    }
}

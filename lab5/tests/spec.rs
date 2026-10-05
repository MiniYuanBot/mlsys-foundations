//! GPU tests: tree attention and end-to-end speculative decoding.
//!   cargo test --release --test spec
//! Greedy speculative decoding must produce the SAME text as plain decoding
//! (up to bf16 near-ties, see the README); sampling must at least run and accept.
use std::sync::{Mutex, MutexGuard, OnceLock};

use half::bf16;
use minillm::batch::{BatchInput, DeviceBatch};
use minillm::config::model_dir;
use minillm::gpu::Gpu;
use minillm::model_student::Qwen3;
use minillm::ops::{self, AttnArgs, AttnImpl};
use minillm::text::Text;
use specdec::decode::{generate, Method};
use specdec::ops5;

fn lock() -> MutexGuard<'static, ()> {
    static L: Mutex<()> = Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}

fn randn(n: usize, seed: u64) -> Vec<bf16> {
    let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            bf16::from_f32(((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32)
        })
        .collect()
}

/// C1: a chain-shaped tree must give exactly lab4's causal attention, and a
/// branching tree must equal causal attention over each node's own path.
#[test]
fn c1_tree_attention_matches_causal_attention_on_paths() {
    let _g = lock();
    let gpu = ops5::gpu().unwrap();
    let (hq, hk, bs, prefix) = (16usize, 8usize, 16usize, 37usize);
    //         0
    //       /   \
    //      1     2
    //     / \     \
    //    3   4     5
    let parents = [0usize, 0, 0, 1, 1, 2];
    let n = parents.len();
    let mut masks = vec![0u64; n];
    for i in 0..n {
        masks[i] = 1 << i;
        if i > 0 {
            masks[i] |= masks[parents[i]];
        }
    }
    let blocks: Vec<u32> = (0..8).collect();
    let cache_n = blocks.len() * bs * hk * 128;
    let (kc, vc) = (gpu.upload(&randn(cache_n, 1)).unwrap(), gpu.upload(&randn(cache_n, 2)).unwrap());
    let q = gpu.upload(&randn(n * hq * 128, 3)).unwrap();
    let tables = gpu.upload(&blocks.iter().map(|&b| b as i32).collect::<Vec<_>>()).unwrap();
    let dmask = gpu.upload(&masks).unwrap();
    let dummy = gpu.upload(&vec![0i32; n]).unwrap();
    let mut tree_out = gpu.alloc::<bf16>(n * hq * 128).unwrap();
    let args = AttnArgs { block_tables: &tables, token_seq: &dummy, token_pos: &dummy, t: n, heads: hq, kv_heads: hk, max_blocks: blocks.len(), block_size: bs, tree: Some((&dmask, prefix)) };
    ops::paged_attention(&gpu, AttnImpl::Fast, &q, &kc, &vc, &mut tree_out, &args).unwrap();
    let tree_out = gpu.download(&tree_out).unwrap();

    // Reference: for each node, rebuild a causal layout where its path follows
    // the prefix contiguously, and run lab4's causal kernel for that one query.
    let kh = gpu.download(&kc).unwrap();
    let vh = gpu.download(&vc).unwrap();
    let row = hk * 128;
    for i in 0..n {
        let mut path = vec![i];
        while *path.last().unwrap() != 0 {
            let p = parents[*path.last().unwrap()];
            path.push(p);
        }
        path.reverse();
        let (mut k2, mut v2) = (kh.clone(), vh.clone());
        for (d, &j) in path.iter().enumerate() {
            let (src, dst) = ((prefix + j) * row, (prefix + d) * row);
            k2[dst..dst + row].copy_from_slice(&kh[src..src + row]);
            v2[dst..dst + row].copy_from_slice(&vh[src..src + row]);
        }
        let b = BatchInput::build(&[(&[0], prefix + path.len() - 1, &blocks, true)], bs);
        let db = DeviceBatch::upload(&gpu, &b).unwrap();
        let qi = gpu.upload(&gpu.download(&q).unwrap()[i * hq * 128..(i + 1) * hq * 128].to_vec()).unwrap();
        let mut out = gpu.alloc::<bf16>(hq * 128).unwrap();
        let a = AttnArgs { block_tables: &db.block_tables, token_seq: &db.token_seq, token_pos: &db.positions, t: 1, heads: hq, kv_heads: hk, max_blocks: db.max_blocks, block_size: bs, tree: None };
        ops::paged_attention(&gpu, AttnImpl::Fast, &qi, &gpu.upload(&k2).unwrap(), &gpu.upload(&v2).unwrap(), &mut out, &a).unwrap();
        let want = gpu.download(&out).unwrap();
        for (x, y) in tree_out[i * hq * 128..(i + 1) * hq * 128].iter().zip(&want) {
            assert!((x.to_f32() - y.to_f32()).abs() <= 1e-2, "node {i}: tree attention differs from causal attention over its path");
        }
    }
}

struct Models {
    gpu: Gpu,
    target: Qwen3,
    draft: Qwen3,
    prompt: Vec<u32>,
}

fn models() -> &'static Models {
    static M: OnceLock<Models> = OnceLock::new();
    M.get_or_init(|| {
        let gpu = ops5::gpu().unwrap();
        let target = Qwen3::load(&gpu, &model_dir("Qwen3-1.7B")).expect("Qwen3-1.7B (run tools/setup_wsl.sh)");
        let draft = Qwen3::load(&gpu, &model_dir("Qwen3-0.6B")).unwrap();
        let text = Text::load(&model_dir("Qwen3-1.7B")).unwrap();
        let prompt = text.chat_prompt("List the steps of a TCP handshake, then explain each in one sentence.").unwrap();
        Models { gpu, target, draft, prompt }
    })
}

fn agreement(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn check_greedy(method: Method) {
    let _g = lock();
    let m = models();
    let n = 64;
    let plain = generate(&m.gpu, &m.target, None, &m.prompt, n, &Method::Plain).unwrap();
    let spec = generate(&m.gpu, &m.target, Some(&m.draft), &m.prompt, n, &method).unwrap();
    let same = agreement(&plain.output, &spec.output);
    eprintln!("{method:?}: {same}/{n} tokens identical to plain decoding, acceptance {:.2}, {:.2} tokens/target call",
        spec.acceptance_rate(), spec.output.len() as f64 / spec.target_calls as f64);
    // Verifying k+1 tokens in one GEMM rounds differently from k+1 single
    // steps, so a bf16 near-tie may flip late; the bulk must agree.
    assert!(same >= 32, "{method:?}: only {same} leading tokens agree with plain greedy decoding");
    assert!(spec.target_calls < plain.target_calls, "{method:?}: no target calls were saved");
}

#[test]
fn a3_ngram_greedy_is_lossless() {
    check_greedy(Method::Ngram { n: 3, k: 4 });
}

#[test]
fn a3_draft_greedy_is_lossless() {
    check_greedy(Method::Draft { k: 3 });
}

#[test]
fn c_tree_greedy_is_lossless() {
    check_greedy(Method::Tree { widths: vec![2, 2, 1] });
}

#[test]
fn b2_sampling_runs_and_accepts() {
    let _g = lock();
    let m = models();
    let r = generate(&m.gpu, &m.target, Some(&m.draft), &m.prompt, 48, &Method::Sample { k: 3, temp: 0.8, seed: 3 }).unwrap();
    eprintln!("sampling: acceptance {:.2}, {} target calls for {} tokens", r.acceptance_rate(), r.target_calls, r.output.len());
    assert_eq!(r.output.len(), 48);
    assert!(r.acceptance_rate() > 0.2, "acceptance rate {:.2} is suspiciously low", r.acceptance_rate());
    assert!(r.target_calls < 48);
}

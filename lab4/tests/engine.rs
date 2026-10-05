//! End-to-end engine behaviour on the GPU.   cargo test --release --test engine
mod common;

use minillm::config::model_dir;
use minillm::engine::{Engine, EngineConfig};
use minillm::scheduler_student::SchedulerConfig;
use minillm::text::Text;

const PROMPTS: &[&str] = &[
    "Explain continuous batching in one sentence.",
    "List three prime numbers.",
    "What does a GPU warp consist of?",
    "Translate 'good morning' into French.",
    "Write a haiku about memory bandwidth.",
    "Why is decoding memory-bound?",
];

fn engine(cfg: EngineConfig) -> Engine {
    Engine::new(&model_dir("Qwen3-0.6B"), cfg).expect("engine (out of GPU memory? close other GPU programs)")
}

fn prompts() -> Vec<Vec<u32>> {
    let t = Text::load(&model_dir("Qwen3-0.6B")).unwrap();
    PROMPTS.iter().map(|p| t.chat_prompt(p).unwrap()).collect()
}

/// Fraction of the first `n` tokens that agree position by position.
fn agreement(a: &[u32], b: &[u32], n: usize) -> f64 {
    a.iter().zip(b).take(n).filter(|(x, y)| x == y).count() as f64 / n as f64
}

#[test]
fn b3_batched_generation_matches_one_at_a_time() {
    let _gpu = common::gpu_lock();
    let ps = prompts();
    let n = 32;
    let mut e = engine(EngineConfig { kv_cache_bytes: 512 << 20, prefix_caching: false, ..Default::default() });
    let solo: Vec<Vec<u32>> = ps.iter().map(|p| e.generate(p, n, &[]).unwrap()).collect();
    let mut e = engine(EngineConfig { kv_cache_bytes: 512 << 20, prefix_caching: false, ..Default::default() });
    let ids: Vec<usize> = ps.iter().map(|p| e.add_request(p.clone(), n, vec![])).collect();
    e.run_to_completion().unwrap();
    // Different batch sizes pick different GEMM algorithms, so bf16 rounding
    // differs slightly and a near-tie can flip late in a sequence. Require
    // high agreement, not bit equality.
    let rates: Vec<f64> = ids.iter().zip(&solo).map(|(id, s)| agreement(e.seqs[id].output(), s, n)).collect();
    eprintln!("agreement per prompt: {rates:?}");
    assert!(rates.iter().sum::<f64>() / rates.len() as f64 >= 0.9);
    assert!(e.stats.steps < (n + 2) * 2, "requests must share steps (got {} steps)", e.stats.steps);
}

#[test]
fn c1_chunked_prefill_gives_the_same_text() {
    let _gpu = common::gpu_lock();
    let ps = prompts();
    let long: Vec<u32> = ps.iter().flatten().cloned().collect(); // ~150 tokens
    let mut a = engine(EngineConfig { prefix_caching: false, kv_cache_bytes: 512 << 20, ..Default::default() });
    let whole = a.generate(&long, 24, &[]).unwrap();
    let mut b = engine(EngineConfig {
        prefix_caching: false,
        kv_cache_bytes: 512 << 20,
        sched: SchedulerConfig { max_batched_tokens: 32, ..Default::default() },
        ..Default::default()
    });
    let chunked = b.generate(&long, 24, &[]).unwrap();
    assert!(agreement(&whole, &chunked, 24) >= 0.9, "{whole:?}\nvs\n{chunked:?}");
}

#[test]
fn c2_prefix_cache_hits_and_gives_the_same_text() {
    let _gpu = common::gpu_lock();
    let t = Text::load(&model_dir("Qwen3-0.6B")).unwrap();
    let system = "You are a concise assistant for a course on machine learning systems. ".repeat(6);
    let ps: Vec<Vec<u32>> = PROMPTS.iter().map(|p| t.chat_prompt(&format!("{system}{p}")).unwrap()).collect();
    // Two engines at once: keep their KV caches small (two copies of the weights already take 3 GB).
    let mut off = engine(EngineConfig { prefix_caching: false, kv_cache_bytes: 512 << 20, ..Default::default() });
    let mut on = engine(EngineConfig { prefix_caching: true, kv_cache_bytes: 512 << 20, ..Default::default() });
    let mut agree = 0.0;
    for p in &ps {
        let x = off.generate(p, 16, &[]).unwrap();
        let y = on.generate(p, 16, &[]).unwrap();
        agree += agreement(&x, &y, 16);
    }
    let cached: usize = on.seqs.values().map(|s| s.num_cached_tokens).sum();
    eprintln!("prefix cache: {cached} prompt tokens reused, hit rate {:.2}", on.kv.stats.prefix_hits as f64 / on.kv.stats.prefix_queries as f64);
    assert!(cached >= (ps.len() - 1) * 64, "the shared system prompt should be reused");
    assert!(agree / ps.len() as f64 >= 0.9);
}

#[test]
fn c_preemption_under_a_tiny_cache_still_finishes() {
    let _gpu = common::gpu_lock();
    let ps = prompts();
    // ~20 blocks of 16 tokens: far too small for 6 sequences of ~100 tokens at once.
    let bytes = 20 * 16 * 2 * 28 * 8 * 128 * 2;
    let mut e = engine(EngineConfig { kv_cache_bytes: bytes, prefix_caching: false, ..Default::default() });
    let ids: Vec<usize> = ps.iter().map(|p| e.add_request(p.clone(), 64, vec![])).collect();
    e.run_to_completion().unwrap();
    assert!(e.stats.preemptions > 0, "expected preemptions with a tiny cache");
    for id in ids {
        assert_eq!(e.seqs[&id].num_generated(), 64);
    }
}

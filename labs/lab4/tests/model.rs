//! The whole model against Hugging Face transformers (fp32 reference).
//! Run `python lab4/scripts/make_reference.py` once first.
//!   cargo test --release --test model
//! Needs only A1 + A2: the model runs alone on a fixed block table with the
//! provided naive attention (no Engine, BlockManager, Scheduler or B2 kernel).
mod common;

use common::*;
use minillm::batch::{BatchInput, DeviceBatch};
use minillm::config::model_dir;
use minillm::gpu::Gpu;
use minillm::kvcache::KvCache;
use minillm::model_student::Qwen3;
use minillm::ops;
use serde::Deserialize;

#[derive(Deserialize)]
struct RefEntry {
    name: String,
    ids: Vec<u32>,
    greedy: Vec<u32>,
    vocab: usize,
}

fn load_ref(model: &str) -> Option<Vec<(RefEntry, Vec<f32>)>> {
    let dir = ref_dir(model);
    let Ok(text) = std::fs::read_to_string(dir.join("index.json")) else {
        eprintln!(
            "SKIP: no reference in {} -- run `python lab4/scripts/make_reference.py` first",
            dir.display()
        );
        return None;
    };
    let entries: Vec<RefEntry> = serde_json::from_str(&text).unwrap();
    Some(
        entries
            .into_iter()
            .map(|e| {
                let raw = std::fs::read(dir.join(format!("{}.logits.f32", e.name))).unwrap();
                let logits = raw
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                (e, logits)
            })
            .collect(),
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

#[test]
fn a2_teacher_forced_logits_match_hf() {
    let _gpu = common::gpu_lock();
    let Some(refs) = load_ref("Qwen3-0.6B") else {
        return;
    };
    let gpu = Gpu::new().unwrap();
    let model = Qwen3::load(&gpu, &model_dir("Qwen3-0.6B")).unwrap();
    for (e, want) in &refs {
        let mut cache = KvCache::new(&gpu, &model.cfg, e.ids.len().div_ceil(16), 16).unwrap();
        let batch = BatchInput::single_sequence(&e.ids, 16);
        let db = DeviceBatch::upload(&gpu, &batch).unwrap();
        let logits = model.forward(&gpu, &db, &mut cache).unwrap();
        let n = e.ids.len() * e.vocab;
        let mut f = gpu.alloc::<f32>(n).unwrap();
        ops::bf16_to_f32(&gpu, &logits, &mut f, n).unwrap();
        let got = gpu.download(&f).unwrap();
        let (mut worst_cos, mut agree) = (1.0f64, 0usize);
        for t in 0..e.ids.len() {
            let (g, w) = (
                &got[t * e.vocab..(t + 1) * e.vocab],
                &want[t * e.vocab..(t + 1) * e.vocab],
            );
            worst_cos = worst_cos.min(cosine(g, w));
            agree += (argmax(g) == argmax(w)) as usize;
        }
        let rate = agree as f64 / e.ids.len() as f64;
        eprintln!(
            "{}: {} positions, min cosine {:.5}, top-1 agreement {:.3}",
            e.name,
            e.ids.len(),
            worst_cos,
            rate
        );
        assert!(
            worst_cos > 0.99,
            "{}: logits diverge from HF (min cosine {worst_cos:.4})",
            e.name
        );
        assert!(rate >= 0.9, "{}: top-1 agreement {rate:.3} < 0.9", e.name);
    }
}

/// Greedy decoding with the model alone: one prefill step, then one decode step
/// per token, on a fixed block table 0, 1, 2, ... (no Engine, scheduler or
/// block manager, which are Part B).
fn greedy(gpu: &Gpu, model: &Qwen3, prompt: &[u32], n: usize) -> Vec<u32> {
    let bs = 16;
    let num_blocks = (prompt.len() + n).div_ceil(bs);
    let mut cache = KvCache::new(gpu, &model.cfg, num_blocks, bs).unwrap();
    let table: Vec<u32> = (0..num_blocks as u32).collect();
    let mut tokens = prompt.to_vec();
    let mut computed = 0;
    while tokens.len() < prompt.len() + n {
        let batch = BatchInput::build(&[(&tokens[computed..], computed, &table, true)], bs);
        let db = DeviceBatch::upload(gpu, &batch).unwrap();
        let logits = model.forward(gpu, &db, &mut cache).unwrap();
        let mut next = gpu.alloc::<i32>(1).unwrap();
        ops::argmax_rows(gpu, &logits, &mut next, 1, model.cfg.vocab_size).unwrap();
        computed = tokens.len();
        tokens.push(gpu.download(&next).unwrap()[0] as u32);
    }
    tokens.split_off(prompt.len())
}

#[test]
fn a2_greedy_generation_matches_hf() {
    let _gpu = common::gpu_lock();
    let Some(refs) = load_ref("Qwen3-0.6B") else {
        return;
    };
    let gpu = Gpu::new().unwrap();
    let model = Qwen3::load(&gpu, &model_dir("Qwen3-0.6B")).unwrap();
    for (e, _) in &refs {
        let got = greedy(&gpu, &model, &e.ids, e.greedy.len());
        let same = got
            .iter()
            .zip(&e.greedy)
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "{}: first {same} of {} greedy tokens identical to HF",
            e.name,
            e.greedy.len()
        );
        // bf16 vs fp32 can flip a near-tie; the first several tokens must agree.
        assert!(same >= 8, "{}: only {same} leading tokens match HF", e.name);
    }
}

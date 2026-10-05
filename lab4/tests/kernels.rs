//! Kernel tests: each GPU kernel against a CPU reference that rounds to bf16
//! at the same points as Hugging Face.   cargo test --release --test kernels a1
mod common;

use common::*;
use half::bf16;
use minillm::batch::{BatchInput, DeviceBatch};
use minillm::gpu::Gpu;
use minillm::ops::{self, AttnArgs, AttnImpl};

fn gpu() -> Gpu {
    Gpu::new().expect("GPU init (run inside WSL; are the kernels compiling?)")
}

#[test]
fn a1a_embedding() {
    let g = gpu();
    let (vocab, h) = (50, 64);
    let table = randn(vocab * h, 1, 1.0);
    let tokens = vec![3i32, 0, 49, 3];
    let mut out = g.alloc::<bf16>(tokens.len() * h).unwrap();
    ops::embedding(&g, &g.upload(&to_bf16(&table)).unwrap(), &g.upload(&tokens).unwrap(), &mut out, tokens.len(), h).unwrap();
    let got = to_f32(&g.download(&out).unwrap());
    for (t, &tok) in tokens.iter().enumerate() {
        assert_eq!(&got[t * h..(t + 1) * h], &table[tok as usize * h..(tok as usize + 1) * h]);
    }
}

fn rmsnorm_ref(x: &[f32], w: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    x.chunks(dim)
        .flat_map(|row| {
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
            row.iter().zip(w).map(move |(v, wi)| b(b(v * inv) * wi)).collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn a1b_rmsnorm() {
    let g = gpu();
    for (rows, dim) in [(7usize, 128usize), (5, 1024), (3, 2048), (1, 100)] {
        let x = randn(rows * dim, 2, 3.0);
        let w = randn(dim, 3, 1.0);
        let mut out = g.alloc::<bf16>(rows * dim).unwrap();
        ops::rmsnorm(&g, &g.upload(&to_bf16(&x)).unwrap(), &g.upload(&to_bf16(&w)).unwrap(), &mut out, rows, dim, 1e-6).unwrap();
        // One bf16 ulp of slack: the GPU sums in a different order.
        assert_close("rmsnorm", &to_f32(&g.download(&out).unwrap()), &rmsnorm_ref(&x, &w, dim, 1e-6), 1e-2, 1e-2);
    }
}

#[test]
fn a1c_fused_add_rmsnorm() {
    let g = gpu();
    let (rows, dim) = (9, 1024);
    let x = randn(rows * dim, 4, 2.0);
    let r = randn(rows * dim, 5, 2.0);
    let w = randn(dim, 6, 1.0);
    let sum: Vec<f32> = x.iter().zip(&r).map(|(a, c)| b(a + c)).collect();
    let mut dx = g.upload(&to_bf16(&x)).unwrap();
    let mut dr = g.upload(&to_bf16(&r)).unwrap();
    ops::fused_add_rmsnorm(&g, &mut dx, &mut dr, &g.upload(&to_bf16(&w)).unwrap(), rows, dim, 1e-6).unwrap();
    assert_eq!(to_f32(&g.download(&dr).unwrap()), sum, "residual must become x + residual (rounded to bf16)");
    assert_close("fused_add_rmsnorm", &to_f32(&g.download(&dx).unwrap()), &rmsnorm_ref(&sum, &w, dim, 1e-6), 1e-2, 1e-2);
}

fn rope_ref(x: &mut [f32], heads: usize, pos: usize, theta: f32) {
    for h in 0..heads {
        let v = &mut x[h * 128..(h + 1) * 128];
        for i in 0..64 {
            let inv = 1.0 / theta.powf(2.0 * i as f32 / 128.0);
            let ang = pos as f32 * inv;
            let (c, s) = (b(ang.cos()), b(ang.sin()));
            let (x1, x2) = (v[i], v[i + 64]);
            v[i] = b(b(x1 * c) - b(x2 * s));
            v[i + 64] = b(b(x2 * c) + b(x1 * s));
        }
    }
}

#[test]
fn a1d_rope() {
    let g = gpu();
    let (t, hq, hk, theta) = (4usize, 16usize, 8usize, 1_000_000.0f32);
    let positions = vec![0i32, 1, 17, 3000];
    let mut q = randn(t * hq * 128, 7, 2.0);
    let mut k = randn(t * hk * 128, 8, 2.0);
    let mut dq = g.upload(&to_bf16(&q)).unwrap();
    let mut dk = g.upload(&to_bf16(&k)).unwrap();
    ops::rope(&g, &mut dq, &mut dk, &g.upload(&positions).unwrap(), t, hq, hk, theta).unwrap();
    for (i, &p) in positions.iter().enumerate() {
        rope_ref(&mut q[i * hq * 128..(i + 1) * hq * 128], hq, p as usize, theta);
        rope_ref(&mut k[i * hk * 128..(i + 1) * hk * 128], hk, p as usize, theta);
    }
    // position 0 must be the identity
    assert_close("rope q", &to_f32(&g.download(&dq).unwrap()), &q, 2e-2, 1e-2);
    assert_close("rope k", &to_f32(&g.download(&dk).unwrap()), &k, 2e-2, 1e-2);
}

#[test]
fn a1e_silu_mul() {
    let g = gpu();
    let n = 100_003;
    let gate = randn(n, 9, 6.0);
    let up = randn(n, 10, 2.0);
    let mut out = g.alloc::<bf16>(n).unwrap();
    ops::silu_mul(&g, &g.upload(&to_bf16(&gate)).unwrap(), &g.upload(&to_bf16(&up)).unwrap(), &mut out, n).unwrap();
    let want: Vec<f32> = gate.iter().zip(&up).map(|(&x, &u)| b(b(x / (1.0 + (-x).exp())) * u)).collect();
    assert_close("silu_mul", &to_f32(&g.download(&out).unwrap()), &want, 1e-2, 1e-2);
}

#[test]
fn a1f_argmax() {
    let g = gpu();
    let (rows, cols) = (5usize, 151_936usize);
    let mut x = randn(rows * cols, 11, 4.0);
    let want: Vec<i32> = vec![123, 0, 151_935, 77_777, 5];
    for (r, &w) in want.iter().enumerate() {
        x[r * cols + w as usize] = 100.0;
    }
    x[4 * cols + 9] = 100.0; // tie in row 4: the smaller index (5) wins
    let mut out = g.alloc::<i32>(rows).unwrap();
    ops::argmax_rows(&g, &g.upload(&to_bf16(&x)).unwrap(), &mut out, rows, cols).unwrap();
    assert_eq!(g.download(&out).unwrap(), want);
}

/// CPU paged attention, fp32 math on bf16 inputs.
#[allow(clippy::too_many_arguments)]
fn attention_ref(q: &[f32], kc: &[f32], vc: &[f32], b: &BatchInput, hq: usize, hk: usize, bs: usize) -> Vec<f32> {
    let mut out = vec![0f32; q.len()];
    let scale = 1.0 / (128f32).sqrt();
    for t in 0..b.tokens.len() {
        let table = &b.block_tables[b.token_seq[t] as usize * b.max_blocks..];
        for h in 0..hq {
            let kvh = h / (hq / hk);
            let qv = &q[(t * hq + h) * 128..][..128];
            let n = b.positions[t] as usize + 1;
            let idx = |j: usize| ((table[j / bs] as usize * bs + j % bs) * hk + kvh) * 128;
            let s: Vec<f32> = (0..n).map(|j| (0..128).map(|d| qv[d] * kc[idx(j) + d]).sum::<f32>() * scale).collect();
            let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
            let l: f32 = p.iter().sum();
            for d in 0..128 {
                out[(t * hq + h) * 128 + d] = (0..n).map(|j| p[j] * vc[idx(j) + d]).sum::<f32>() / l;
            }
        }
    }
    out
}

fn attention_case(which: AttnImpl) {
    let g = gpu();
    let (hq, hk, bs, num_blocks) = (16usize, 8usize, 16usize, 40usize);
    // Three sequences with scattered physical blocks: a 37-token prefill,
    // a decode token at position 200, and a 5-token chunk at positions 60..65.
    let tables: Vec<Vec<u32>> = vec![vec![7, 3, 11], (12..25).rev().collect(), vec![30, 2, 31, 1, 39]];
    let toks_a: Vec<u32> = (0..37).collect();
    let toks_c: Vec<u32> = (0..5).collect();
    let batch = BatchInput::build(&[(&toks_a, 0, &tables[0], true), (&[9], 200, &tables[1], true), (&toks_c, 60, &tables[2], true)], bs);
    let t = batch.num_tokens();
    let cache_n = num_blocks * bs * hk * 128;
    let (kc, vc) = (randn(cache_n, 12, 1.0), randn(cache_n, 13, 1.0));
    let q = randn(t * hq * 128, 14, 1.0);
    let db = DeviceBatch::upload(&g, &batch).unwrap();
    let mut out = g.alloc::<bf16>(t * hq * 128).unwrap();
    let args = AttnArgs { block_tables: &db.block_tables, token_seq: &db.token_seq, token_pos: &db.positions, t, heads: hq, kv_heads: hk, max_blocks: db.max_blocks, block_size: bs };
    ops::paged_attention(&g, which, &g.upload(&to_bf16(&q)).unwrap(), &g.upload(&to_bf16(&kc)).unwrap(), &g.upload(&to_bf16(&vc)).unwrap(), &mut out, &args).unwrap();
    let want = attention_ref(&q, &kc, &vc, &batch, hq, hk, bs);
    assert_close(&format!("paged_attention {which:?}"), &to_f32(&g.download(&out).unwrap()), &want, 1e-2, 1e-2);
}

#[test]
fn b0_paged_attention_naive_reference() {
    attention_case(AttnImpl::Naive);
}

#[test]
fn b2_paged_attention_fast() {
    attention_case(AttnImpl::Fast);
}

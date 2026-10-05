//! Shared test helpers.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use half::bf16;

pub fn b(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

pub fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

pub fn to_f32(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// Deterministic pseudo-random numbers in [-scale, scale], already bf16-exact.
pub fn randn(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            b(((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale)
        })
        .collect()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Elements of `got` that differ from `want` by more than `atol + rtol*|want|`.
pub fn assert_close(what: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite() && (g - w).abs() <= atol + rtol * w.abs(), "{what}: element {i}: got {g}, want {w}");
    }
}

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

pub fn ref_dir(model: &str) -> PathBuf {
    std::env::var("MLSYS_LAB4_REF").map(PathBuf::from).unwrap_or_else(|_| repo_root().join("build/lab4/ref")).join(model)
}

/// GPU tests that load the model take this lock: cargo runs tests in parallel
/// threads and two or three copies of the weights do not fit in 8 GB.
pub fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

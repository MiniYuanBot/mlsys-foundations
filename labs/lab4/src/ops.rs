//! Thin, typed wrappers around cuBLAS and the kernels. Provided.
//! Every function enqueues work on the single stream and returns immediately.

use anyhow::Result;
use cudarc::cublas::{sys::cublasOperation_t, Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use half::bf16;

use crate::gpu::Gpu;

pub type Buf = CudaSlice<bf16>;

fn grid(n: usize, block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// y[T, out] = x[T, in] @ w[out, in]^T   (bf16 in/out, fp32 accumulate).
///
/// cuBLAS is column-major. Row-major x [T, in] is column-major x^T (in x T);
/// row-major w [out, in] is column-major w^T (in x out). We want column-major
/// y^T (out x T) = w x^T = op(w^T) * x^T with op = transpose. So:
/// m = out, n = T, k = in, A = w (transposed, lda = in), B = x (lda = in), C = y (ldc = out).
pub fn linear(
    gpu: &Gpu,
    x: &Buf,
    w: &Buf,
    y: &mut Buf,
    t: usize,
    d_in: usize,
    d_out: usize,
) -> Result<()> {
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: d_out as i32,
        n: t as i32,
        k: d_in as i32,
        alpha: bf16::ONE,
        lda: d_in as i32,
        ldb: d_in as i32,
        beta: bf16::ZERO,
        ldc: d_out as i32,
    };
    unsafe { gpu.blas.gemm(cfg, w, x, y)? };
    Ok(())
}

pub fn embedding(
    gpu: &Gpu,
    table: &Buf,
    tokens: &CudaSlice<i32>,
    out: &mut Buf,
    t: usize,
    hidden: usize,
) -> Result<()> {
    let (ti, hi) = (t as i32, hidden as i32);
    let mut b = gpu.stream.launch_builder(gpu.func("embedding"));
    b.arg(table).arg(tokens).arg(out).arg(&ti).arg(&hi);
    unsafe { b.launch(grid(t, 256))? };
    Ok(())
}

/// out = rmsnorm(x) * w, row by row. `rows` rows of `dim` elements.
pub fn rmsnorm(
    gpu: &Gpu,
    x: &Buf,
    w: &Buf,
    out: &mut Buf,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    let (ri, di) = (rows as i32, dim as i32);
    let mut b = gpu.stream.launch_builder(gpu.func("rmsnorm"));
    b.arg(x).arg(w).arg(out).arg(&ri).arg(&di).arg(&eps);
    unsafe { b.launch(grid(rows, if dim <= 128 { 128 } else { 256 }))? };
    Ok(())
}

/// In place: residual = x + residual; x = rmsnorm(residual) * w.
pub fn fused_add_rmsnorm(
    gpu: &Gpu,
    x: &mut Buf,
    residual: &mut Buf,
    w: &Buf,
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    let (ri, di) = (rows as i32, dim as i32);
    let mut b = gpu.stream.launch_builder(gpu.func("fused_add_rmsnorm"));
    b.arg(x).arg(residual).arg(w).arg(&ri).arg(&di).arg(&eps);
    unsafe { b.launch(grid(rows, 256))? };
    Ok(())
}

pub fn rope(
    gpu: &Gpu,
    q: &mut Buf,
    k: &mut Buf,
    positions: &CudaSlice<i32>,
    t: usize,
    heads: usize,
    kv_heads: usize,
    theta: f32,
) -> Result<()> {
    let (ti, hi, ki) = (t as i32, heads as i32, kv_heads as i32);
    let mut b = gpu.stream.launch_builder(gpu.func("rope"));
    b.arg(q)
        .arg(k)
        .arg(positions)
        .arg(&ti)
        .arg(&hi)
        .arg(&ki)
        .arg(&theta);
    unsafe { b.launch(grid(t, 256))? };
    Ok(())
}

pub fn silu_mul(gpu: &Gpu, gate: &Buf, up: &Buf, out: &mut Buf, n: usize) -> Result<()> {
    let ni = n as i64;
    let mut b = gpu.stream.launch_builder(gpu.func("silu_mul"));
    b.arg(gate).arg(up).arg(out).arg(&ni);
    unsafe { b.launch(grid(n.div_ceil(256).min(65535), 256))? };
    Ok(())
}

pub fn argmax_rows(
    gpu: &Gpu,
    logits: &Buf,
    out: &mut CudaSlice<i32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let (ri, ci) = (rows as i32, cols as i32);
    let mut b = gpu.stream.launch_builder(gpu.func("argmax_rows"));
    b.arg(logits).arg(out).arg(&ri).arg(&ci);
    unsafe { b.launch(grid(rows, 1024))? };
    Ok(())
}

pub fn reshape_and_cache(
    gpu: &Gpu,
    k: &Buf,
    v: &Buf,
    k_cache: &mut Buf,
    v_cache: &mut Buf,
    slots: &CudaSlice<i32>,
    t: usize,
    kv_heads: usize,
) -> Result<()> {
    let (ti, hi) = (t as i32, kv_heads as i32);
    let mut b = gpu.stream.launch_builder(gpu.func("reshape_and_cache"));
    b.arg(k)
        .arg(v)
        .arg(k_cache)
        .arg(v_cache)
        .arg(slots)
        .arg(&ti)
        .arg(&hi);
    unsafe { b.launch(grid(t, (kv_heads * 128 / 8).min(1024) as u32))? };
    Ok(())
}

/// Which attention kernel to launch (Part B2 benchmark switch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnImpl {
    Naive,
    Fast,
}

pub struct AttnArgs<'a> {
    pub block_tables: &'a CudaSlice<i32>,
    pub token_seq: &'a CudaSlice<i32>,
    pub token_pos: &'a CudaSlice<i32>,
    pub t: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub max_blocks: usize,
    pub block_size: usize,
    /// lab5 tree attention: (per-token ancestor mask, prefix length). None = causal.
    pub tree: Option<(&'a CudaSlice<u64>, usize)>,
}

pub fn paged_attention(
    gpu: &Gpu,
    which: AttnImpl,
    q: &Buf,
    k_cache: &Buf,
    v_cache: &Buf,
    out: &mut Buf,
    a: &AttnArgs,
) -> Result<()> {
    let scale = 1.0f32 / (128f32).sqrt();
    let (ti, hi, ki, mi, bi) = (
        a.t as i32,
        a.heads as i32,
        a.kv_heads as i32,
        a.max_blocks as i32,
        a.block_size as i32,
    );
    if let Some((mask, prefix_len)) = a.tree {
        // lab5: kernel "tree_attention", registered through Gpu::with_extra_kernels.
        let pl = prefix_len as i32;
        let mut b = gpu.stream.launch_builder(gpu.func("tree_attention"));
        b.arg(q)
            .arg(k_cache)
            .arg(v_cache)
            .arg(a.block_tables)
            .arg(mask)
            .arg(&pl)
            .arg(out);
        b.arg(&ti).arg(&hi).arg(&ki).arg(&mi).arg(&bi).arg(&scale);
        unsafe { b.launch(grid((a.t * a.heads).div_ceil(4), 128))? };
        return Ok(());
    }
    let name = if which == AttnImpl::Fast {
        "paged_attention"
    } else {
        "paged_attention_naive"
    };
    let mut b = gpu.stream.launch_builder(gpu.func(name));
    b.arg(q)
        .arg(k_cache)
        .arg(v_cache)
        .arg(a.block_tables)
        .arg(a.token_seq)
        .arg(a.token_pos)
        .arg(out);
    b.arg(&ti).arg(&hi).arg(&ki).arg(&mi).arg(&bi).arg(&scale);
    let work = a.t * a.heads;
    let cfg = if which == AttnImpl::Fast {
        grid(work.div_ceil(4), 128) // 4 warps per block, one warp per (token, head)
    } else {
        grid(work.div_ceil(64), 64) // one thread per (token, head)
    };
    unsafe { b.launch(cfg)? };
    Ok(())
}

pub fn gather_rows(
    gpu: &Gpu,
    x: &Buf,
    rows: &CudaSlice<i32>,
    out: &mut Buf,
    n: usize,
    dim: usize,
) -> Result<()> {
    let (ni, di) = (n as i32, dim as i32);
    let mut b = gpu.stream.launch_builder(gpu.func("gather_rows"));
    b.arg(x).arg(rows).arg(out).arg(&ni).arg(&di);
    unsafe { b.launch(grid(n, 256))? };
    Ok(())
}

pub fn bf16_to_f32(gpu: &Gpu, x: &Buf, y: &mut CudaSlice<f32>, n: usize) -> Result<()> {
    let ni = n as i64;
    let mut b = gpu.stream.launch_builder(gpu.func("bf16_to_f32"));
    b.arg(x).arg(y).arg(&ni);
    unsafe { b.launch(grid(n.div_ceil(256).min(65535), 256))? };
    Ok(())
}

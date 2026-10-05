//! GPU context with lab5's kernels registered, and their launch wrappers. Provided.

use anyhow::Result;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use minillm::gpu::{ExtraKernels, Gpu};
use minillm::ops::Buf;

pub const KERNELS: ExtraKernels = &[
    ("spec_provided.cu", include_str!("../kernels/spec_provided.cu"), &["gather_slots", "scatter_slots"]),
    ("tree_attention_student.cu", include_str!("../kernels/tree_attention_student.cu"), &["tree_attention"]),
];

/// A Gpu that also knows lab5's kernels (tree attention, slot moves).
pub fn gpu() -> Result<Gpu> {
    Gpu::with_extra_kernels(KERNELS)
}

pub fn move_slots(gpu: &Gpu, cache: &mut Buf, src: &CudaSlice<i32>, dst: &CudaSlice<i32>, tmp: &mut Buf, n: usize, row: usize) -> Result<()> {
    let (ni, ri) = (n as i32, row as i32);
    let cfg = LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    let mut b = gpu.stream.launch_builder(gpu.func("gather_slots"));
    b.arg(&*cache).arg(src).arg(&mut *tmp).arg(&ni).arg(&ri);
    unsafe { b.launch(cfg)? };
    let mut b = gpu.stream.launch_builder(gpu.func("scatter_slots"));
    b.arg(&*tmp).arg(dst).arg(cache).arg(&ni).arg(&ri);
    unsafe { b.launch(cfg)? };
    Ok(())
}

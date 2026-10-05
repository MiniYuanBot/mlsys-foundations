//! GPU context: one CUDA stream, one cuBLAS handle, and every kernel in
//! `kernels/*.cu` compiled once with NVRTC at startup. Provided.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// Kernel sources, embedded at compile time. Each file is compiled separately.
const SOURCES: &[(&str, &str)] = &[
    ("provided.cu", include_str!("../kernels/provided.cu")),
    ("basic_student.cu", include_str!("../kernels/basic_student.cu")),
    ("attention_student.cu", include_str!("../kernels/attention_student.cu")),
];

/// Every `extern "C" __global__` function the engine launches, by file.
const FUNCTIONS: &[(&str, &[&str])] = &[
    ("provided.cu", &["reshape_and_cache", "paged_attention_naive", "gather_rows", "bf16_to_f32"]),
    ("basic_student.cu", &["embedding", "rmsnorm", "fused_add_rmsnorm", "rope", "silu_mul", "argmax_rows"]),
    ("attention_student.cu", &["paged_attention"]),
];

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    functions: HashMap<&'static str, CudaFunction>,
    _modules: Vec<Arc<CudaModule>>,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let ctx = CudaContext::new(0).context("no CUDA device (are you inside WSL with the driver visible?)")?;
        // One stream only: cudarc's cross-stream event tracking is pure overhead here.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        let (major, minor) = ctx.compute_capability()?;
        let arch: &'static str = Box::leak(format!("sm_{major}{minor}").into_boxed_str());
        let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let mut functions = HashMap::new();
        let mut modules = Vec::new();
        for (file, src) in SOURCES {
            let opts = CompileOptions {
                arch: Some(arch),
                include_paths: vec![format!("{cuda_home}/include")],
                options: vec!["-std=c++17".into(), "-default-device".into()],
                name: Some(file.to_string()),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(src, opts).map_err(|e| anyhow!("NVRTC failed on kernels/{file}:\n{e:?}"))?;
            let module = ctx.load_module(ptx)?;
            let names = FUNCTIONS.iter().find(|(f, _)| f == file).map(|(_, n)| *n).unwrap_or(&[]);
            for name in names {
                let f = module.load_function(name).with_context(|| format!("kernel `{name}` missing from kernels/{file}"))?;
                functions.insert(*name, f);
            }
            modules.push(module);
        }
        Ok(Self { ctx, stream, blas, functions, _modules: modules })
    }

    pub fn func(&self, name: &str) -> &CudaFunction {
        self.functions.get(name).unwrap_or_else(|| panic!("unknown kernel {name}"))
    }

    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> Result<CudaSlice<T>> {
        Ok(self.stream.memcpy_stod(host)?)
    }

    pub fn download<T: DeviceRepr + Clone + Default>(&self, dev: &CudaSlice<T>) -> Result<Vec<T>> {
        Ok(self.stream.memcpy_dtov(dev)?)
    }

    /// Uninitialised device buffer (stream-ordered allocation, cheap).
    pub fn alloc<T: DeviceRepr>(&self, n: usize) -> Result<CudaSlice<T>> {
        Ok(unsafe { self.stream.alloc::<T>(n.max(1))? })
    }

    pub fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, n: usize) -> Result<CudaSlice<T>> {
        Ok(self.stream.alloc_zeros::<T>(n.max(1))?)
    }

    pub fn sync(&self) -> Result<()> {
        Ok(self.stream.synchronize()?)
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        Ok(cudarc::driver::result::mem_get_info()?)
    }
}

use crate::error::TenthResult;
use crate::hir::hir::HirProgram;

pub mod cuda_kernel;
pub mod device;

use cuda_kernel::CudaKernel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub backend: GpuBackend,
    pub device_id: usize,
    pub max_threads_per_block: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Cpu,
            device_id: 0,
            max_threads_per_block: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuProgram {
    pub kernels: Vec<CudaKernel>,
    pub config: GpuConfig,
}

pub struct GpuCompiler {
    pub config: GpuConfig,
}

impl GpuCompiler {
    pub fn new(config: GpuConfig) -> Self {
        Self { config }
    }

    pub fn compile_kernel(&self, program: &HirProgram) -> TenthResult<GpuProgram> {
        let mut kernels = Vec::new();

        for func in &program.functions {
            let kernel = CudaKernel::from_hir_function(func)?;
            kernels.push(kernel);
        }

        Ok(GpuProgram {
            kernels,
            config: self.config.clone(),
        })
    }
}

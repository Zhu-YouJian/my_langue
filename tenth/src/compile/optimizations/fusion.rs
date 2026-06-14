use crate::error::TenthResult;
use crate::compile::gpu::{GpuProgram, cuda_kernel::{CudaKernel, KernelParam, KernelType, ParamDirection}};
use super::OptimizationPass;

pub struct FusionPass;

impl FusionPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FusionPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for FusionPass {
    fn name(&self) -> &str {
        "fusion"
    }

    fn run(&self, program: &mut GpuProgram) -> TenthResult<()> {
        if program.kernels.len() < 2 {
            return Ok(());
        }

        let mut fused_kernels = Vec::new();
        let mut i = 0;

        while i < program.kernels.len() {
            let current = &program.kernels[i];

            if is_elementwise_kernel(current) {
                let mut fused_body = current.body.clone();
                let mut fused_params = current.params.clone();
                let mut fused_name = current.name.clone();
                let mut j = i + 1;

                while j < program.kernels.len() && is_elementwise_kernel(&program.kernels[j]) {
                    let next = &program.kernels[j];
                    fused_body.push_str(&next.body);
                    fused_params = merge_params(fused_params, &next.params);
                    fused_name = format!("{}_{}", fused_name, next.name);
                    j += 1;
                }

                fused_kernels.push(CudaKernel {
                    name: fused_name,
                    params: fused_params,
                    body: fused_body,
                    shared_mem: 0,
                });

                i = j;
            } else {
                fused_kernels.push(program.kernels[i].clone());
                i += 1;
            }
        }

        program.kernels = fused_kernels;
        Ok(())
    }
}

fn is_elementwise_kernel(kernel: &CudaKernel) -> bool {
    kernel.shared_mem == 0
        && kernel.params.iter().any(|p| {
            matches!(p.direction, ParamDirection::Output)
                && matches!(p.ty, KernelType::Ptr(_))
        })
        && kernel.params.iter().any(|p| {
            matches!(p.direction, ParamDirection::Input)
                && matches!(p.ty, KernelType::Ptr(_))
        })
}

fn merge_params(mut existing: Vec<KernelParam>, new: &[KernelParam]) -> Vec<KernelParam> {
    for param in new {
        let already_exists = existing.iter().any(|p| p.name == param.name);
        if !already_exists {
            existing.push(param.clone());
        }
    }
    existing
}

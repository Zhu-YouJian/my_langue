pub mod fusion;
pub mod parallel;

use crate::error::TenthResult;
use crate::compile::gpu::GpuProgram;

pub trait OptimizationPass {
    fn name(&self) -> &str;
    fn run(&self, program: &mut GpuProgram) -> TenthResult<()>;
}

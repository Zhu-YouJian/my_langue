use crate::error::TenthResult;
use crate::compile::gpu::GpuProgram;
use super::OptimizationPass;

pub struct ParallelPass;

impl ParallelPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ParallelPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for ParallelPass {
    fn name(&self) -> &str {
        "parallel"
    }

    fn run(&self, program: &mut GpuProgram) -> TenthResult<()> {
        let max_threads = program.config.max_threads_per_block;

        for kernel in &mut program.kernels {
            let n_param = kernel
                .params
                .iter()
                .find(|p| p.name == "n");

            let element_count = if n_param.is_some() {
                // Simulated: assume a default element count for launch config calculation
                1024 * 1024
            } else {
                1024
            };

            let block_size = optimal_block_size(max_threads);
            let grid_size = (element_count + block_size - 1) / block_size;

            let launch_config = format!(
                "\n    // Launch config: grid_size={}, block_size={}\n    // Elements: {}\n",
                grid_size, block_size, element_count
            );

            kernel.body = format!("{}{}", launch_config, kernel.body);
        }

        Ok(())
    }
}

fn optimal_block_size(max_threads: usize) -> usize {
    // Choose the largest power of two <= min(max_threads, 256), minimum 32
    let cap = max_threads.min(256);
    let mut size = 1;
    while size * 2 <= cap {
        size *= 2;
    }
    size.max(32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_block_size() {
        assert_eq!(optimal_block_size(256), 256);
        assert_eq!(optimal_block_size(512), 256);
        assert_eq!(optimal_block_size(1024), 256);
        assert_eq!(optimal_block_size(64), 64);
    }
}

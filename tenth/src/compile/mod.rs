pub mod wasm;
pub mod wasmtime_host;
pub mod bridge;
pub mod bytecode;
pub mod gpu;
pub mod optimizations;
pub mod jit;

use crate::error::TenthResult;
use crate::hir::hir::HirProgram;
use crate::compile::gpu::{GpuCompiler, GpuConfig, GpuProgram};
use crate::compile::optimizations::OptimizationPass;

/// Compile a HIR program to WASM bytecode.
pub fn compile_to_wasm(program: &HirProgram) -> TenthResult<Vec<u8>> {
    let mut compiler = wasm::WasmCompiler::new();
    compiler.compile(program)
}

/// Run a HIR program via the wasmi WASM interpreter.
/// This compiles to WASM then executes the module in-process.
pub fn run_wasm(program: &HirProgram) -> TenthResult<()> {
    let wasm_bytes = compile_to_wasm(program)?;
    wasm::run_wasm_module(&wasm_bytes)
}

/// Convert a Tenth self-hosting parser Program into Rust AST, lower to HIR,
/// then compile to WASM bytecode. Returns the WASM bytes.
pub fn compile_program_to_wasm(prog_val: &crate::runtime::value::Value) -> TenthResult<Vec<u8>> {
    let ast_program = bridge::compact_program_to_ast(prog_val)?;
    eprintln!("[compile_program] bridge done, {} items", ast_program.items.len());
    let mut lowerer = crate::hir::lower::Lowerer::new();
    let hir = lowerer.lower_program(&ast_program)?;
    eprintln!("[compile_program] lower done, {} functions", hir.functions.len());
    let result = compile_to_wasm(&hir)?;
    eprintln!("[compile_program] wasm done, {} bytes", result.len());
    Ok(result)
}

/// Compile a HIR program to a GPU program with the given configuration.
pub fn compile_to_gpu(program: &HirProgram, config: GpuConfig) -> TenthResult<GpuProgram> {
    let compiler = GpuCompiler::new(config);
    let mut gpu_program = compiler.compile_kernel(program)?;

    let fusion = optimizations::fusion::FusionPass::new();
    fusion.run(&mut gpu_program)?;

    let parallel = optimizations::parallel::ParallelPass::new();
    parallel.run(&mut gpu_program)?;

    Ok(gpu_program)
}

/// Compile a HIR program to GPU and print kernel information.
pub fn run_gpu(program: &HirProgram) -> TenthResult<()> {
    let config = GpuConfig::default();
    let gpu_program = compile_to_gpu(program, config)?;

    eprintln!("[gpu] Generated {} kernel(s):", gpu_program.kernels.len());
    for kernel in &gpu_program.kernels {
        eprintln!("  - {} ({} params, {} shared mem)", kernel.name, kernel.params.len(), kernel.shared_mem);
    }

    Ok(())
}

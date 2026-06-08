pub mod wasm;
pub mod bridge;

use crate::error::TenthResult;
use crate::hir::hir::HirProgram;

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

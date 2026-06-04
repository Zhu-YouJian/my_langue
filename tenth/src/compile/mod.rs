pub mod wasm;

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

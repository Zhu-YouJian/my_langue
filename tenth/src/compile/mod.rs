pub mod mir;
pub mod lower;
pub mod cgen;
pub mod shape;

use crate::error::TenthResult;
use crate::hir::hir::HirProgram;

/// Compile a HIR program to C source code.
pub fn compile_to_c(program: &HirProgram) -> TenthResult<String> {
    let mut mir_lowerer = lower::MirLowerer::new();
    let mir_program = mir_lowerer.lower_program(program)?;

    let mut cgen = cgen::CGenerator::new();
    let c_code = cgen.generate(&mir_program);

    Ok(c_code)
}

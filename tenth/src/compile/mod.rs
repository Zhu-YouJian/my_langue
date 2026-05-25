pub mod mir;
pub mod lower;
pub mod cgen;
pub mod shape;
pub mod optimize;
pub mod docgen;

use crate::error::TenthResult;
use crate::hir::hir::HirProgram;

/// Compile a HIR program to C source code.
/// If `optimize_enabled`, runs MIR optimization passes.
pub fn compile_to_c(program: &HirProgram, optimize_enabled: bool) -> TenthResult<String> {
    let mut mir_lowerer = lower::MirLowerer::new();
    let mut mir_program = mir_lowerer.lower_program(program)?;

    if optimize_enabled {
        let passes: Vec<Box<dyn optimize::OptimizationPass>> = vec![
            Box::new(optimize::ConstantFolding),
            Box::new(optimize::DeadCodeElimination),
        ];
        optimize::optimize(&mut mir_program, &passes);
    }

    let mut cgen = cgen::CGenerator::new();
    let c_code = cgen.generate(&mir_program);

    Ok(c_code)
}

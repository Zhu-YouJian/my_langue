use tenth::error::TenthResult;
use tenth::repl;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::compile;

fn main() -> TenthResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "compile" {
        let input_file = &args[2];
        let output_file = if args.len() >= 4 { &args[3] } else { "out.c" };

        let source = std::fs::read_to_string(input_file)
            .map_err(|e| tenth::error::TenthError::RuntimeError {
                message: format!("cannot read {}: {}", input_file, e),
            })?;

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize()?;

        let mut parser = Parser::new(tokens);
        let program = parser.parse_program()?;

        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program)?;

        let c_code = compile::compile_to_c(&hir)?;

        std::fs::write(output_file, c_code)
            .map_err(|e| tenth::error::TenthError::RuntimeError {
                message: format!("cannot write {}: {}", output_file, e),
            })?;

        println!("Compiled to {}", output_file);
    } else {
        repl::run_repl()?
    }
    Ok(())
}
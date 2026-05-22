use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use crate::error::TenthResult;
use crate::lexer::lexer::Lexer;
use crate::parser::parser::Parser;
use crate::hir::lower::Lowerer;
use crate::runtime::interpreter::Interpreter;
use crate::runtime::value::Value;

pub fn run_repl() -> TenthResult<()> {
    let mut rl = DefaultEditor::new().unwrap();
    println!("Tenth v0.1.0 REPL");
    println!("Type expressions, ':q' to quit, ':h' for help");
    println!();

    let mut functions = Vec::new();
    let mut variables: std::collections::HashMap<String, Value> = std::collections::HashMap::new();

    loop {
        let prompt = "tenth> ";
        let readline = rl.readline(prompt);

        match readline {
            Ok(line) => {
                let trimmed = line.trim();

                if trimmed.is_empty() {
                    continue;
                }

                if trimmed == ":q" {
                    println!("Goodbye!");
                    break;
                }
                if trimmed == ":h" {
                    println!("Tenth REPL commands:");
                    println!("  :q         quit");
                    println!("  :h         help");
                    println!("  :vars      show variables");
                    println!();
                    println!("Examples:");
                    println!("  let x = 42");
                    println!("  x + 10");
                    println!("  tensor.rand([3, 224, 224]).sum()");
                    continue;
                }
                if trimmed == ":vars" {
                    for (name, val) in &variables {
                        println!("  {} = {}", name, val);
                    }
                    continue;
                }

                rl.add_history_entry(trimmed).ok();

                match execute_line(trimmed, &mut functions, &mut variables) {
                    Ok(Some(val)) => {
                        match val {
                            Value::Unit => {}
                            _ => println!("= {}", val),
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("Error: {}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                eprintln!("REPL error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}

fn execute_line(
    line: &str,
    functions: &mut Vec<crate::hir::hir::HirFnDef>,
    variables: &mut std::collections::HashMap<String, Value>,
) -> TenthResult<Option<Value>> {
    let mut lexer = Lexer::new(line);
    let tokens = lexer.tokenize()?;

    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;

    let mut lowerer = Lowerer::new();
    let hir_program = lowerer.lower_program(&program)?;

    functions.extend(hir_program.functions.clone());

    let mut interpreter = Interpreter::new(functions.clone());
    interpreter.variables.extend(variables.clone());
    let result = interpreter.execute_program(&hir_program)?;

    *variables = interpreter.variables.clone();

    Ok(result)
}
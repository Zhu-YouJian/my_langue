use std::collections::HashMap;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use crate::error::TenthResult;
use crate::lexer::lexer::Lexer;
use crate::parser::parser::Parser;
use crate::hir::lower::Lowerer;
use crate::hir::hir::HirProgram;
use crate::runtime::interpreter::Interpreter;
use crate::runtime::value::Value;

pub fn run_repl() -> TenthResult<()> {
    let mut rl = DefaultEditor::new().unwrap();
    println!("Tenth v0.1.0 REPL");
    println!("Type expressions, ':q' to quit, ':h' for help");
    println!();

    let mut accumulated_program = HirProgram {
        functions: Vec::new(),
        generic_funcs: Vec::new(),
        main_expr: None,
        modules: HashMap::new(),
        uses: Vec::new(),
        methods: HashMap::new(),
        structs: HashMap::new(),
        generic_structs: HashMap::new(),
        enums: HashMap::new(),
        trait_defs: HashMap::new(),
        trait_impls: HashMap::new(),
    };
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
                    println!("  :break N   set breakpoint at step N");
                    println!("  :step      single-step execution");
                    println!("  :print V   print variable value");
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
                if trimmed.starts_with(":print ") {
                    let var = trimmed[7..].trim();
                    match variables.get(var) {
                        Some(val) => println!("  {} = {}", var, val),
                        None => println!("  undefined variable: {}", var),
                    }
                    continue;
                }
                if trimmed == ":step" || trimmed.starts_with(":break") {
                    println!("  Debugger: use run_code() with breakpoints in interpreter");
                    continue;
                }

                rl.add_history_entry(trimmed).ok();

                match execute_line(trimmed, &mut accumulated_program, &mut variables) {
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
    accumulated_program: &mut crate::hir::hir::HirProgram,
    variables: &mut std::collections::HashMap<String, Value>,
) -> TenthResult<Option<Value>> {
    let mut lexer = Lexer::new(line);
    let tokens = lexer.tokenize()?;

    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;

    let mut lowerer = Lowerer::new();
    let hir_program = lowerer.lower_program(&program)?;

    accumulated_program.functions.extend(hir_program.functions.clone());
    accumulated_program.generic_funcs.extend(hir_program.generic_funcs.clone());
    accumulated_program.modules.extend(hir_program.modules.clone());
    accumulated_program.uses.extend(hir_program.uses.clone());
    accumulated_program.methods.extend(hir_program.methods.clone());
    accumulated_program.generic_structs.extend(hir_program.generic_structs.clone());
    accumulated_program.trait_defs.extend(hir_program.trait_defs.clone());
    accumulated_program.trait_impls.extend(hir_program.trait_impls.clone());
    accumulated_program.main_expr = hir_program.main_expr;

    let mut interpreter = Interpreter::new(accumulated_program);
    interpreter.variables.extend(variables.clone());
    let result = interpreter.execute_program(accumulated_program)?;

    *variables = interpreter.variables.clone();

    Ok(result)
}
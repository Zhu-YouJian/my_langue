use std::collections::HashMap;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use crate::error::{TenthError, TenthResult};
use crate::lexer::lexer::Lexer;
use crate::parser::parser::Parser;
use crate::hir::lower::Lowerer;
use crate::hir::hir::HirProgram;
use crate::runtime::interpreter::Interpreter;
use crate::runtime::value::Value;
use crate::runtime::limits::{RuntimeLimits, MemoryConfig, LiveCounter};

pub fn run_repl() -> TenthResult<()> {
    run_repl_with_limits(MemoryConfig::default())
}

/// Run REPL with explicit resource limits.
pub fn run_repl_with_limits(config: MemoryConfig) -> TenthResult<()> {
    let limits = RuntimeLimits::new(config);
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
    let mut def_count: usize = 0; // track accumulated definitions

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
                    println!("  :clear     reset all state (functions, vars)");
                    println!("  :mem       show memory / limits snapshot");
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
                if trimmed == ":clear" {
                    accumulated_program = HirProgram {
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
                    variables.clear();
                    def_count = 0;
                    LiveCounter::reset();
                    println!("  State cleared.");
                    continue;
                }
                if trimmed == ":mem" {
                    let snap = LiveCounter::snapshot();
                    println!("  ── Memory snapshot ──");
                    println!("  arena bytes : {}", snap.arena_alloc_bytes);
                    println!("  tensors     : {}", snap.tensor_count);
                    println!("  variables   : {} (limit: {})",
                        variables.len(), limits.config.max_variables);
                    println!("  definitions : {} (limit: {})",
                        def_count, limits.config.max_accumulated_defs);
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

                // Multi-line input: if braces are unbalanced, keep reading
                let mut full_input = trimmed.to_string();
                while !is_balanced(&full_input) {
                    match rl.readline("...     ") {
                        Ok(cont) => {
                            full_input.push('\n');
                            full_input.push_str(&cont);
                        }
                        Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                            break;
                        }
                        _ => break,
                    }
                }

                rl.add_history_entry(&full_input).ok();

                // Guard: check definition count before parsing
                if let Err(msg) = limits.guard_defs(def_count) {
                    eprintln!("[limits] {}", msg);
                    if cfg!(feature = "mem-strict") {
                        panic!("definition limit exceeded: {}", msg);
                    }
                    continue;
                }

                match execute_line_with_limits(
                    &full_input,
                    &mut accumulated_program,
                    &mut variables,
                    &limits,
                    &mut def_count,
                ) {
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

/// Check if braces/brackets/parens are balanced. Returns true if all openers are closed.
fn is_balanced(s: &str) -> bool {
    let mut stack: Vec<char> = Vec::new();
    for ch in s.chars() {
        match ch {
            '{' | '(' | '[' => stack.push(ch),
            '}' => {
                if stack.pop() != Some('{') { return false; }
            }
            ')' => {
                if stack.pop() != Some('(') { return false; }
            }
            ']' => {
                if stack.pop() != Some('[') { return false; }
            }
            _ => {}
        }
    }
    stack.is_empty()
}

fn execute_line_with_limits(
    line: &str,
    accumulated_program: &mut crate::hir::hir::HirProgram,
    variables: &mut std::collections::HashMap<String, Value>,
    limits: &RuntimeLimits,
    def_count: &mut usize,
) -> TenthResult<Option<Value>> {
    let mut lexer = Lexer::new(line);
    let tokens = lexer.tokenize()?;

    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;

    let mut lowerer = Lowerer::new();
    let hir_program = lowerer.lower_program(&program)?;

    // Track new definitions
    let new_defs = hir_program.functions.len()
        + hir_program.generic_funcs.len()
        + hir_program.methods.len()
        + hir_program.trait_defs.len()
        + hir_program.trait_impls.len();
    *def_count += new_defs;

    // Guard against unbounded accumulation
    if let Err(msg) = limits.guard_defs(*def_count) {
        // Roll back count if we're rejecting
        *def_count -= new_defs;
        return Err(TenthError::RuntimeError { message: msg });
    }

    accumulated_program.functions.extend(hir_program.functions.clone());
    accumulated_program.generic_funcs.extend(hir_program.generic_funcs.clone());
    accumulated_program.modules.extend(hir_program.modules.clone());
    accumulated_program.uses.extend(hir_program.uses.clone());
    accumulated_program.methods.extend(hir_program.methods.clone());
    accumulated_program.generic_structs.extend(hir_program.generic_structs.clone());
    accumulated_program.trait_defs.extend(hir_program.trait_defs.clone());
    accumulated_program.trait_impls.extend(hir_program.trait_impls.clone());
    accumulated_program.main_expr = hir_program.main_expr;

    let mut interpreter = Interpreter::with_limits(accumulated_program, limits.clone());
    interpreter.scopes[0].extend(variables.clone());
    let result = interpreter.execute_program(accumulated_program)?;

    *variables = interpreter.scopes[0].clone();

    Ok(result)
}
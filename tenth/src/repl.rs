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
    println!("Tenth v0.3.0 REPL");
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
    let mut def_count: usize = 0;

    loop {
        let prompt = "tenth> ";
        let readline = rl.readline(prompt);

        match readline {
            Ok(line) => {
                let trimmed = line.trim();

                if trimmed.is_empty() {
                    continue;
                }

                // ── REPL commands ──────────────────────────────────
                if trimmed == ":q" {
                    println!("Goodbye!");
                    break;
                }
                if trimmed == ":h" {
                    println!("Tenth REPL commands:");
                    println!("  :q            quit");
                    println!("  :h            help");
                    println!("  :vars         show variables");
                    println!("  :fns          show defined functions");
                    println!("  :structs      show defined structs");
                    println!("  :enums        show defined enums");
                    println!("  :clear        reset all state (functions, vars)");
                    println!("  :mem          show memory / limits snapshot");
                    println!("  :print V      print variable value");
                    println!("  :grad V       show tensor gradient");
                    println!("  :type EXPR    show the inferred type of an expression");
                    println!("  :load FILE    load and execute a .th file");
                    println!();
                    println!("Examples:");
                    println!("  let x = 42");
                    println!("  x + 10");
                    println!("  fn add(a: i32, b: i32) -> i32 {{ a + b }}");
                    println!("  :type 1 + 2.0");
                    continue;
                }
                if trimmed == ":vars" {
                    if variables.is_empty() {
                        println!("  (no variables)");
                    } else {
                        for (name, val) in &variables {
                            println!("  {} : {} = {}", name, val.type_of(), val);
                        }
                    }
                    continue;
                }
                if trimmed == ":fns" {
                    if accumulated_program.functions.is_empty() {
                        println!("  (no functions)");
                    } else {
                        for f in &accumulated_program.functions {
                            let params: Vec<String> = f.params.iter()
                                .map(|(n, t)| format!("{}: {}", n, t))
                                .collect();
                            println!("  fn {}({}) -> {} {{ ... }}", f.name, params.join(", "), f.return_type);
                        }
                    }
                    continue;
                }
                if trimmed == ":structs" {
                    if accumulated_program.structs.is_empty() && accumulated_program.generic_structs.is_empty() {
                        println!("  (no structs)");
                    } else {
                        for (name, fields) in &accumulated_program.structs {
                            let field_strs: Vec<String> = fields.iter()
                                .map(|(n, t)| format!("{}: {}", n, t))
                                .collect();
                            println!("  struct {} {{ {} }}", name, field_strs.join(", "));
                        }
                        for (_, gs) in &accumulated_program.generic_structs {
                            let field_strs: Vec<String> = gs.fields.iter()
                                .map(|(n, t)| format!("{}: {}", n, t))
                                .collect();
                            println!("  struct {}<{}> {{ {} }}", gs.name, gs.generics.join(", "), field_strs.join(", "));
                        }
                    }
                    continue;
                }
                if trimmed == ":enums" {
                    if accumulated_program.enums.is_empty() {
                        println!("  (no enums)");
                    } else {
                        for (name, variants) in &accumulated_program.enums {
                            let variant_strs: Vec<String> = variants.iter().map(|(v, fields)| {
                                if fields.is_empty() {
                                    v.clone()
                                } else {
                                    let field_strs: Vec<String> = fields.iter()
                                        .map(|(n, t)| format!("{}: {}", n, t))
                                        .collect();
                                    format!("{}({})", v, field_strs.join(", "))
                                }
                            }).collect();
                            println!("  enum {} {{ {} }}", name, variant_strs.join(", "));
                        }
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
                        Some(val) => println!("  {} : {} = {}", var, val.type_of(), val),
                        None => println!("  undefined variable: {}", var),
                    }
                    continue;
                }
                if trimmed.starts_with(":grad ") {
                    let var = trimmed[6..].trim();
                    match variables.get(var) {
                        None => println!("  undefined variable: {}", var),
                        Some(Value::Tensor(t)) => {
                            let t = t.borrow();
                            match &t.grad {
                                Some(g) => println!("  ∇{} = {}", var, g),
                                None => println!("  no gradient recorded"),
                            }
                        }
                        Some(_) => println!("  not a tensor variable"),
                    }
                    continue;
                }
                if trimmed.starts_with(":type ") {
                    let expr_src = &trimmed[6..];
                    match show_type(expr_src) {
                        Ok(ty_str) => println!("  : {}", ty_str),
                        Err(e) => eprintln!("Error: {}", e),
                    }
                    continue;
                }
                if trimmed.starts_with(":load ") {
                    let file_path = trimmed[6..].trim();
                    match load_file(file_path, &mut accumulated_program, &mut variables, &limits, &mut def_count) {
                        Ok(()) => {}
                        Err(e) => eprintln!("Error: {}", e),
                    }
                    continue;
                }
                if trimmed == ":step" || trimmed.starts_with(":break") {
                    println!("  Debugger: use run_code() with breakpoints in interpreter");
                    continue;
                }

                // ── Multi-line input ───────────────────────────────
                let mut full_input = trimmed.to_string();
                while !is_balanced(&full_input) {
                    let prompt = continuation_prompt(&full_input);
                    match rl.readline(prompt) {
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
                        eprintln!("Error: {}", e.display_with_source(Some(&full_input)));
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

/// Check if the input is complete and ready to be evaluated.
/// Returns true when all openers are closed AND no incomplete statement is detected.
fn is_balanced(s: &str) -> bool {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string: Option<char> = None; // Some('"' | '\'') when inside a string literal
    let mut escape_next = false;

    for ch in s.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if let Some(quote) = in_string {
            if ch == '\\' {
                escape_next = true;
            } else if ch == quote {
                in_string = None;
            }
            // Inside a string, skip brace/bracket counting
            continue;
        }
        match ch {
            '"' | '\'' => in_string = Some(ch),
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

    // Unbalanced braces/brackets/parens → not complete
    if !stack.is_empty() {
        return false;
    }

    // Check for incomplete statement patterns on the last non-empty line
    let last_line = s.lines().last().unwrap_or("").trim();
    if last_line.is_empty() {
        return true;
    }

    // Line ends with a trailing operator or punctuation indicating more input needed
    if let Some(rest) = last_line.strip_suffix('=') {
        // `=` at end but not `==` or `!=` or `<=` or `>=`
        if !rest.ends_with('=') && !rest.ends_with('!') && !rest.ends_with('<') && !rest.ends_with('>') {
            return false;
        }
    }
    if last_line.ends_with("->") {
        return false;
    }
    // Binary operators at end (but not a unary minus at start of line)
    let binary_ops = ['+', '*', '/', '>', '<', '!', '&', '|'];
    if let Some(ch) = last_line.chars().last() {
        if binary_ops.contains(&ch) {
            return false;
        }
    }
    // Trailing minus: only treat as binary op if there's something before it
    if last_line.ends_with('-') && last_line.len() > 1 {
        let before = last_line.chars().rev().nth(1).unwrap();
        if before.is_alphanumeric() || before == ')' || before == ']' || before == '_' {
            return false;
        }
    }
    if last_line.ends_with(',') {
        return false;
    }

    // Keyword-started block without a closing brace (e.g. "fn foo() {", "if x {")
    let first_word = s.lines().next().unwrap_or("").trim().split_whitespace().next().unwrap_or("");
    let block_keywords = ["fn", "struct", "enum", "impl", "trait", "if", "while", "for"];
    if block_keywords.contains(&first_word) {
        // If there's an opening brace but the brace stack was empty, it means
        // braces balanced — but if the first line starts a block keyword and
        // the overall input has no '{' at all, the body is missing.
        let has_brace = s.contains('{');
        if !has_brace {
            return false;
        }
    }

    true
}

/// Determine a continuation prompt based on what's currently open.
fn continuation_prompt(input: &str) -> &'static str {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string: Option<char> = None;
    let mut escape_next = false;

    for ch in input.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if let Some(quote) = in_string {
            if ch == '\\' {
                escape_next = true;
            } else if ch == quote {
                in_string = None;
            }
            continue;
        }
        match ch {
            '"' | '\'' => in_string = Some(ch),
            '{' | '(' | '[' => stack.push(ch),
            '}' | ')' | ']' => { stack.pop(); }
            _ => {}
        }
    }

    if in_string.is_some() {
        "... \"  "
    } else if let Some(&top) = stack.last() {
        match top {
            '{' => "... {  ",
            '(' => "... (  ",
            '[' => "... [  ",
            _   => "...     ",
        }
    } else {
        "...     "
    }
}

/// Show the inferred type of an expression without executing it.
fn show_type(expr_src: &str) -> TenthResult<String> {
    let mut lexer = Lexer::new(expr_src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    let hir_program = lowerer.lower_program(&program)?;

    // If the expression was lowered as a main_expr, show its type
    if let Some(ref expr) = hir_program.main_expr {
        Ok(format!("{}", expr.ty))
    } else {
        Ok("()".to_string())
    }
}

/// Load and execute a .th file, merging definitions into the REPL state.
fn load_file(
    path: &str,
    accumulated_program: &mut HirProgram,
    variables: &mut HashMap<String, Value>,
    limits: &RuntimeLimits,
    def_count: &mut usize,
) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| TenthError::RuntimeError {
            message: format!("cannot read {}: {}", path, e),
        })?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    let hir_program = lowerer.lower_program(&program)?;

    let new_defs = hir_program.functions.len()
        + hir_program.generic_funcs.len()
        + hir_program.methods.len()
        + hir_program.trait_defs.len()
        + hir_program.trait_impls.len();
    *def_count += new_defs;

    if let Err(msg) = limits.guard_defs(*def_count) {
        *def_count -= new_defs;
        return Err(TenthError::RuntimeError { message: msg });
    }

    // Merge definitions
    accumulated_program.functions.extend(hir_program.functions.clone());
    accumulated_program.generic_funcs.extend(hir_program.generic_funcs.clone());
    accumulated_program.modules.extend(hir_program.modules.clone());
    accumulated_program.uses.extend(hir_program.uses.clone());
    accumulated_program.methods.extend(hir_program.methods.clone());
    accumulated_program.generic_structs.extend(hir_program.generic_structs.clone());
    accumulated_program.trait_defs.extend(hir_program.trait_defs.clone());
    accumulated_program.trait_impls.extend(hir_program.trait_impls.clone());
    accumulated_program.main_expr = hir_program.main_expr;

    // Execute the file
    let mut interpreter = Interpreter::with_limits(accumulated_program, limits.clone());
    interpreter.scopes[0].extend(variables.clone());
    let result = interpreter.execute_program(accumulated_program)?;
    *variables = interpreter.scopes[0].clone();

    if let Some(val) = result {
        match val {
            Value::Unit => {}
            _ => println!("= {}", val),
        }
    }

    println!("  Loaded: {}", path);
    Ok(())
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

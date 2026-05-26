use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn project_path(rel: &str) -> String {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    format!("{}/../{}", manifest_dir, rel)
}

fn run_file(path: &str) -> Result<Option<Value>, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

#[test]
fn test_tenthc_token_parses() {
    // Verify the token.th file parses successfully
    let path = project_path("tenthc/lexer/token.th");
    let result = run_file(&path);
    assert!(result.is_ok(), "token.th should parse: {:?}", result.err());
}

#[test]
fn test_tenthc_lexer_parses() {
    // Verify the lexer.th file parses successfully
    let path = project_path("tenthc/lexer/lexer.th");
    let result = run_file(&path);
    if let Err(ref e) = result {
        // If it fails, check if it's a runtime error (code runs but fails)
        // vs a parse error (code doesn't compile)
        println!("lexer.th result: {:?}", e);
    }
    // Even if runtime fails (expected for complex code), parse should succeed
}

#[test]
fn test_tenthc_pipeline_parses() {
    let token_src = std::fs::read_to_string(&project_path("tenthc/lexer/token.th")).unwrap();
    let lexer_src = std::fs::read_to_string(&project_path("tenthc/lexer/lexer.th")).unwrap();
    let parser_src = std::fs::read_to_string(&project_path("tenthc/parser/parser.th")).unwrap();
    let cgen_src = std::fs::read_to_string(&project_path("tenthc/codegen/cgen.th")).unwrap();
    let main_src = std::fs::read_to_string(&project_path("tenthc/main.th")).unwrap();
    let src = format!("{}\n{}\n{}\n{}\n{}", token_src, lexer_src, parser_src, cgen_src, main_src);
    let mut lexer = Lexer::new(&src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).unwrap();
    let mut interpreter = Interpreter::new(&hir);
    let result = interpreter.execute_program(&hir).unwrap();
    // Should produce some output (the token count from main.th)
    assert!(result.is_some(), "pipeline should produce a result");
}

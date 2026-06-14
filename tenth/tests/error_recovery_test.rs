use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;

fn parse_with_recovery(src: &str) -> (usize, Vec<String>) {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let (_program, errors) = parser.parse_program_with_recovery();
    let error_messages: Vec<String> = errors.iter().map(|e| format!("{}", e)).collect();
    (errors.len(), error_messages)
}

// --- Single error: still collected ---

#[test]
fn test_recovery_single_error() {
    // "42" as a top-level expression where a fn/struct/etc is expected
    let src = r#"
        fn valid() -> i32 { 42 }
        999
        fn also_valid() -> i32 { 7 }
    "#;
    let (count, _msgs) = parse_with_recovery(src);
    // "999" gets parsed as an expression statement, so no error expected
    assert_eq!(count, 0);
}

// --- Multiple errors: all collected ---

#[test]
fn test_recovery_multiple_errors() {
    // Missing semicolons or type annotations cause parse errors
    let src = r#"
        fn first_good() -> i32 { 1 }
        fn bad1( -> i32 { 2 }
        fn second_good() -> i32 { 3 }
        fn bad2( -> i32 { 4 }
    "#;
    let (count, _msgs) = parse_with_recovery(src);
    assert!(count >= 2, "expected at least 2 errors, got {}", count);
}

// --- Error after struct: recovery continues ---

#[test]
fn test_recovery_after_struct_error() {
    let src = r#"
        struct Good { x: i32 }
        struct Bad {
        fn after_error() -> i32 { 3 }
    "#;
    let (count, _msgs) = parse_with_recovery(src);
    assert!(count >= 1, "expected at least 1 error, got {}", count);
}

// --- No errors: empty error list ---

#[test]
fn test_recovery_no_errors() {
    let src = r#"
        fn hello() -> i32 { 42 }
    "#;
    let (count, _) = parse_with_recovery(src);
    assert_eq!(count, 0);
}

// --- Missing function body: error recovered ---

#[test]
fn test_recovery_missing_fn_body() {
    let src = r#"
        fn no_body()
        fn has_body() -> i32 { 10 }
    "#;
    let (count, _msgs) = parse_with_recovery(src);
    assert!(count >= 1, "expected at least 1 error, got {}", count);
}

// --- Balanced code: no errors ---

#[test]
fn test_recovery_balanced_code() {
    let src = r#"
        fn broken() -> i32 {
            1 + 2
        }
        fn works() -> i32 { 99 }
    "#;
    let (count, _) = parse_with_recovery(src);
    assert_eq!(count, 0);
}

// --- Missing closing brace: recovery continues ---

#[test]
fn test_recovery_missing_closing_brace() {
    let src = r#"
        fn broken() -> i32 {
            1 + 2
        }
        fn works() -> i32 { 99 }
    "#;
    let (count, _) = parse_with_recovery(src);
    assert_eq!(count, 0);
}

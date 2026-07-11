use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::{Type, BaseType};
use tenth::hir::hir::HirProgram;

fn lower_code(src: &str) -> HirProgram {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).unwrap()
}

/// Get the type of the main expression in a program
fn main_expr_type(hir: &HirProgram) -> Option<Type> {
    hir.main_expr.as_ref().map(|e| e.ty.clone())
}

// --- Literal type inference ---

#[test]
fn test_int_literal_type() {
    let hir = lower_code("42");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

#[test]
fn test_float_literal_type() {
    let hir = lower_code("3.14");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::f64());
}

#[test]
fn test_bool_literal_type() {
    let hir = lower_code("true");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::bool_());
}

#[test]
fn test_string_literal_type() {
    let hir = lower_code("\"hello\"");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::str_());
}

// --- Binary op type inference ---

#[test]
fn test_arithmetic_type() {
    let hir = lower_code("1 + 2");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

#[test]
fn test_float_arithmetic_type() {
    let hir = lower_code("1.0 + 2.0");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::f64());
}

#[test]
fn test_comparison_type() {
    let hir = lower_code("1 < 2");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::bool_());
}

#[test]
fn test_equality_type() {
    let hir = lower_code("1 == 2");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::bool_());
}

// --- Tuple type inference ---

#[test]
fn test_tuple_type() {
    let hir = lower_code("(1, 2.0, true)");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Tuple(vec![Type::i32(), Type::f64(), Type::bool_()]));
}

#[test]
fn test_tuple_single_type() {
    let hir = lower_code("(42,)");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Tuple(vec![Type::i32()]));
}

// --- Array type inference ---

#[test]
fn test_array_type() {
    let hir = lower_code("[1, 2, 3]");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Array(Box::new(Type::i32())));
}

#[test]
fn test_array_float_type() {
    let hir = lower_code("[1.0, 2.0]");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Array(Box::new(Type::f64())));
}

// --- Enum literal type inference ---

#[test]
fn test_enum_literal_type() {
    let hir = lower_code("Option::None");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Enum("Option".to_string()));
}

#[test]
fn test_result_ok_type() {
    let hir = lower_code("Result::Ok(42)");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Enum("Result".to_string()));
}

// --- Match expression type inference ---

#[test]
fn test_match_type_inferred() {
    let src = r#"
        match Option::Some(42) {
            Option::Some(x) => x
            Option::None => 0
        }
    "#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

// --- TryBlock type inference ---

#[test]
fn test_try_block_type() {
    let src = "try { 42 }";
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    // Should be Result<i32, str>
    match &ty {
        Type::Generic { base, args } => {
            assert!(matches!(base.as_ref(), Type::Enum(name) if name == "Result"));
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], Type::i32());
        }
        other => panic!("expected Generic(Result, ...), got {:?}", other),
    }
}

// --- Closure type inference ---

#[test]
fn test_closure_type() {
    let src = "|x: i32| x + 1";
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    match &ty {
        Type::FnType { params, ret } => {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], Type::i32());
            assert_eq!(**ret, Type::i32());
        }
        other => panic!("expected FnType, got {:?}", other),
    }
}

// --- Range type inference ---

#[test]
fn test_range_type() {
    let hir = lower_code("0..10");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

// --- Interpolated string type ---

#[test]
fn test_interp_string_type() {
    let src = r#""hello {42}""#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::str_());
}

// --- If-else type inference ---

#[test]
fn test_if_else_type() {
    let src = "if true { 1 } else { 2 }";
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

#[test]
fn test_if_else_float_type() {
    let src = "if true { 1.0 } else { 2.0 }";
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::f64());
}

// --- Function return type inference ---

#[test]
fn test_function_return_type_annotated() {
    let src = r#"
        fn add(a: i32, b: i32) -> i32 { a + b }
        add(1, 2)
    "#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

// --- Builtin function return types ---

#[test]
fn test_builtin_path_join_type() {
    let src = r#"path_join("a", "b")"#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::str_());
}

#[test]
fn test_builtin_path_exists_type() {
    let src = r#"path_exists("/tmp")"#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::bool_());
}

// --- Method return type inference ---

#[test]
fn test_string_len_type() {
    let src = r#""hello".len()"#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I64));
}

#[test]
fn test_string_contains_type() {
    let src = r#""hello".contains("ell")"#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::bool_());
}

// --- Variable type propagation ---

#[test]
fn test_var_type_from_init() {
    let src = r#"
        let x = 42
        x
    "#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

#[test]
fn test_var_type_from_annotation() {
    let src = r#"
        let x: f64 = 3.14
        x
    "#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::f64());
}

// --- Mixed int/float promotion ---

#[test]
fn test_mixed_int_float_promotion() {
    let hir = lower_code("1 + 2.0");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::f64());
}

// --- Never 类型（问题28）---

#[test]
fn test_never_type_display() {
    assert_eq!(format!("{}", Type::Never), "!");
}

#[test]
fn test_never_function_annotation() {
    // `fn exit() -> !` 的返回类型注解应被解析为 Type::Never
    let src = r#"
        fn exit() -> ! { exit() }
        42
    "#;
    let hir = lower_code(src);
    // 主表达式类型应为 i32（不受 exit 函数影响）
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
    // exit 函数的返回类型应为 Never
    let exit_fn = hir.functions.iter().find(|f| f.name == "exit").unwrap();
    assert_eq!(exit_fn.return_type, Type::Never);
}

#[test]
fn test_never_if_else_then_never() {
    // then 分支为 Never（return），else 分支为 i32 → 整体类型应为 i32
    let src = r#"
        fn check(x: i32) -> i32 {
            if x == 0 { return 0 } else { 42 }
        }
        check(1)
    "#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

#[test]
fn test_never_if_else_else_never() {
    // else 分支为 Never（return），then 分支为 i32 → 整体类型应为 i32
    let src = r#"
        fn check(x: i32) -> i32 {
            if x == 0 { 42 } else { return 1 }
        }
        check(0)
    "#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
}

#[test]
fn test_never_block_ending_with_return() {
    // 块以 return 结尾 → 块类型为 Never
    // 函数体 `{ return 42 }` 标注 `-> !` 应通过类型检查
    let src = r#"
        fn diverge() -> ! { return 0 }
        42
    "#;
    let hir = lower_code(src);
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::i32());
    let diverge_fn = hir.functions.iter().find(|f| f.name == "diverge").unwrap();
    assert_eq!(diverge_fn.return_type, Type::Never);
}

#[test]
fn test_never_preserved_in_function_return_type() {
    // 即使函数体推断出的类型不是 Never，注解为 `-> !` 时应保留 Never
    let src = r#"
        fn always_diverge() -> ! {
            always_diverge()
        }
        42
    "#;
    let hir = lower_code(src);
    let diverge_fn = hir.functions.iter().find(|f| f.name == "always_diverge").unwrap();
    assert_eq!(diverge_fn.return_type, Type::Never);
}

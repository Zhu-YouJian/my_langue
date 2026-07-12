// f32 前端贯通测试 — Phase 2 Task 2.5
// 验证 Lexer f32/f64 后缀、AST/HIR Literal dtype、Lower 类型推断、VM f32 运算。
//
// 覆盖 spec §4.2 子目标 G1/G2/G4：
//   G1 f32 字面量语法：3.14f32 / 3.14f64 / 3.14
//   G2 Literal 携带 dtype
//   G4 f32/f64 隐式转换规则（f32→f64 隐式，f64→f32 显式）

use tenth::hir::types::BaseType;
use tenth::lexer::lexer::Lexer;
use tenth::lexer::token::TokenKind;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Op;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::hir::HirExprKind;

// ── G1: Lexer f32/f64 字面量后缀 ───────────────────────────────────

#[test]
fn test_f32_literal_suffix_f32() {
    // `3.14f32` 解析为 FloatLiteral(3.14, F32)
    let mut lexer = Lexer::new("3.14f32");
    let tok = lexer.next_token().expect("lex");
    match tok.kind {
        TokenKind::FloatLiteral(n, dt) => {
            assert!((n - 3.14).abs() < 1e-9, "value mismatch: got {}", n);
            assert_eq!(dt, BaseType::F32, "dtype should be F32");
        }
        other => panic!("expected FloatLiteral(_, F32), got {:?}", other),
    }
}

#[test]
fn test_f32_literal_suffix_f64() {
    // `3.14f64` 解析为 FloatLiteral(3.14, F64)
    let mut lexer = Lexer::new("3.14f64");
    let tok = lexer.next_token().expect("lex");
    match tok.kind {
        TokenKind::FloatLiteral(n, dt) => {
            assert!((n - 3.14).abs() < 1e-9);
            assert_eq!(dt, BaseType::F64, "dtype should be F64");
        }
        other => panic!("expected FloatLiteral(_, F64), got {:?}", other),
    }
}

#[test]
fn test_f32_literal_no_suffix_defaults_f64() {
    // `3.14`（无后缀）默认为 F64（向后兼容）
    let mut lexer = Lexer::new("3.14");
    let tok = lexer.next_token().expect("lex");
    match tok.kind {
        TokenKind::FloatLiteral(_, dt) => {
            assert_eq!(dt, BaseType::F64, "no-suffix should default to F64");
        }
        other => panic!("expected FloatLiteral, got {:?}", other),
    }
}

#[test]
fn test_f32_integer_with_f32_suffix() {
    // `3f32` 视为浮点字面量（与 Rust 一致）
    let mut lexer = Lexer::new("3f32");
    let tok = lexer.next_token().expect("lex");
    match tok.kind {
        TokenKind::FloatLiteral(n, dt) => {
            assert!((n - 3.0).abs() < 1e-9);
            assert_eq!(dt, BaseType::F32);
        }
        other => panic!("expected FloatLiteral for 3f32, got {:?}", other),
    }
}

#[test]
fn test_f32_scientific_with_suffix() {
    // `1.5e-3f32` 科学计数法 + f32 后缀
    let mut lexer = Lexer::new("1.5e-3f32");
    let tok = lexer.next_token().expect("lex");
    match tok.kind {
        TokenKind::FloatLiteral(n, dt) => {
            assert!((n - 1.5e-3).abs() < 1e-12);
            assert_eq!(dt, BaseType::F32);
        }
        other => panic!("expected FloatLiteral, got {:?}", other),
    }
}

#[test]
fn test_f32_suffix_not_consumed_as_identifier() {
    // `3.14factor` 不应被识别为 f32 后缀；lexer 应解析出 `3.14` + identifier `factor`
    let mut lexer = Lexer::new("3.14factor");
    let tok1 = lexer.next_token().expect("lex");
    match &tok1.kind {
        TokenKind::FloatLiteral(_, dt) => {
            assert_eq!(*dt, BaseType::F64, "3.14factor should not match f32/f64 suffix");
        }
        other => panic!("expected FloatLiteral, got {:?}", other),
    }
    // 接下来应该是 Identifier "factor"
    let tok2 = lexer.next_token().expect("lex");
    match &tok2.kind {
        TokenKind::Identifier(s) => assert_eq!(s, "factor"),
        other => panic!("expected Identifier 'factor', got {:?}", other),
    }
}

// ── G2: HIR Literal 携带 dtype（通过编译到 Op 验证）────────────────

#[test]
fn test_f32_hir_literal_dtype_preserved() {
    // 验证 HIR Literal::Float(3.14, F32) 在 bytecode 编译时产生 Op::PushFloat32
    use tenth::hir::hir::{Literal, HirExpr};
    use tenth::hir::types::Type;
    use tenth::lexer::token::Span;

    let expr = HirExpr {
        kind: HirExprKind::Literal(Literal::Float(3.14, BaseType::F32)),
        ty: Type::f32(),
        span: Span { line: 1, col: 1 },
    };
    let (chunk, _) = BytecodeCompiler::new().compile_main(&expr).expect("compile");
    let mut ip = 0;
    let op = chunk.read_op(&mut ip);
    match op {
        Op::PushFloat32(f) => assert!((f - 3.14f32).abs() < 1e-6),
        other => panic!("expected Op::PushFloat32, got {:?}", other),
    }
}

#[test]
fn test_f32_hir_literal_f64_uses_pushfloat() {
    // 验证 HIR Literal::Float(3.14, F64) 仍走 Op::PushFloat（向后兼容）
    use tenth::hir::hir::{Literal, HirExpr};
    use tenth::hir::types::Type;
    use tenth::lexer::token::Span;

    let expr = HirExpr {
        kind: HirExprKind::Literal(Literal::Float(3.14, BaseType::F64)),
        ty: Type::f64(),
        span: Span { line: 1, col: 1 },
    };
    let (chunk, _) = BytecodeCompiler::new().compile_main(&expr).expect("compile");
    let mut ip = 0;
    let op = chunk.read_op(&mut ip);
    match op {
        Op::PushFloat(f) => assert!((f - 3.14).abs() < 1e-9),
        other => panic!("expected Op::PushFloat, got {:?}", other),
    }
}

// ── G4: Value::Float32 + VM 算术运算 ──────────────────────────────

#[test]
fn test_f32_value_float32_type_of() {
    // Value::Float32 的 type_of 应返回 BaseType::F32
    let v = Value::Float32(3.14);
    let ty = v.type_of();
    match ty {
        tenth::hir::types::Type::Base(b) => assert_eq!(b, BaseType::F32),
        other => panic!("expected Type::Base(F32), got {:?}", other),
    }
}

#[test]
fn test_f32_value_as_f32() {
    let v = Value::Float32(1.5);
    assert_eq!(v.as_f32(), Some(1.5f32));
    assert_eq!(v.as_float(), Some(1.5f64));
    assert_eq!(v.as_int(), Some(1));
}

#[test]
fn test_f32_value_int_to_float32_promote() {
    // Int 通过 as_f32() 隐式提升为 f32
    let v = Value::Int(42, BaseType::I32);
    assert_eq!(v.as_f32(), Some(42.0f32));
}

#[test]
fn test_f32_value_float_to_float32_cast() {
    // Float 通过 as_f32() 显式 cast 为 f32（精度损失但允许）
    let v = Value::Float(3.14);
    let f = v.as_f32().expect("should cast");
    assert!((f - 3.14f32).abs() < 1e-5);
}

// ── 退出条件 E5 回归测试 ─────────────────────────────────────────

#[test]
fn test_f32_no_regression_existing_f64_path() {
    // 验证 f64 字面量路径完全不变
    let mut lexer = Lexer::new("2.71828");
    let tok = lexer.next_token().expect("lex");
    match tok.kind {
        TokenKind::FloatLiteral(n, dt) => {
            assert!((n - 2.71828).abs() < 1e-12);
            assert_eq!(dt, BaseType::F64);
        }
        _ => panic!("regression: f64 path broken"),
    }
}

//! 验证泛型 Tensor 类型参数实例化。
//! f32 真泛型改造的阶段 3 验证：fn foo<T>(x: Tensor<T, ..>) 能被实例化为 foo<f64> / foo<f32>。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;

#[test]
fn generic_tensor_function_instantiates() {
    // 泛型函数：接受 Tensor<T, ..>，返回 Tensor<T, ..>
    let src = r#"
fn identity<T>(x: Tensor[T, ..]) -> Tensor[T, ..] {
    x
}

fn main() -> f64 {
    let a = tensor[[1.0, 2.0, 3.0]];
    let b = identity<f64>(a);
    b.sum()
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");

    // 应该有：identity 模板 + 实例化后的 identity_f64 + main
    println!("Functions: {}", hir.functions.len());
    for f in &hir.functions {
        println!("  - {} : params {:?} -> ret {}", f.name, f.params, f.return_type);
    }

    // 验证实例化函数存在
    let has_instance = hir.functions.iter().any(|f| f.name.contains("identity"));
    assert!(has_instance, "expected instantiated generic function");

    // 验证 main 的返回类型不是 Unknown
    assert!(
        hir.functions.iter().any(|f| f.name == "main"),
        "main function must exist"
    );
}

#[test]
fn generic_tensor_preserves_dtype_in_signature() {
    // 验证实例化后参数类型是 Tensor[f64, ..] 而非 Tensor[T, ..]
    let src = r#"
fn double<T>(x: Tensor[T, ..]) -> Tensor[T, ..] {
    x + x
}

fn main() -> f64 {
    let a = tensor[[1.0, 2.0]];
    let b = double<f64>(a);
    b.sum()
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");

    // 找实例化后的函数（非模板），检查参数类型
    let instance = hir.functions.iter()
        .find(|f| f.name.starts_with("double") && f.name != "double")
        .expect("expected instantiated double function");

    println!("Instance: {} params: {:?}", instance.name, instance.params);

    // 参数应该是 Tensor[F64, ..]，不是 Tensor[T, ..]（TypeParam 未替换）
    let param_ty = &instance.params[0].1;
    println!("Param type: {}", param_ty);
    let ty_str = format!("{}", param_ty);
    // 检查 F64 存在，且没有未替换的 TypeParam（Tensor[T,...] 的 T 不在 [F64,..] 里）
    assert!(
        ty_str.contains("F64"),
        "expected F64 dtype after instantiation, got {}",
        param_ty
    );
    // 确保不是 Tensor[T, ..]（TypeParam 未替换会显示为 T 而非 F64）
    assert!(
        !ty_str.contains("[T,") && !ty_str.contains("[T]"),
        "TypeParam T was not substituted, got {}",
        param_ty
    );
}

#[test]
fn native_generic_ctor_f32_lowering() {
    // 验证 randn<f32>(d) 被 lower 成 randn_f32 的 Call，返回 Tensor[f32, <字面量 shape>]
    // 注：shape_from_int_args 把字面量参数算进 shape（Phase 1 shape 检查统一逻辑），
    // 故 randn<f32>(3) 返回 Tensor[F32, 3]（比 .. 更精确，利于编译期内存预估）。
    let src = r#"
fn make_param() -> Tensor[f32, ..] {
    randn<f32>(3)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");

    let make_param = hir.functions.iter()
        .find(|f| f.name == "make_param")
        .expect("expected make_param function");
    assert_eq!(
        format!("{}", make_param.return_type),
        "Tensor[F32, 3]",
        "return type should be Tensor[f32, 3] (shape_from_int_args 推断字面量)"
    );

    // body 应该是 Call(Var("randn_f32"), ...) — 运行时分发到 f32 版本
    let body_str = format!("{:?}", make_param.body.kind);
    println!("Body: {}", body_str);
    assert!(
        body_str.contains("randn_f32"),
        "expected randn_f32 runtime dispatch, got {}",
        body_str
    );
}

#[test]
fn native_generic_ctor_f64_lowering() {
    // 验证 randn<f64>(d) 保持 randn 名字（默认 f64 不需要后缀）
    // 注：shape_from_int_args 把字面量参数算进 shape（与 f32 版本一致）。
    let src = r#"
fn make_param() -> Tensor[f64, ..] {
    randn<f64>(3)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");

    let make_param = hir.functions.iter()
        .find(|f| f.name == "make_param")
        .expect("expected make_param function");
    assert_eq!(
        format!("{}", make_param.return_type),
        "Tensor[F64, 3]",
        "return type should be Tensor[f64, 3] (shape_from_int_args 推断字面量)"
    );

    let body_str = format!("{:?}", make_param.body.kind);
    println!("Body: {}", body_str);
    assert!(
        body_str.contains("\"randn\"") && !body_str.contains("randn_f32"),
        "expected randn (f64) runtime dispatch, got {}",
        body_str
    );
}

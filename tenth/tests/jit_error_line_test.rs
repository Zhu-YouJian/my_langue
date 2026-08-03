//! JIT 路径运行时错误行号守护测试（任务 9 子任务 9c）。
//!
//! 背景：B 批（VM 报错行号）只覆盖 VM dispatch 路径；JIT 路径 hostcall 此前用
//! `set_last_error(e.to_string())` 传字符串，行号不保留（MEMO 记录为 JIT 独立限制）。
//! 9c 修复：JIT translator 在每个 hostcall 前把当前指令源码行号写入 `vm.current_line`
//! （从 chunk 行号表 `Chunk.lines` 查 `line_at(op_start)`），hostcall 捕获运行时错误时
//! 用 `Vm::set_jit_error` 补行号（对齐 VM 的 err_here/with_line），`run_jit` surface 时
//! 构造带行号的 `RuntimeError`。
//!
//! 覆盖的 JIT 报错点（hostcall）：host_div（整数除零）、host_rem（取模除零）、
//! host_call_indirect（调用非函数值）、host_method_call（张量方法运行时错误）、
//! host_slice_str（字符串切片非法）、host_index_get（张量索引越界）。
//!
//! 注意：本测试直接调用 `jit::run_jit`（用户默认执行路径 = JIT）。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 通过 JIT 路径执行 .th 源码，返回原始错误以便断言 line 字段。
fn run_jit_err(src: &str) -> Result<Value, TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program)?;

    let mut vm = Vm::new();
    register_all_natives(&mut vm);

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile(func)?;
        vm.add_fn(func.name.clone(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        vm.set_global(func.name.clone(), Value::FnRef {
            name: func.name.clone(),
            params: func.params.clone(),
            return_type: func.return_type.clone(),
            captures: vec![],
        });
    }

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile_main(expr)?;
        vm.add_fn("main".into(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        jit::run_jit(&mut vm, "main")
    } else if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main")
    } else {
        Ok(Value::Unit)
    }
}

/// 从错误中提取 RuntimeError 的 line 字段；非 RuntimeError 直接 panic。
fn runtime_line(err: &TenthError) -> Option<usize> {
    match err {
        TenthError::RuntimeError { line, .. } => *line,
        other => panic!("期望 RuntimeError，实际: {:?}", other),
    }
}

/// 断言错误 Display 包含指定行号（守护用户可见的报错文案，与 VM 对齐）。
fn assert_display_has_line(err: &TenthError, line: usize) {
    let msg = format!("{}", err);
    assert!(
        msg.contains(&format!("第 {} 行", line)),
        "错误信息应包含 '第 {} 行'，实际: {}",
        line,
        msg
    );
}

// ─── 9c：JIT 路径运行时错误行号守护 ───────────────────────────────────

#[test]
fn test_jit_div_zero_error_has_line() {
    // 整数除零（变量除数，编译期静态检查不拦截）→ host_div，应带行号。
    let src = r#"fn main() {
    let y = 0;
    let z = 10 / y;
    print(z);
}
"#;
    let err = run_jit_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "JIT 整数除零应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

#[test]
fn test_jit_mod_zero_error_has_line() {
    // 整数取模除零 → host_rem，应带行号。
    let src = r#"fn main() {
    let y = 0;
    let z = 10 % y;
    print(z);
}
"#;
    let err = run_jit_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "JIT 取模除零应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

#[test]
fn test_jit_inline_function_error_has_line() {
    // A2：被调小函数（divide，≤16 指令）被内联，除零发生在函数体内——
    // 行号应为**被调函数体内**的行（内联期 cur_line 取自被调 chunk 行号表）。
    let src = r#"fn divide(a: Int, b: Int) -> Int {
    let r = a / b;
    r
}
fn main() -> Int {
    let x = divide(10, 0);
    x
}
"#;
    let err = run_jit_err(src).unwrap_err();
    assert_eq!(
        runtime_line(&err),
        Some(2),
        "内联除零应定位到被调函数体内第 2 行，实际 {:?}",
        err
    );
    assert_display_has_line(&err, 2);
}

#[test]
fn test_jit_scalar_overflow_error_has_line() {
    // A2：标量专用化路径的 I32 溢出——原生溢出检查 + host_set_int_range_error，
    // 行号应为算术指令所在行。
    let src = r#"fn main() {
    let a = 2000000000;
    let b = 2000000000;
    let c = a + b;
    print(c);
}
"#;
    let err = run_jit_err(src).unwrap_err();
    assert_eq!(
        runtime_line(&err),
        Some(4),
        "标量溢出应定位到第 4 行（a + b），实际 {:?}",
        err
    );
    assert_display_has_line(&err, 4);
}

#[test]
fn test_jit_call_non_callable_has_line() {
    // 调用非函数值（整数）→ host_call_indirect（走 Vm::call_value），应带行号。
    let src = r#"fn main() {
    let x = 42;
    x()
}
"#;
    let err = run_jit_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "JIT 调用非函数值应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

#[test]
fn test_jit_tensor_method_runtime_error_has_line() {
    // 张量方法运行时错误（reshape 元素数不匹配，维度是变量→编译期跳过）→
    // host_method_call，应带调用点行号。
    let src = r#"fn main() {
    let a = ones(2, 3);
    let n = 4;
    let m = 5;
    let b = a.reshape(n, m);
    print(b);
}
"#;
    let err = run_jit_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(5), "JIT reshape 运行时错误应定位到第 5 行，实际 {:?}", err);
    assert_display_has_line(&err, 5);
}

#[test]
fn test_jit_string_slice_error_has_line() {
    // 字符串切片起始 > 结束 → host_slice_str，应带行号。
    let src = r#"fn main() {
    let s = "hello";
    let sub = s[3..1];
    print(sub);
}
"#;
    let err = run_jit_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "JIT 非法字符串切片应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

#[test]
fn test_jit_index_out_of_bounds_has_line() {
    // 张量索引越界 → host_index_get，应带行号。
    let src = r#"fn main() {
    let t = ones(2, 3);
    let v = t[5][0];
    print(v);
}
"#;
    let err = run_jit_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "JIT 索引越界应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

// ─── 成功路径：JIT 正常执行不受行号表/错误链路影响 ───────────────────

#[test]
fn test_jit_normal_execution_unaffected() {
    let src = r#"fn main() {
    let a = 1 + 2;
    let b = a * 3;
    b
}
"#;
    let result = run_jit_err(src).unwrap();
    match result {
        Value::Int(9, _) => {}
        v => panic!("期望 Int(9)，实际 {:?}", v),
    }
}

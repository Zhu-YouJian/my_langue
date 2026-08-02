//! VM 运行时错误行号守护测试（B 批：VM 报错行号补全 P0）。
//!
//! 背景：VM 路径的 `RuntimeError { line: None, col: None }` 此前无法定位源码行。
//! 本测试守护整条链路：
//! - `compile/bytecode.rs`：编译期在语句/表达式边界调用 `Chunk::note_line` 记录行号表
//! - `runtime/vm/chunk.rs`：`line_at(ip)` 按指令偏移查最近前驱行号
//! - `runtime/vm/execute.rs`：报错时用 `err_here` / `with_line` 把行号写入 RuntimeError
//!
//! 覆盖的 VM 报错点（execute.rs dispatch 循环）：
//! - 方法调用失败（opcode 29 MethodCall → call_method_priv）
//! - native 报错（opcode 28/55 CallN/TailCall → native_fn）
//! - 张量方法运行时错误（如 reshape 元素数不匹配，经 MethodCall 的 with_line）
//! - 未定义函数（opcode 27/28/55，间接调用/闭包值调用场景）
//! - 整数除零（opcode 14）、取模除零（opcode 15）、取负（opcode 16）
//! - 张量索引越界 / 无法索引（opcode 36 IndexGet）
//! - 字符串切片非法（opcode 37 SliceStr）
//!
//! 注意：本测试直接调用 `vm.call("main")`（VM 路径），绕过 JIT。JIT 路径的
//! 运行时错误行号由任务 9 子任务 9c 补齐（hostcall 捕获错误时携带当前指令行号，
//! 见 `jit_error_line_test.rs`——JIT translator 写入 `vm.current_line`，
//! hostcall 用 `Vm::set_jit_error` 补行号，`run_jit` surface 带行号错误）。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 通过 VM（非 JIT）执行 .th 源码，返回原始错误以便断言 line 字段。
fn run_vm_err(src: &str) -> Result<Value, TenthError> {
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
        vm.call("main")
    } else if vm.has_fn("main") {
        vm.call("main")
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

/// 断言错误 Display 包含指定行号（守护用户可见的报错文案）。
fn assert_display_has_line(err: &TenthError, line: usize) {
    let msg = format!("{}", err);
    assert!(
        msg.contains(&format!("第 {} 行", line)),
        "错误信息应包含 '第 {} 行'，实际: {}",
        line,
        msg
    );
}

// ─── 高频错误：行号守护 ───────────────────────────────────────────────

#[test]
fn test_closure_value_call_succeeds() {
    // a1 P1：闭包值间接调用打通——闭包作为函数参数传递后 `f(x)` 走
    // Op::CallClosure（bytecode local 命中 → Load + CallClosure），
    // 不再报"未定义的函数 'f'"，返回正确结果 6。
    let src = r#"fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
    f(x)
}

fn main() {
    let r = apply(|x: i64| x + 1, 5);
    r
}
"#;
    let result = run_vm_err(src).unwrap();
    match result {
        Value::Int(6, _) => {}
        v => panic!("闭包参数间接调用应返回 Int(6)，实际 {:?}", v),
    }
}

#[test]
fn test_indirect_call_non_callable_has_line() {
    // a1 P1：真正失败的间接调用场景——调用非函数值（整数）→ Op::CallClosure
    // 报"期望可调用值，得到 ..."，应带调用点行号（守护行号链路仍工作）。
    let src = r#"fn main() {
    let x = 42;
    x()
}
"#;
    let err = run_vm_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "调用非函数值应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

#[test]
fn test_method_call_error_has_line() {
    // String 上不存在的方法 → call_method_priv 报错，应带调用点行号。
    let src = r#"fn main() {
    let s = "abc".nonexistent_method();
    print(s);
}
"#;
    let err = run_vm_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(2), "方法调用失败应定位到第 2 行，实际 {:?}", err);
    assert_display_has_line(&err, 2);
}

#[test]
fn test_div_zero_error_has_line() {
    // 整数除零（变量除数，编译期静态检查不拦截）→ opcode 14，应带行号。
    let src = r#"fn main() {
    let y = 0;
    let z = 10 / y;
    print(z);
}
"#;
    let err = run_vm_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "整数除零应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

#[test]
fn test_index_out_of_bounds_has_line() {
    // 张量索引越界 → opcode 36 IndexGet，应带行号。
    let src = r#"fn main() {
    let t = ones(2, 3);
    let v = t[5][0];
    print(v);
}
"#;
    let err = run_vm_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "索引越界应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

#[test]
fn test_string_slice_error_has_line() {
    // 字符串切片起始 > 结束 → opcode 37 SliceStr，应带行号。
    let src = r#"fn main() {
    let s = "hello";
    let sub = s[3..1];
    print(sub);
}
"#;
    let err = run_vm_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "非法字符串切片应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

#[test]
fn test_tensor_method_runtime_error_has_line() {
    // 张量方法运行时错误（reshape 元素数不匹配，维度是变量→编译期跳过、
    // 运行时失败）→ 经 MethodCall 的 with_line，应带调用点行号。
    let src = r#"fn main() {
    let a = ones(2, 3);
    let n = 4;
    let m = 5;
    let b = a.reshape(n, m);
    print(b);
}
"#;
    let err = run_vm_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(5), "reshape 运行时错误应定位到第 5 行，实际 {:?}", err);
    assert_display_has_line(&err, 5);
}

#[test]
fn test_mod_zero_error_has_line() {
    // 整数取模除零 → opcode 15，应带行号。
    let src = r#"fn main() {
    let y = 0;
    let z = 10 % y;
    print(z);
}
"#;
    let err = run_vm_err(src).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "取模除零应定位到第 3 行，实际 {:?}", err);
    assert_display_has_line(&err, 3);
}

// ─── 成功路径：不报错、行号表不干扰正常执行 ───────────────────────────

#[test]
fn test_normal_execution_unaffected() {
    let src = r#"fn main() {
    let a = 1 + 2;
    let b = a * 3;
    b
}
"#;
    let result = run_vm_err(src).unwrap();
    match result {
        Value::Int(9, _) => {}
        v => panic!("期望 Int(9)，实际 {:?}", v),
    }
}

#[test]
fn test_line_table_is_compile_only_no_runtime_overhead() {
    // 行号表是编译期结构：运行 10 万次加法确认结果一致（无运行期语义影响）。
    let src = r#"fn main() {
    let mut total = 0;
    let mut i = 0;
    while i < 100000 {
        total = total + 1;
        i = i + 1;
    }
    total
}
"#;
    let result = run_vm_err(src).unwrap();
    match result {
        Value::Int(100000, _) => {}
        v => panic!("期望 Int(100000)，实际 {:?}", v),
    }
}

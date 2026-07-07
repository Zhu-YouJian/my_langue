//! 元组类型集成测试。
//!
//! 覆盖元组的创建、解构、嵌套、函数返回多值、空元组与 Unit 等价性。
//! 同时验证解释器（Interpreter）与字节码 VM（路径 A 默认后端）两条路径。
//!
//! VM 新增 4 个 Op：`MakeTuple(n)`、`IsTuple(n)`、`TupleGet(i)`、`Try`（opcode 49-52）。
//! `Value::Tuple(Vec<Value>)` 变体表示非空元组；空元组 `()` 编译为 `Value::Unit`。
//!
//! ## 历史 bug（已修复）
//!
//! `let (a, b) = expr;` 语法曾存在 parser bug：`parse_stmt` 的无条件
//! `self.advance()` 在 LParen 解构路径下会错误消耗 `=` token，导致 `init` 为 None。
//! 已修复：`advance()` 改为只在 Identifier 路径下执行，LParen 路径不再多 advance。
//! 解构功能也可通过 `match (a, b) => ...` 路径验证（match pattern 走不同代码路径）。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// Run source through lexer → parser → HIR → interpreter.
fn run(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// Run source through the bytecode VM (path A default backend).
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

// ─── 1. 元组创建：解释器路径 ────────────────────────────────────────────────

#[test]
fn test_tuple_create() {
    let src = "(1, 2)";
    let result = run(src).unwrap();
    match result {
        Some(Value::Tuple(items)) => {
            assert_eq!(items.len(), 2, "二元组应有 2 个元素");
            match &items[0] {
                Value::Int(1) => {}
                v => panic!("第一个元素应为 Int(1), got {:?}", v),
            }
            match &items[1] {
                Value::Int(2) => {}
                v => panic!("第二个元素应为 Int(2), got {:?}", v),
            }
        }
        v => panic!("期望 Tuple, got {:?}", v),
    }
}

// ─── 2. 元组解构（let 语法）：解释器路径 ────────────────────────────────────

#[test]
fn test_tuple_destructure_let() {
    let src = r#"
        let (a, b) = (1, 2);
        a + b
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("期望 Int(3), got {:?}", v),
    }
}

// ─── 2b. 元组解构（match 路径）：解释器路径 ──────────────────────────────────
// match pattern 走不同代码路径，能正常解构元组。

#[test]
fn test_tuple_destructure_match() {
    let src = r#"
        let t = (1, 2);
        match t {
            (a, b) => a + b,
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("期望 Int(3), got {:?}", v),
    }
}

// ─── 3. 三元组：解释器路径 ──────────────────────────────────────────────────

#[test]
fn test_tuple_triple() {
    let src = r#"
        let t = (1, "hello", true);
        match t {
            (a, b, c) => if c { a } else { 0 },
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(1)) => {}
        v => panic!("期望 Int(1), got {:?}", v),
    }
}

// ─── 4. 嵌套元组：解释器路径 ────────────────────────────────────────────────
// 嵌套元组 ((1, 2), 3)：用 match 解构外层，再对内层元组用 TupleGet 访问。

#[test]
fn test_tuple_nested() {
    let src = r#"
        let t = ((1, 2), 3);
        match t {
            (inner, b) => match inner {
                (x, y) => x + y + b,
            },
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(6)) => {}
        v => panic!("期望 Int(6) 表示嵌套解构 1+2+3, got {:?}", v),
    }
}

// ─── 5. 函数返回元组：解释器路径 ────────────────────────────────────────────

#[test]
fn test_tuple_function_return() {
    let src = r#"
        fn make_pair() -> (i64, i64) {
            return (1, 2);
        }
        let t = make_pair();
        match t {
            (a, b) => a + b,
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("期望 Int(3), got {:?}", v),
    }
}

// ─── 6. 空元组等于 Unit：解释器路径 ─────────────────────────────────────────

#[test]
fn test_tuple_empty_unit() {
    let src = "()";
    let result = run(src).unwrap();
    match result {
        Some(Value::Unit) | None => {}
        v => panic!("期望 Unit 表示空元组, got {:?}", v),
    }
}

// ─── VM 路径（路径 A 默认后端）──────────────────────────────────────────────

#[test]
fn test_vm_tuple_create() {
    let src = "(1, 2)";
    let result = run_vm(src).unwrap();
    match result {
        Value::Tuple(items) => {
            assert_eq!(items.len(), 2, "VM: 二元组应有 2 个元素");
            match &items[0] {
                Value::Int(1) => {}
                v => panic!("VM: 第一个元素应为 Int(1), got {:?}", v),
            }
            match &items[1] {
                Value::Int(2) => {}
                v => panic!("VM: 第二个元素应为 Int(2), got {:?}", v),
            }
        }
        v => panic!("VM: 期望 Tuple, got {:?}", v),
    }
}

// ─── VM 路径：let 解构 ──────────────────────────────────────────────────────

#[test]
fn test_vm_tuple_destructure_let() {
    let src = r#"
        fn main() -> i64 {
            let (a, b) = (1, 2);
            a + b
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(3) => {}
        v => panic!("VM: 期望 Int(3), got {:?}", v),
    }
}

#[test]
fn test_vm_tuple_destructure_match() {
    let src = r#"
        fn main() -> i64 {
            let t = (1, 2);
            match t {
                (a, b) => a + b,
            }
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(3) => {}
        v => panic!("VM: 期望 Int(3), got {:?}", v),
    }
}

#[test]
fn test_vm_tuple_function_return() {
    let src = r#"
        fn make_pair() -> (i64, i64) {
            return (1, 2);
        }
        fn main() -> i64 {
            let t = make_pair();
            match t {
                (a, b) => a + b,
            }
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(3) => {}
        v => panic!("VM: 期望 Int(3), got {:?}", v),
    }
}

#[test]
fn test_vm_tuple_empty_unit() {
    let src = "fn main() -> () { () }";
    let result = run_vm(src).unwrap();
    match result {
        Value::Unit => {}
        v => panic!("VM: 期望 Unit 表示空元组, got {:?}", v),
    }
}

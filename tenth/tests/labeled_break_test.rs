//! M2.3：标签 break / continue（`break 'outer` / `continue 'outer`）测试。
//!
//! 覆盖：
//! - 语法：`'label: while/for/loop/do { ... }` 循环标签 + `break 'label` / `continue 'label`
//! - 嵌套循环标签 break / continue（两层、三层）
//! - 带值 break + 标签组合（`break 'outer val`）
//! - 未定义标签报错（编译期 TypeError）
//! - 标签在非循环外报错（编译期 TypeError）
//! - 嵌套循环同标签 shadow（内层标签遮蔽外层）
//! - 无标签 break/continue 回归（行为不变）
//! - VM/解释器 parity

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;

/// 解释器路径：lex → parse → lower → run。
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

/// 只做 lower（用于断言编译期错误）。
fn lower_error(src: &str) -> Result<(), String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ()).map_err(|e| e.to_string())
}

/// VM 路径：lex → parse → lower → bytecode → 执行 main。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("assert".into(), |_vm, args| {
        if let Some(Value::Bool(b)) = args.first() {
            assert!(*b, "assertion failed");
        }
        Ok(Value::Unit)
    });
    vm.add_native("assert_eq".into(), |_vm, args| {
        if args.len() >= 2 {
            assert_eq!(format!("{}", args[0]), format!("{}", args[1]));
        }
        Ok(Value::Unit)
    });

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

fn expect_int(v: Option<Value>, expected: i64, label: &str) {
    match v {
        Some(Value::Int(n, _)) => assert_eq!(n, expected, "{}: 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

/// VM 与解释器双路径结果一致。
fn assert_parity(src: &str, expected: i64, label: &str) {
    let interp = run(src).unwrap_or_else(|e| panic!("{}: 解释器执行失败: {}", label, e));
    expect_int(interp, expected, &format!("{}(解释器)", label));
    let vm = run_vm(src).unwrap_or_else(|e| panic!("{}: VM 执行失败: {}", label, e));
    match vm {
        Value::Int(n, _) => assert_eq!(n, expected, "{}(VM): 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}(VM): 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

// ── 1. 嵌套循环标签 break：while 嵌套 while ──

#[test]
fn labeled_break_while_nested_while() {
    // 'outer 外层 while；内层 while 满足条件时 break 'outer 直接跳出两层
    let src = r#"
        let mut i = 0;
        let mut s = 0;
        'outer: while i < 10 {
            let mut j = 0;
            while j < 10 {
                if j == 3 { break 'outer; }
                s = s + 1;
                j = j + 1;
            }
            i = i + 1;
        }
        s
    "#;
    // 内层 j=0,1,2 → s=3；j==3 时 break 'outer 跳出两层循环
    assert_parity(src, 3, "嵌套 while 标签 break");
}

// ── 2. 三层循环跳最外层 ──

#[test]
fn labeled_break_three_levels() {
    // loop → while → for 三层，最内层 break 'outer 直接跳出最外层
    let src = r#"
        let mut count = 0;
        'outer: loop {
            let mut i = 0;
            while i < 5 {
                for k in 0..5 {
                    count = count + 1;
                    if count == 7 { break 'outer; }
                }
                i = i + 1;
            }
            // 永不执行（break 'outer 直接跳出）
            count = count + 1000;
        }
        count
    "#;
    // count 累计到 7 即 break 'outer，此后 1000 不执行
    assert_parity(src, 7, "三层循环跳最外层");
}

// ── 3. 标签 continue：for 嵌套 while，continue 'outer 继续外层 ──

#[test]
fn labeled_continue_outer() {
    // 'outer: for i in 0..5；内层 while 在 i 为偶数时 continue 'outer 跳过外层体
    let src = r#"
        let mut s = 0;
        'outer: for i in 0..5 {
            let mut j = 0;
            while j < 3 {
                if i % 2 == 0 { continue 'outer; }
                s = s + 1;
                j = j + 1;
            }
            // i 为奇数时到达这里
            s = s + 10;
        }
        s
    "#;
    // i=0(偶): continue 'outer → 跳过；i=1(奇): j 循环 s+3，再 +10 → s=13
    // i=2(偶): continue；i=3(奇): s=13+3+10=26；i=4(偶): continue → s=26
    assert_parity(src, 26, "标签 continue 外层 for");
}

// ── 4. 带值 break + 标签组合 ──

#[test]
fn labeled_break_with_value() {
    // `break 'outer 99` —— 值与标签组合可解析、可编译、可运行（loop 不产出值，结果丢弃）
    let src = r#"
        let mut s = 0;
        'outer: loop {
            while s < 100 {
                s = s + 1;
                if s == 5 { break 'outer 99; }
            }
        }
        s
    "#;
    assert_parity(src, 5, "带值 break + 标签");
}

// ── 5. 未定义标签报错 ──

#[test]
fn undefined_label_errors() {
    let src = r#"
        while true {
            break 'unknown;
        }
    "#;
    let err = lower_error(src).expect_err("break 'unknown 应编译期报错");
    assert!(err.contains("未定义循环标签 'unknown'"), "错误消息不符: {}", err);

    let src2 = r#"
        for i in 0..5 {
            continue 'nope;
        }
    "#;
    let err2 = lower_error(src2).expect_err("continue 'nope 应编译期报错");
    assert!(err2.contains("未定义循环标签 'nope'"), "错误消息不符: {}", err2);
}

// ── 6. 标签在非循环外报错 ──

#[test]
fn label_outside_loop_errors() {
    let src = r#"
        let x = 1;
        break 'outer;
        x
    "#;
    let err = lower_error(src).expect_err("循环外的标签 break 应编译期报错");
    assert!(err.contains("不在任何循环内"), "错误消息不符: {}", err);

    let src2 = r#"
        fn f() -> i64 {
            continue 'outer;
            1
        }
        { f() }
    "#;
    let err2 = lower_error(src2).expect_err("函数内循环外的标签 continue 应编译期报错");
    assert!(err2.contains("不在任何循环内"), "错误消息不符: {}", err2);
}

// ── 7. 无标签 break/continue 回归 ──

#[test]
fn unlabeled_break_continue_regression() {
    // 与 parity_test 相同的语义：无标签 break/continue 行为不变
    let src = r#"
        fn test(n: i64) -> i64 {
            let mut s = 0;
            for i in 0..n {
                if i == 3 { break; }
                if i == 1 { continue; }
                s = s + i;
            }
            s
        }
        { test(10) }
    "#;
    // i=0: s=0; i=1: continue; i=2: s=2; i=3: break → s=2
    assert_parity(src, 2, "无标签 break/continue 回归");

    let src2 = r#"
        fn test(n: i64) -> i64 {
            let mut i = 0;
            let mut s = 0;
            while i < n {
                i = i + 1;
                if i == 5 { break; }
                if i == 2 { continue; }
                s = s + i;
            }
            s
        }
        { test(10) }
    "#;
    // i=1: s=1; i=2: continue; i=3: s=4; i=4: s=8; i=5: break → s=8
    assert_parity(src2, 8, "while 无标签 break/continue 回归");
}

// ── 8. 嵌套循环同标签 shadow ──

#[test]
fn label_shadow_inner_wins() {
    // 内层循环也用 'outer 标签——break 'outer 应跳内层（shadow 语义）
    let src = r#"
        let mut s = 0;
        'outer: while s < 100 {
            s = s + 1;
            'outer: while s < 100 {
                s = s + 1;
                if s == 3 { break 'outer; }
            }
            // 内层 break 'outer 后应到达这里（而非跳出外层）
            s = s + 100;
            if s > 200 { break; }
        }
        s
    "#;
    // 内层：s 1→2→3（break 'outer 跳内层）→ s=103；外层条件 s<100 false → 结束
    assert_parity(src, 103, "同标签 shadow 内层优先");
}

// ── 9. do-while 标签 ──

#[test]
fn labeled_do_while() {
    let src = r#"
        let mut s = 0;
        let mut i = 0;
        'outer: do {
            i = i + 1;
            let mut j = 0;
            do {
                j = j + 1;
                if j == 2 { break 'outer; }
                s = s + j;
            } while j < 5;
            s = s + 100;
        } while i < 5;
        s
    "#;
    // 内层 do：j=1 → s=1；j=2 → break 'outer 跳出外层 do
    assert_parity(src, 1, "do-while 标签 break");
}

// ── 10. for 循环标签 break（Range 迭代器） ──

#[test]
fn labeled_break_for_range() {
    let src = r#"
        let mut s = 0;
        'outer: for i in 0..10 {
            for j in 0..10 {
                if i + j >= 5 { break 'outer; }
                s = s + 1;
            }
        }
        s
    "#;
    // 枚举 (i,j)：(0,0),(0,1),(0,2),(0,3),(0,4) → s=5；i+j=5 时 break 'outer
    assert_parity(src, 5, "for 标签 break");
}

// ── 11. 标签 continue 在 while 中跳到外层 while ──

#[test]
fn labeled_continue_while_outer() {
    let src = r#"
        let mut s = 0;
        let mut i = 0;
        'outer: while i < 5 {
            i = i + 1;
            let mut j = 0;
            while j < 10 {
                j = j + 1;
                if j > i { continue 'outer; }
                s = s + 1;
            }
        }
        s
    "#;
    // i=1: j=1 (j>i? no) s=1; j=2>1 continue 'outer
    // i=2: j=1 s=2; j=2 s=3; j=3>2 continue
    // i=3: j=1 s=4; j=2 s=5; j=3 s=6; j=4>3 continue
    // i=4: j=1 s=7; j=2 s=8; j=3 s=9; j=4 s=10; j=5>4 continue
    // i=5: j=1 s=11; j=2 s=12; j=3 s=13; j=4 s=14; j=5 s=15; j=6>5 continue
    // i=5 后外层条件 i<5 false → 结束；总数 = 1+2+3+4+5 = 15
    assert_parity(src, 15, "continue 外层 while");
}

// ── 12. 标签 + 多层 if 内 break ──

#[test]
fn labeled_break_deep_in_blocks() {
    let src = r#"
        let mut s = 0;
        'outer: loop {
            let mut i = 0;
            while i < 10 {
                i = i + 1;
                if i > 1 {
                    if i > 2 {
                        if i > 3 {
                            break 'outer;
                        }
                        s = s + 30;
                    }
                    s = s + 300;
                }
                s = s + 1;
            }
        }
        s
    "#;
    // i=1: s=1; i=2: s=301; i=3: s=632; i=4: break 'outer
    // 解释：i=1 → s=0+1=1; i=2 → (i>1) (i>2 false) s=1+300+1=302... 
    // 实际追踪：i=1: s=1; i=2: 满足 i>1,不满足 i>2 → s=1+300=301, 再 +1=302;
    // i=3: 满足 i>2 → s=302+30=332, +300=632, +1=633; i=4: break 'outer
    assert_parity(src, 633, "多层 if 内标签 break");
}

// ── 13. 无标签 continue 在标签循环内仍跳最近循环（回归） ──

#[test]
fn unlabeled_continue_inside_labeled_loop() {
    let src = r#"
        let mut s = 0;
        'outer: for i in 0..5 {
            for j in 0..5 {
                if j == 1 { continue; }
                if j == 3 { continue 'outer; }
                s = s + 1;
            }
        }
        s
    "#;
    // 每个 i：j=0 → s+1；j=1 continue；j=2 → s+1；j=3 continue 'outer
    // 每个 i 贡献 2 → 5 次外层迭代 → 10
    assert_parity(src, 10, "无标签 continue 仍跳最近循环");
}

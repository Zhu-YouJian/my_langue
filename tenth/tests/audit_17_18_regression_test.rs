//! 回归测试：AUDIT #17（同函数多标签循环）/ #18（同函数双泛型枚举 match）
//!
//! 根因：bytecode.rs 局部变量槽位查找曾用 `position`（首匹配），同名变量
//! 重绑定（如两个循环都用 `j`、两个 match 都绑 `x`）会读写旧槽位，
//! 导致第二个构造取值错误 + 污染其他变量槽。rposition（最近绑定）修复后
//! 应全部正确。本文件把"原规避场景"（拆独立函数）合并回同一函数验证。

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
        if let Some(Value::Bool(b)) = args.first() { assert!(*b, "assertion failed"); }
        Ok(Value::Unit)
    });
    vm.add_native("assert_eq".into(), |_vm, args| {
        if args.len() >= 2 { assert_eq!(format!("{}", args[0]), format!("{}", args[1])); }
        Ok(Value::Unit)
    });

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures { vm.add_fn(name, closure_chunk); }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures { vm.add_fn(name, closure_chunk); }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else { Ok(Value::Unit) }
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

// ──────────────────────────────────────────────────────────────────────
// AUDIT #17：同一函数内连续多个带标签循环（原规避 = 拆独立函数）
// ──────────────────────────────────────────────────────────────────────

/// 同一 fn：带标签 while + break 'outer，随后带标签 for + continue 'outer。
/// 两个循环共用变量名 `j`（原缺陷：position 首匹配让第二个循环的 j 读写
/// 第一个循环的 j 槽 → s2 应为 26 实测 50）。
#[test]
fn labeled_loops_two_in_same_fn() {
    let src = r#"
fn both() -> i64 {
    // 第一个标签循环：break 'outer
    let mut s = 0;
    'outer: while true {
        let mut j = 0;
        while j < 10 {
            if j == 3 { break 'outer; }
            s = s + 1;
            j = j + 1;
        }
    }
    // 第二个标签循环：continue 'outer（共用变量名 j）
    let mut s2 = 0;
    'outer2: for k in 0..5 {
        let mut j = 0;
        while j < 3 {
            j = j + 1;
            if k % 2 == 0 { continue 'outer2; }
            s2 = s2 + 1;
        }
        s2 = s2 + 10;
    }
    s * 1000 + s2
}
fn main() {
    let r = both();
    assert_eq(r, 3026);
    r
}
"#;
    assert_parity(src, 3026, "同函数两个标签循环(break+continue)");
}

/// 两个标签循环使用同名标签 'outer（标签 shadow 场景）。
#[test]
fn labeled_loops_same_label_name() {
    let src = r#"
fn both() -> i64 {
    let mut s = 0;
    'outer: while s < 5 {
        let mut j = 0;
        while j < 10 {
            if j == 3 { break 'outer; }
            s = s + 1;
            j = j + 1;
        }
    }
    let mut s2 = 0;
    'outer: for k in 0..5 {
        let mut j = 0;
        while j < 3 {
            j = j + 1;
            if k % 2 == 0 { continue 'outer; }
            s2 = s2 + 1;
        }
        s2 = s2 + 10;
    }
    s * 1000 + s2
}
fn main() {
    let r = both();
    assert_eq(r, 3026);
    r
}
"#;
    assert_parity(src, 3026, "同函数同名标签两个循环");
}

/// 同一函数内三个标签循环（while + for + loop 混合，嵌套）。
#[test]
fn labeled_loops_three_mixed() {
    let src = r#"
fn three() -> i64 {
    // 1) while + break 'a
    let mut a = 0;
    'a: while a < 100 {
        let mut j = 0;
        while j < 10 {
            if j == 3 { break 'a; }
            a = a + 1;
            j = j + 1;
        }
    }
    // 2) for + continue 'b（含同名 j）
    let mut b = 0;
    'b: for k in 0..4 {
        let mut j = 0;
        while j < 2 {
            j = j + 1;
            if k % 2 == 0 { continue 'b; }
            b = b + 1;
        }
        b = b + 100;
    }
    // 3) loop + break 'c（嵌套两层，最内层 break 'c 直接跳出）
    let mut c = 0;
    'c: loop {
        let mut m = 0;
        while m < 20 {
            if m == 5 { break 'c; }
            c = c + 1;
            m = m + 1;
        }
        c = c + 1000;
    }
    a * 10000 + b * 100 + c
}
fn main() {
    let r = three();
    assert_eq(r, 50405);
    r
}
"#;
    // a=3（j==3 时 break 'a）；b：k=1,3（奇数）内层 while 两轮各 b+=1 → 4，再各 +100 → 204；
    // k=0,2（偶数）continue 'b 跳过。c：m==5 时 break 'c → 5。
    // 3*10000 + 204*100 + 5 = 30000 + 20400 + 5 = 50405（解释器权威正确路径）
    assert_parity(src, 50405, "同函数三个标签循环");
}

// ──────────────────────────────────────────────────────────────────────
// AUDIT #18：同一函数内连续两个泛型枚举 match（i64 + str 绑定）
// ──────────────────────────────────────────────────────────────────────

/// 同函数：先 match Maybe<i64>（绑 x）、再 match Maybe<str>（绑 x）。
/// 两个 match 绑定同名 x（原缺陷：position 首匹配让第二个 x 读到第一个
/// x 的槽 → "hi" 实测 42）。
#[test]
fn generic_enum_two_matches_same_fn() {
    let src = r#"
enum Maybe<T> { Just(T), Nothing }
fn both() -> (i64, str) {
    let mb = Maybe<i64>::Just(42);
    let v = match mb { Maybe::Just(x) => x, Maybe::Nothing => -1 };
    let ms = Maybe<str>::Just("hi");
    let vs = match ms { Maybe::Just(x) => x, Maybe::Nothing => "empty" };
    (v, vs)
}
fn main() {
    let (v, vs) = both();
    assert_eq(v, 42);
    assert_eq(vs, "hi");
    v
}
"#;
    assert_parity(src, 42, "同函数双泛型枚举 match(i64+str)");
}

/// 多字段 + 嵌套泛型（Wrap<Vec<i64>> 解构），let 绑定 x 与 match 绑定 x 同名 shadow。
#[test]
fn generic_enum_two_matches_multi_nested() {
    let src = r#"
enum Maybe<T> { Just(T), Nothing }
enum Pair<A, B> { Make(A, B), None }
enum Wrap<T> { Item(T) }
fn both() -> (i64, i64, str) {
    // match 1：双字段泛型枚举，返回 a（多字段解构）
    let p = Pair<i64, str>::Make(7, "seven");
    let x = match p {
        Pair::Make(a, b) => a,
        Pair::None => -1,
    };
    // match 2：Maybe<str>，绑定 x（与上面 let 绑定的 x 同名 → shadow）
    let ms = Maybe<str>::Just("hi");
    let vs = match ms { Maybe::Just(x) => x, Maybe::Nothing => "empty" };
    // match 3：嵌套泛型 Wrap<Vec<i64>>，绑定 v
    let w = Wrap<Vec<i64>>::Item([10, 20, 30]);
    let n = match w {
        Wrap::Item(v) => v.len(),
    };
    (x, n, vs)
}
fn main() {
    let (x, n, vs) = both();
    assert_eq(x, 7);
    assert_eq(n, 3);
    assert_eq(vs, "hi");
    x
}
"#;
    assert_parity(src, 7, "同函数多字段+嵌套泛型枚举 match");
}

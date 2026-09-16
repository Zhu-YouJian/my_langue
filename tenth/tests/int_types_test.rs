use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::{Type, BaseType};
use tenth::runtime::value::{Value, promote_int_dtype};
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use std::rc::Rc;
use std::cell::RefCell;

fn lower_code(src: &str) -> tenth::hir::hir::HirProgram {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).unwrap()
}

/// 尝试 lower；返回 Err(String) 供编译期错误断言。
fn try_lower(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

fn setup_vm(hir: &tenth::hir::hir::HirProgram) -> Vm {
    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a} "); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("format".into(), |_vm, args| {
        let mut s = String::new();
        for a in args { s.push_str(&a.to_string()); }
        Ok(Value::String(s))
    });
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile(func) {
            vm.add_fn(func.name.clone(), chunk);
            for (name, closure_chunk) in closures { vm.add_fn(name, closure_chunk); }
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile_main(expr) {
            vm.add_fn("main".into(), chunk);
            for (name, closure_chunk) in closures { vm.add_fn(name, closure_chunk); }
        }
    }
    vm
}

/// **默认后端路径**（与 CLI 一致：JIT，失败再回退 VM）。
fn run_default(src: &str) -> Result<Value, String> {
    let hir = try_lower(src)?;
    let mut vm = setup_vm(&hir);
    if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// 纯 VM 字节码路径（不经 JIT）。
fn run_vm(src: &str) -> Result<Value, String> {
    let hir = try_lower(src)?;
    let mut vm = setup_vm(&hir);
    if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// 树遍解释器路径（= `TENTH_NO_VM=1`）。
fn run_interp(src: &str) -> Result<Value, String> {
    let hir = try_lower(src)?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir)
        .map_err(|e| e.to_string())
        .map(|opt| opt.unwrap_or(Value::Unit))
}

fn int_str(v: &Value) -> String {
    match v { Value::Int(n, _) => n.to_string(), other => format!("{other}") }
}

/// 三路径同值断言（差分 + 金标准）。
fn assert3(src: &str, expected: i64, label: &str) {
    let d = run_default(src).unwrap_or_else(|e| panic!("[{label}] 默认后端失败: {e}"));
    let v = run_vm(src).unwrap_or_else(|e| panic!("[{label}] VM 失败: {e}"));
    let i = run_interp(src).unwrap_or_else(|e| panic!("[{label}] 解释器失败: {e}"));
    let ds = int_str(&d); let vs = int_str(&v); let is = int_str(&i);
    assert_eq!(ds, vs, "[{label}] 默认后端/VM 结果不一致: {ds} vs {vs}");
    assert_eq!(ds, is, "[{label}] 默认后端/解释器 结果不一致: {ds} vs {is}");
    assert_eq!(ds, expected.to_string(), "[{label}] 结果不等于金标准");
}

/// 三路径同错误断言（错误必须响亮，不许静默/回绕）。
fn assert3_err(src: &str, needle: &str, label: &str) {
    let d = run_default(src);
    let v = run_vm(src);
    let i = run_interp(src);
    for (name, r) in [("默认后端", &d), ("VM", &v), ("解释器", &i)] {
        let e = r.as_ref().err().unwrap_or_else(|| panic!("[{label}] {name} 应报错，实际 {:?}", r));
        assert!(e.contains(needle), "[{label}] {name} 错误消息不含 {needle:?}: {e}");
    }
}

fn main_expr_type(hir: &tenth::hir::hir::HirProgram) -> Option<Type> {
    hir.main_expr.as_ref().map(|e| e.ty.clone())
}

#[test]
fn test_int_default_is_i32() {
    let hir = lower_code("42");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I32));
}

#[test]
fn test_int_u8_suffix() {
    let hir = lower_code("42u8");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::U8));
}

#[test]
fn test_int_i64_suffix() {
    let hir = lower_code("42i64");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I64));
}

#[test]
fn test_int_u32_suffix() {
    let hir = lower_code("42u32");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::U32));
}

#[test]
fn test_int_i16_suffix() {
    let hir = lower_code("42i16");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I16));
}

#[test]
fn test_u8_max_ok() {
    let hir = lower_code("255u8");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::U8));
}

#[test]
fn test_u8_overflow_fails() {
    let mut lexer = Lexer::new("256u8");
    let result = lexer.tokenize();
    assert!(result.is_err(), "256u8 应超出 u8 范围报错");
}

#[test]
fn test_i8_overflow_fails() {
    let mut lexer = Lexer::new("128i8");
    let result = lexer.tokenize();
    assert!(result.is_err(), "128i8 应超出 i8 范围报错");
}

#[test]
fn test_i8_max_ok() {
    let hir = lower_code("127i8");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I8));
}

#[test]
fn test_u16_max_ok() {
    let hir = lower_code("65535u16");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::U16));
}

#[test]
fn test_u16_overflow_fails() {
    let mut lexer = Lexer::new("65536u16");
    let result = lexer.tokenize();
    assert!(result.is_err(), "65536u16 应超出 u16 范围报错");
}

#[test]
fn test_value_int_dtype_preserved() {
    let v = Value::Int(42, BaseType::U8);
    assert_eq!(v.type_of(), Type::Base(BaseType::U8));
}

#[test]
fn test_value_int_i64_dtype() {
    let v = Value::Int(1000000, BaseType::I64);
    assert_eq!(v.type_of(), Type::Base(BaseType::I64));
}

#[test]
fn test_value_int_default_i32() {
    let v = Value::Int(42, BaseType::I32);
    assert_eq!(v.type_of(), Type::Base(BaseType::I32));
}

// ═══════════════════════════════════════════════════════════════════════════
// AUDIT-11.4.53：端到端用例（lexer→parser→lower→bytecode→VM/JIT/解释器）
//
// 覆盖缺口：本文件此前 14 项**全部停在 HIR/Value 层**（手构造的值天然带 dtype），
// 无一项走端到端——这正是 11.4.53「整型算术被硬限在 i32」整条逃逸的原因。
// ═══════════════════════════════════════════════════════════════════════════

/// ① 无后缀字面量超 i32 范围 → 自动提升为 i64（算术不再被 i32 硬限）。
#[test]
fn e2e_unsuffixed_literal_promotes_to_i64() {
    assert3("fn main() -> Int { let a = 3000000000; a + a }", 6000000000, "①自动提升");
}

/// ① 且 `println` 仍正常（值不被截断）。
#[test]
fn e2e_unsuffixed_literal_prints_full_value() {
    assert3("fn main() -> Int { let a = 3000000000; let b = a; b }", 3000000000, "①println 路径");
}

/// ② 显式 i64 后缀：`2000000000i64 * 100i64`。
#[test]
fn e2e_i64_suffix_arithmetic() {
    assert3("fn main() -> Int { 2000000000i64 * 100i64 }", 200000000000, "②i64 后缀");
}

/// ③ `let x: i64` 标注**强制** init 的 dtype（R1 卡点）。
/// 字面量 2000000000 在 i32 范围内、lexer 不提升；若不按注解改写则运行期仍是 I32，
/// `x * 100` 会被 i32 范围检查误报溢出。
#[test]
fn e2e_let_i64_annotation_forces_dtype() {
    assert3("fn main() -> Int { let x: i64 = 2000000000; x * 100 }", 200000000000, "③let i64 标注");
}

/// ④ `i64` 形参生效（实参字面量按形参 dtype 强制）。
#[test]
fn e2e_i64_param() {
    assert3("fn scale(x: i64) -> i64 { x * 100 }\nfn main() -> Int { scale(2000000000) }",
        200000000000, "④i64 形参");
}

/// ⑤ `i64` 返回类型生效（体内字面量按返回 dtype 求值）。
#[test]
fn e2e_i64_return_type() {
    assert3("fn big() -> i64 { 2000000000 * 1000 }\nfn main() -> Int { big() }",
        2000000000000, "⑤i64 返回");
}

/// ⑨ radix 字面量（R5）同样遵守「默认 i32、超范围提升 i64」。
#[test]
fn e2e_radix_literal_promotes_like_decimal() {
    assert3("fn main() -> Int { let r = 0x1_0000_0000; r + r }", 8589934592, "⑨radix 提升");
    assert3("fn main() -> Int { 0xFFFF }", 65535, "⑨radix i32 内不提");
}

/// ⑥ 混合运算**交换律**（R4）：`x_i64 + 1` 与 `1 + x_i64` 同值。
#[test]
fn e2e_mixed_arithmetic_commutative() {
    // 加法
    assert3("fn main() -> Int { let x: i64 = 2000000000; x + 1 }", 2000000001, "⑥a x+1");
    assert3("fn main() -> Int { let x: i64 = 2000000000; 1 + x }", 2000000001, "⑥b 1+x");
    // 乘法
    assert3("fn main() -> Int { let x: i64 = 2000000000; x * 3 }", 6000000000, "⑥c x*3");
    assert3("fn main() -> Int { let x: i64 = 2000000000; 3 * x }", 6000000000, "⑥d 3*x");
    // 减法/除法在 i64 层同样不被 i32 硬限（顺序敏感但结果定义明确）
    assert3("fn main() -> Int { let x: i64 = 3000000000; x - 1 }", 2999999999, "⑥e x-1");
}

/// ⑦ i32 溢出**仍然响亮报错**（护城河红线：不许改成回绕/饱和）。
#[test]
fn e2e_i32_overflow_still_loud() {
    assert3_err("fn main() -> Int { let p: i32 = 2000000000; p * 2 }",
        "溢出 i32 范围", "⑦i32 溢出");
    assert3_err("fn main() -> Int { let a = 2147483647; let b = 1; a + b }",
        "溢出 i32 范围", "⑦无名 i32 溢出");
    // i64 层溢出同样响亮（@checked_*）
    assert3_err("fn main() -> Int { let a = 9223372036854775807i64; let b = 1i64; a + b }",
        "整数运算结果溢出", "⑦i64 层溢出");
}

/// ⑧ 窄 dtype（i8/i16/u8）超范围行为：运算结果越界 → 响亮报错；不越界 → 正常。
#[test]
fn e2e_narrow_dtype_range() {
    assert3("fn main() -> Int { let s: i8 = 100; let t: i8 = 27; s + t }", 127, "⑧i8 上界内");
    assert3_err("fn main() -> Int { let s: i8 = 100; let t: i8 = 100; s + t }",
        "溢出 i8 范围", "⑧i8 越界");
    assert3_err("fn main() -> Int { let s: u8 = 200; let t: u8 = 200; s + t }",
        "溢出 u8 范围", "⑧u8 越界");
    assert3_err("fn main() -> Int { let s: i16 = 30000; let t: i16 = 10000; s + t }",
        "溢出 i16 范围", "⑧i16 越界");
}

/// ⑧ 注解放不下的字面量 → **编译期错误**（手册承诺「超出范围报编译期错误」）。
#[test]
fn e2e_annotation_out_of_range_is_compile_error() {
    let e = try_lower("fn main() -> Int { let x: u8 = 300; x }")
        .err().expect("300 放不进 u8，应编译期报错");
    assert!(e.contains("超出 u8 范围"), "错误消息不符: {e}");
    let e = try_lower("fn main() -> Int { let x: i8 = 128; x }")
        .err().expect("128 放不进 i8，应编译期报错");
    assert!(e.contains("超出 i8 范围"), "错误消息不符: {e}");
    // 反向：`-2147483648` 是合法 i32（负号取值范围检查须看取负后的值）
    assert!(try_lower("fn main() -> Int { let x: i32 = -2147483648; x }").is_ok(),
        "-2147483648 是合法 i32 字面量");
}

/// R4 秩表：混合提升必须**可交换**且「同宽异号 → 下一更宽有符号」。
#[test]
fn promote_int_dtype_rank_table_and_commutativity() {
    use BaseType::*;
    // rank：更宽者胜
    assert_eq!(promote_int_dtype(I32, I64), I64);
    assert_eq!(promote_int_dtype(I8, I32), I32);
    assert_eq!(promote_int_dtype(U8, U16), U16);
    assert_eq!(promote_int_dtype(I8, U8), I16);
    // 同 rank 异号 → 下一更宽有符号
    assert_eq!(promote_int_dtype(U8, I8), I16);
    assert_eq!(promote_int_dtype(I16, U16), I32);
    assert_eq!(promote_int_dtype(U32, I32), I64);
    assert_eq!(promote_int_dtype(U64, I64), I64);
    // 同型 → 自身
    assert_eq!(promote_int_dtype(I8, I8), I8);
    assert_eq!(promote_int_dtype(U64, U64), U64);
    // 交换律：全组合对称
    let all = [I8, I16, I32, I64, U8, U16, U32, U64];
    for &l in &all {
        for &r in &all {
            assert_eq!(promote_int_dtype(l, r), promote_int_dtype(r, l),
                "promote_int_dtype 不可交换: {l:?}, {r:?}");
        }
    }
}

/// HIR 静态 dtype 也遵守同一可交换秩表（静态/运行期同源，防双源漂移）。
#[test]
fn hir_mixed_operand_dtype_commutative() {
    let hir_l = lower_code("fn main() -> Int { let x: i64 = 1; x + 1 }");
    let hir_r = lower_code("fn main() -> Int { let x: i64 = 1; 1 + x }");
    let fl = hir_l.functions.iter().find(|f| f.name == "main").unwrap();
    let fr = hir_r.functions.iter().find(|f| f.name == "main").unwrap();
    assert_eq!(fl.body.ty, Type::Base(BaseType::I64), "x + 1 应为 I64");
    assert_eq!(fl.body.ty, fr.body.ty, "x + 1 与 1 + x 的 HIR dtype 必须一致（交换律）");
}

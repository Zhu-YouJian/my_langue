//! 哈希函数测试 — SHA-256/SHA-512/MD5
//!
//! 覆盖 6 个 native：
//!   - sha256(data: Vec) / sha256_str(s: String)
//!   - sha512(data: Vec) / sha512_str(s: String)
//!   - md5(data: Vec)    / md5_str(s: String)
//!
//! 测试向量来自：
//!   - SHA-2: RFC 6234（SHA-256/SHA-512 空串与 "abc"）
//!   - MD5:   RFC 1321（空串与 "abc"）
//!
//! 双侧验证（VM + 解释器）确保 native 注册对齐。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;

/// 通过 VM 执行 .th 源码，返回结果。使用 register_all_natives 注册全部 native。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    tenth::runtime::natives::register_all_natives(&mut vm);

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
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

/// 通过解释器执行 .th 源码，返回结果。
fn run_interp(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut interp = Interpreter::new(&hir);
    match interp.execute_program(&hir) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Ok(Value::Unit),
        Err(e) => Err(e.to_string()),
    }
}

/// 提取 String 值
fn as_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => panic!("期望 String，实际: {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 标准测试向量：SHA-256（RFC 6234）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sha256_empty_vm() {
    // sha256(b"") == e3b0c442...
    let v = run_vm("sha256([])").expect("VM 执行失败");
    assert_eq!(as_str(&v), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

#[test]
fn test_sha256_abc_vm() {
    // sha256(b"abc") == ba7816bf...
    // "abc" = [97, 98, 99]
    let v = run_vm("sha256([97, 98, 99])").expect("VM 执行失败");
    assert_eq!(as_str(&v), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

#[test]
fn test_sha256_empty_interp() {
    let v = run_interp("sha256([])").expect("解释器执行失败");
    assert_eq!(as_str(&v), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

#[test]
fn test_sha256_abc_interp() {
    let v = run_interp("sha256([97, 98, 99])").expect("解释器执行失败");
    assert_eq!(as_str(&v), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

// ══════════════════════════════════════════════════════════════════════
// 标准测试向量：SHA-512（RFC 6234）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sha512_empty_vm() {
    let v = run_vm("sha512([])").expect("VM 执行失败");
    assert_eq!(as_str(&v), "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}

#[test]
fn test_sha512_abc_vm() {
    let v = run_vm("sha512([97, 98, 99])").expect("VM 执行失败");
    assert_eq!(as_str(&v), "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
}

#[test]
fn test_sha512_empty_interp() {
    let v = run_interp("sha512([])").expect("解释器执行失败");
    assert_eq!(as_str(&v), "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
}

#[test]
fn test_sha512_abc_interp() {
    let v = run_interp("sha512([97, 98, 99])").expect("解释器执行失败");
    assert_eq!(as_str(&v), "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
}

// ══════════════════════════════════════════════════════════════════════
// 标准测试向量：MD5（RFC 1321）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_md5_empty_vm() {
    let v = run_vm("md5([])").expect("VM 执行失败");
    assert_eq!(as_str(&v), "d41d8cd98f00b204e9800998ecf8427e");
}

#[test]
fn test_md5_abc_vm() {
    let v = run_vm("md5([97, 98, 99])").expect("VM 执行失败");
    assert_eq!(as_str(&v), "900150983cd24fb0d6963f7d28e17f72");
}

#[test]
fn test_md5_empty_interp() {
    let v = run_interp("md5([])").expect("解释器执行失败");
    assert_eq!(as_str(&v), "d41d8cd98f00b204e9800998ecf8427e");
}

#[test]
fn test_md5_abc_interp() {
    let v = run_interp("md5([97, 98, 99])").expect("解释器执行失败");
    assert_eq!(as_str(&v), "900150983cd24fb0d6963f7d28e17f72");
}

// ══════════════════════════════════════════════════════════════════════
// _str 便捷版：与 Vec 版本一致性验证
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sha256_str_consistency_vm() {
    // sha256_str("abc") == sha256([97, 98, 99])
    let v_str = run_vm("sha256_str(\"abc\")").expect("VM 执行失败");
    let v_vec = run_vm("sha256([97, 98, 99])").expect("VM 执行失败");
    assert_eq!(as_str(&v_str), as_str(&v_vec));
    assert_eq!(as_str(&v_str), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

#[test]
fn test_md5_str_empty_consistency_vm() {
    // md5_str("") == md5([])
    let v_str = run_vm("md5_str(\"\")").expect("VM 执行失败");
    let v_vec = run_vm("md5([])").expect("VM 执行失败");
    assert_eq!(as_str(&v_str), as_str(&v_vec));
    assert_eq!(as_str(&v_str), "d41d8cd98f00b204e9800998ecf8427e");
}

#[test]
fn test_sha512_str_consistency_interp() {
    let v_str = run_interp("sha512_str(\"abc\")").expect("解释器执行失败");
    let v_vec = run_interp("sha512([97, 98, 99])").expect("解释器执行失败");
    assert_eq!(as_str(&v_str), as_str(&v_vec));
}

#[test]
fn test_md5_str_consistency_interp() {
    let v_str = run_interp("md5_str(\"abc\")").expect("解释器执行失败");
    assert_eq!(as_str(&v_str), "900150983cd24fb0d6963f7d28e17f72");
}

// ══════════════════════════════════════════════════════════════════════
// 输入验证：错误参数类型应返回 Err
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sha256_wrong_arg_type_vm() {
    // sha256 传入 String 应报错（期望 Vec）
    let result = run_vm("sha256(\"not a vec\")");
    assert!(result.is_err(), "应返回错误，实际: {:?}", result);
}

#[test]
fn test_md5_str_wrong_arg_type_vm() {
    // md5_str 传入 Vec 应报错（期望 String）
    let result = run_vm("md5_str([1, 2, 3])");
    assert!(result.is_err(), "应返回错误，实际: {:?}", result);
}

#[test]
fn test_sha256_wrong_arg_type_interp() {
    let result = run_interp("sha256(\"not a vec\")");
    assert!(result.is_err(), "应返回错误，实际: {:?}", result);
}

// ══════════════════════════════════════════════════════════════════════
// VM 与解释器结果一致性（parity）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_parity_sha256_abc() {
    let v_vm = run_vm("sha256([97, 98, 99])").expect("VM 执行失败");
    let v_interp = run_interp("sha256([97, 98, 99])").expect("解释器执行失败");
    assert_eq!(as_str(&v_vm), as_str(&v_interp));
}

#[test]
fn test_parity_sha512_str() {
    let v_vm = run_vm("sha512_str(\"hello world\")").expect("VM 执行失败");
    let v_interp = run_interp("sha512_str(\"hello world\")").expect("解释器执行失败");
    assert_eq!(as_str(&v_vm), as_str(&v_interp));
}

#[test]
fn test_parity_md5_longer_input() {
    // 测试较长输入的 parity
    let src = "md5([104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100])"; // "hello world"
    let v_vm = run_vm(src).expect("VM 执行失败");
    let v_interp = run_interp(src).expect("解释器执行失败");
    assert_eq!(as_str(&v_vm), as_str(&v_interp));
}

// ══════════════════════════════════════════════════════════════════════
// 输出格式验证：小写 hex
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_sha256_output_is_lowercase_hex() {
    // 输出应全部为小写十六进制 [0-9a-f]
    let v = run_vm("sha256([255, 255, 255])").expect("VM 执行失败");
    let s = as_str(&v);
    assert_eq!(s.len(), 64, "SHA-256 输出长度应为 64");
    assert!(s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "应全为小写 hex，实际: {}", s);
}

#[test]
fn test_sha512_output_is_lowercase_hex() {
    let v = run_vm("sha512([0, 1, 2])").expect("VM 执行失败");
    let s = as_str(&v);
    assert_eq!(s.len(), 128, "SHA-512 输出长度应为 128");
    assert!(s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "应全为小写 hex，实际: {}", s);
}

#[test]
fn test_md5_output_is_lowercase_hex() {
    let v = run_vm("md5([255])").expect("VM 执行失败");
    let s = as_str(&v);
    assert_eq!(s.len(), 32, "MD5 输出长度应为 32");
    assert!(s.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "应全为小写 hex，实际: {}", s);
}

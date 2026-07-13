//! B批基本功核查第四大点：字符串/文本处理测试
//!
//! 覆盖第 38-44 项：
//! - f"..." 模板字符串（第 41 项）
//! - format() 命名参数（第 38 项）
//! - format() 格式说明符 {:>5}/{:.2f}（第 39 项）
//! - format() 越界报错（第 40 项）
//! - Unicode NFC/NFD 规范化（第 42 项）
//! - UTF-8/UTF-16/GBK 编码转换（第 43 项）
//! - Base64/Hex/URL 编解码（第 44 项）

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

/// 提取 Vec 值为 Vec<i64>
fn as_i64_vec(v: &Value) -> Vec<i64> {
    match v {
        Value::Vec(arr) => arr.borrow().iter().map(|e| match e {
            Value::Int(n, _) => *n,
            _ => 0,
        }).collect(),
        _ => panic!("期望 Vec，实际: {:?}", v),
    }
}

/// 从 Result::Ok 中提取内部值；若是 Err 则 panic。
fn unwrap_ok(v: &Value) -> Value {
    match v {
        Value::Enum { variant, fields, .. } => {
            if variant == "Ok" || variant == "ok" {
                let f = fields.borrow();
                if let Some((_, val)) = f.first() {
                    return val.clone();
                }
            }
            panic!("期望 Result::Ok，实际: {:?}", v);
        }
        _ => panic!("期望 Result，实际: {:?}", v),
    }
}

/// 断言 Value 是 Result::Err
fn assert_is_err(v: &Value) {
    match v {
        Value::Enum { variant, .. } => {
            assert!(variant == "Err" || variant == "err",
                "期望 Result::Err，实际: {:?}", v);
        }
        _ => panic!("期望 Result，实际: {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 第 41 项：f"..." 模板字符串
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_fstring_basic_vm() {
    let src = "fn main() { let name = \"world\"; f\"hello {name}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello world");
}

#[test]
fn test_fstring_basic_interp() {
    let src = "fn main() { let name = \"world\"; f\"hello {name}\" }";
    let v = run_interp(src).expect("解释器执行失败");
    assert_eq!(as_str(&v), "hello world");
}

#[test]
fn test_fstring_no_interpolation() {
    // f"hello"（无插值）应等同普通字符串 "hello"
    let src = "fn main() { f\"hello\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn test_fstring_multiple_vars() {
    let src = "fn main() { let x = 1; let y = 2; f\"{x} + {y} = 3\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "1 + 2 = 3");
}

// ══════════════════════════════════════════════════════════════════════
// 第 38 项：format() 命名参数
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_format_named_arg_vm() {
    // 使用原始字符串 r"..." 避免 {name} 被 Tenth 字符串插值解析
    let v = run_vm("format(r\"hello {name}\", \"name\", \"world\")").expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello world");
}

#[test]
fn test_format_named_arg_interp() {
    let v = run_interp("format(r\"hello {name}\", \"name\", \"world\")").expect("解释器执行失败");
    assert_eq!(as_str(&v), "hello world");
}

#[test]
fn test_format_mixed_positional_and_named() {
    // 1 个位置占位符 {} + 1 个命名占位符 {name}
    // 位置参数: 42，命名参数: name=answer
    let v = run_vm("format(r\"{} is {name}\", 42, \"name\", \"answer\")").expect("VM 执行失败");
    assert_eq!(as_str(&v), "42 is answer");
}

#[test]
fn test_format_named_missing_error() {
    // 命名参数不存在时应报错
    let result = run_vm("format(r\"hello {missing}\", \"name\", \"world\")");
    assert!(result.is_err(), "应返回错误，实际: {:?}", result);
}

// ══════════════════════════════════════════════════════════════════════
// 第 39 项：format() 格式说明符
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_format_width_right_align() {
    // {:>5} 右对齐宽度 5
    let v = run_vm("format(\"{:>5}\", 42)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "   42");
}

#[test]
fn test_format_width_left_align() {
    // {:<5} 左对齐宽度 5
    let v = run_vm("format(\"{:<5}\", 42)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "42   ");
}

#[test]
fn test_format_width_center() {
    // {:^5} 居中宽度 5
    let v = run_vm("format(\"{:^5}\", 42)").expect("VM 执行失败");
    assert_eq!(as_str(&v), " 42  ");
}

#[test]
fn test_format_float_precision() {
    // {:.2f} 浮点保留 2 位小数
    let v = run_vm("format(\"{:.2f}\", 3.14159)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "3.14");
}

#[test]
fn test_format_float_precision_interp() {
    let v = run_interp("format(\"{:.2f}\", 3.14159)").expect("解释器执行失败");
    assert_eq!(as_str(&v), "3.14");
}

// ══════════════════════════════════════════════════════════════════════
// 第 40 项：format() 越界报错（不再原样输出 {placeholder}）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_format_out_of_bounds_error_vm() {
    let result = run_vm("format(\"{} {}\", 1)");
    assert!(result.is_err(), "应返回越界错误，实际: {:?}", result);
}

#[test]
fn test_format_out_of_bounds_error_interp() {
    let result = run_interp("format(\"{} {}\", 1)");
    assert!(result.is_err(), "应返回越界错误，实际: {:?}", result);
}

// ══════════════════════════════════════════════════════════════════════
// 第 42 项：Unicode NFC/NFD 规范化
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_unicode_nfc_vm() {
    // NFD（分解）→ NFC（组合）：e + ´ → é
    let nfd = "e\u{0301}";  // U+0065 + U+0301
    let src = format!("unicode_nfc(\"{}\")", nfd);
    let v = run_vm(&src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "\u{00E9}");  // é (U+00E9)
}

#[test]
fn test_unicode_nfd_vm() {
    // NFC（组合）→ NFD（分解）：é → e + ´
    let nfc = "\u{00E9}";  // é (U+00E9)
    let src = format!("unicode_nfd(\"{}\")", nfc);
    let v = run_vm(&src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "e\u{0301}");  // e + ´
}

#[test]
fn test_unicode_nfc_interp() {
    let nfd = "e\u{0301}";
    let src = format!("unicode_nfc(\"{}\")", nfd);
    let v = run_interp(&src).expect("解释器执行失败");
    assert_eq!(as_str(&v), "\u{00E9}");
}

#[test]
fn test_unicode_nfc_idempotent() {
    // 已是 NFC 的字符串再 NFC 应不变
    let v = run_vm("unicode_nfc(\"hello\")").expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello");
}

// ══════════════════════════════════════════════════════════════════════
// 第 43 项：编码转换 UTF-8/UTF-16/GBK
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_str_to_utf16_basic() {
    let v = run_vm("str_to_utf16(\"hello\")").expect("VM 执行失败");
    assert_eq!(as_i64_vec(&v), vec![104, 101, 108, 108, 111]);
}

#[test]
fn test_utf16_to_str_roundtrip() {
    // "hello" → utf16 → str 应还原
    let v = run_vm("utf16_to_str(str_to_utf16(\"hello\"))").expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn test_str_to_bytes_basic() {
    let v = run_vm("str_to_bytes(\"AB\")").expect("VM 执行失败");
    assert_eq!(as_i64_vec(&v), vec![65, 66]);
}

#[test]
fn test_bytes_to_str_roundtrip() {
    let v = run_vm("bytes_to_str(str_to_bytes(\"hello\"))").expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn test_utf16_chinese() {
    // 中文字符在 UTF-16 中为单码元（BMP 内）
    let v = run_vm("str_to_utf16(\"你\")").expect("VM 执行失败");
    assert_eq!(as_i64_vec(&v), vec![0x4F60]);
}

#[test]
fn test_gbk_roundtrip_ascii() {
    // ASCII 字符的 GBK 编码等于其字节值
    let v = run_vm("from_gbk(to_gbk(\"hello\"))").expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn test_gbk_roundtrip_chinese() {
    let v = run_vm("from_gbk(to_gbk(\"你好\"))").expect("VM 执行失败");
    assert_eq!(as_str(&v), "你好");
}

#[test]
fn test_gbk_chinese_bytes() {
    // "你" 的 GBK 编码是 0xC4E3
    let v = run_vm("to_gbk(\"你\")").expect("VM 执行失败");
    assert_eq!(as_i64_vec(&v), vec![0xC4, 0xE3]);
}

// ══════════════════════════════════════════════════════════════════════
// 第 44 项：Base64 / Hex / URL 编解码
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_base64_encode_basic() {
    // "Hello" → Base64 → "SGVsbG8="
    let v = run_vm("base64_encode(str_to_bytes(\"Hello\"))").expect("VM 执行失败");
    assert_eq!(as_str(&v), "SGVsbG8=");
}

#[test]
fn test_base64_decode_basic() {
    // "SGVsbG8=" → bytes → "Hello"
    let v = run_vm("base64_decode(\"SGVsbG8=\")").expect("VM 执行失败");
    let inner = unwrap_ok(&v);
    let bytes = as_i64_vec(&inner);
    let s: String = bytes.iter().map(|&b| b as u8 as char).collect();
    assert_eq!(s, "Hello");
}

#[test]
fn test_base64_roundtrip() {
    let v = run_vm("base64_decode(base64_encode(str_to_bytes(\"test data\")))").expect("VM 执行失败");
    let inner = unwrap_ok(&v);
    let bytes = as_i64_vec(&inner);
    let s: String = bytes.iter().map(|&b| b as u8 as char).collect();
    assert_eq!(s, "test data");
}

#[test]
fn test_base64_decode_invalid() {
    // 无效 Base64 应返回 Result::Err
    let v = run_vm("base64_decode(\"!!!invalid!!!\")").expect("VM 执行失败");
    assert_is_err(&v);
}

#[test]
fn test_hex_encode_basic() {
    // [255, 0, 128] → "ff0080"
    let v = run_vm("hex_encode([255, 0, 128])").expect("VM 执行失败");
    assert_eq!(as_str(&v), "ff0080");
}

#[test]
fn test_hex_decode_basic() {
    let v = run_vm("hex_decode(\"ff0080\")").expect("VM 执行失败");
    let inner = unwrap_ok(&v);
    assert_eq!(as_i64_vec(&inner), vec![255, 0, 128]);
}

#[test]
fn test_hex_roundtrip() {
    let v = run_vm("hex_decode(hex_encode([255, 0, 128, 42]))").expect("VM 执行失败");
    let inner = unwrap_ok(&v);
    assert_eq!(as_i64_vec(&inner), vec![255, 0, 128, 42]);
}

#[test]
fn test_hex_decode_odd_length_error() {
    // 奇数长度应报错
    let v = run_vm("hex_decode(\"abc\")").expect("VM 执行失败");
    assert_is_err(&v);
}

#[test]
fn test_url_encode_basic() {
    // "hello world" → "hello%20world"
    let v = run_vm("url_encode(\"hello world\")").expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello%20world");
}

#[test]
fn test_url_decode_basic() {
    let v = run_vm("url_decode(\"hello%20world\")").expect("VM 执行失败");
    let inner = unwrap_ok(&v);
    assert_eq!(as_str(&inner), "hello world");
}

#[test]
fn test_url_roundtrip() {
    let v = run_vm("url_decode(url_encode(\"hello world 123\"))").expect("VM 执行失败");
    let inner = unwrap_ok(&v);
    assert_eq!(as_str(&inner), "hello world 123");
}

// ══════════════════════════════════════════════════════════════════════
// 编码转换新 API 别名（std/string/encoding.th）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_to_utf8_alias() {
    let v = run_vm("to_utf8(\"AB\")").expect("VM 执行失败");
    assert_eq!(as_i64_vec(&v), vec![65, 66]);
}

#[test]
fn test_to_utf16_alias() {
    let v = run_vm("to_utf16(\"hello\")").expect("VM 执行失败");
    assert_eq!(as_i64_vec(&v), vec![104, 101, 108, 108, 111]);
}

#[test]
fn test_from_utf16_alias() {
    let v = run_vm("from_utf16(str_to_utf16(\"hello\"))").expect("VM 执行失败");
    assert_eq!(as_str(&v), "hello");
}

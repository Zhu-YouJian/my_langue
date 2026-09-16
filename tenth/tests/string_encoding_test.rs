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

// AUDIT-11.4.27：f-string 表达式插值 + 格式说明符（修复后守护）

#[test]
fn test_fstring_expr_interp_vm() {
    // 任意表达式插值：{a+b}
    let src = "fn main() { let a = 1; let b = 2; f\"{a+b}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "3");
}

#[test]
fn test_fstring_expr_interp_interp() {
    let src = "fn main() { let a = 1; let b = 2; f\"{a+b}\" }";
    let v = run_interp(src).expect("解释器执行失败");
    assert_eq!(as_str(&v), "3");
}

#[test]
fn test_fstring_float_literal_interp() {
    // 手册 §2.3：f"value = {x}, pi = {3.14}"
    let src = "fn main() { let x = 5; f\"value = {x}, pi = {3.14}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "value = 5, pi = 3.14");
}

#[test]
fn test_fstring_format_spec_vm() {
    // 手册 §12.15.1 末行：f"pi ≈ {3.14159:.2f}" → "pi ≈ 3.14"
    let src = "fn main() { f\"pi ≈ {3.14159:.2f}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "pi ≈ 3.14");
}

#[test]
fn test_fstring_format_spec_interp() {
    let src = "fn main() { f\"pi ≈ {3.14159:.2f}\" }";
    let v = run_interp(src).expect("解释器执行失败");
    assert_eq!(as_str(&v), "pi ≈ 3.14");
}

#[test]
fn test_fstring_format_spec_width_vm() {
    // 变量 + 宽度说明符：{n:>5}
    let src = "fn main() { let n = 42; f\"[{n:>5}]\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "[   42]");
}

#[test]
fn test_fstring_escaped_braces_vm() {
    // {{ }} → 字面 {}（f-string 花括号转义）
    let src = "fn main() { let name = \"X\"; f\"{{name}}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "{name}");
}

#[test]
fn test_fstring_escaped_braces_interp() {
    let src = "fn main() { let name = \"X\"; f\"{{name}}\" }";
    let v = run_interp(src).expect("解释器执行失败");
    assert_eq!(as_str(&v), "{name}");
}

#[test]
fn test_interp_string_nonstring_vm() {
    // 普通字符串插值含非字符串值（数字）：VM 路径此前走 format(x) 报错，
    // 2026-08-03 改 to_string(x) 后与解释器一致（手册 §2.3 示例的组成部分）
    let src = "fn main() { let name = \"Alice\"; let age = 30; \"name={name}, age={age}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "name=Alice, age=30");
}

#[test]
fn test_interp_string_nonstring_interp() {
    let src = "fn main() { let name = \"Alice\"; let age = 30; \"name={name}, age={age}\" }";
    let v = run_interp(src).expect("解释器执行失败");
    assert_eq!(as_str(&v), "name=Alice, age=30");
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

// AUDIT-11.4.26：format 进制说明符 {:x}/{:X}/{:o}/{:b}/{:d}（修复后守护）

#[test]
fn test_format_hex_lower_vm() {
    // 手册 §12.15.1：format("0x{:x}", 255) → "0xff"
    let v = run_vm("format(\"0x{:x}\", 255)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "0xff");
}

#[test]
fn test_format_hex_lower_interp() {
    let v = run_interp("format(\"0x{:x}\", 255)").expect("解释器执行失败");
    assert_eq!(as_str(&v), "0xff");
}

#[test]
fn test_format_hex_upper_vm() {
    let v = run_vm("format(\"{:X}\", 255)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "FF");
}

#[test]
fn test_format_hex_upper_interp() {
    let v = run_interp("format(\"{:X}\", 255)").expect("解释器执行失败");
    assert_eq!(as_str(&v), "FF");
}

#[test]
fn test_format_octal_vm() {
    let v = run_vm("format(\"{:o}\", 8)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "10");
}

#[test]
fn test_format_octal_interp() {
    let v = run_interp("format(\"{:o}\", 8)").expect("解释器执行失败");
    assert_eq!(as_str(&v), "10");
}

#[test]
fn test_format_binary_vm() {
    let v = run_vm("format(\"{:b}\", 5)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "101");
}

#[test]
fn test_format_binary_interp() {
    let v = run_interp("format(\"{:b}\", 5)").expect("解释器执行失败");
    assert_eq!(as_str(&v), "101");
}

#[test]
fn test_format_decimal_spec_vm() {
    // {:d} 十进制（若已有则保持）
    let v = run_vm("format(\"{:d}\", 42)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "42");
}

#[test]
fn test_format_hex_zero_padded_vm() {
    // 宽度 + 补零：{:08x} → 000000ff
    let v = run_vm("format(\"{:08x}\", 255)").expect("VM 执行失败");
    assert_eq!(as_str(&v), "000000ff");
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

// AUDIT-11.4.25：字符串 \u{...} Unicode 转义（修复后守护）

#[test]
fn test_unicode_escape_combining_vm() {
    // 手册 §12.15.2："cafe\u{0301}" = "cafe" + U+0301（5 code points）
    let src = "fn main() { \"cafe\\u{0301}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "cafe\u{0301}");
}

#[test]
fn test_unicode_escape_combining_interp() {
    let src = "fn main() { \"cafe\\u{0301}\" }";
    let v = run_interp(src).expect("解释器执行失败");
    assert_eq!(as_str(&v), "cafe\u{0301}");
}

#[test]
fn test_unicode_escape_ascii_codepoint() {
    // \u{30} → '0'（ASCII 码点）
    let src = "fn main() { \"\\u{30}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "0");
}

#[test]
fn test_unicode_escape_lowercase_hex() {
    // \u{1f600} → 😀（U+1F600，非 BMP，4 字节 UTF-8）
    let src = "fn main() { \"\\u{1f600}\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "\u{1F600}");
}

#[test]
fn test_hex_byte_escape_vm() {
    // \x41 → 'A'（与字节串 \xNN 一致的十六进制字节转义）
    let src = "fn main() { \"\\x41\" }";
    let v = run_vm(src).expect("VM 执行失败");
    assert_eq!(as_str(&v), "A");
}

#[test]
fn test_hex_byte_escape_interp() {
    let src = "fn main() { \"\\x41\" }";
    let v = run_interp(src).expect("解释器执行失败");
    assert_eq!(as_str(&v), "A");
}

#[test]
fn test_unicode_escape_manual_normalization() {
    // 手册 §12.15.2 完整示例：unicode_nfc("cafe\u{0301}") == "café"
    let src = "fn main() { unicode_nfc(\"cafe\\u{0301}\") == \"café\" }";
    let v = run_vm(src).expect("VM 执行失败");
    match v {
        Value::Bool(b) => assert!(b, "NFC 归一后应相等"),
        other => panic!("期望 Bool，实际 {:?}", other),
    }
}

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

// ══════════════════════════════════════════════════════════════════════
// AUDIT-11.4.63：编码族**双路径**覆盖（VM + 解释器）
//
// 缺口长期隐身的原因：上方第 43/44 项与别名断言**只跑 VM**（无解释器覆盖），
// 而解释器侧缺 4 个公开名（`to_utf8` / `to_utf16` / `utf16_to_str` /
// `bytes_to_str`；只有 `_` 前缀私有名且**无别名映射**）⇒ 套件全绿之下
// `TENTH_NO_VM=1` 用公开名会响亮报 `undefined function`。
// 以下每条断言都同时跑两条路径，并逐值结构比对（先 peel 解释器侧的
// `Value::Shared` 包裹再比较）。
// ══════════════════════════════════════════════════════════════════════

/// peel 解释器侧容器元素包装（Value::Shared），便于双路径结构比较。
fn peel_shared(v: &Value) -> Value {
    match v {
        Value::Shared(rc) => peel_shared(&rc.borrow().clone()),
        Value::Ref(rc) => peel_shared(&rc.borrow().clone()),
        other => other.clone(),
    }
}

/// 跨路径结构等价（Int 只比数值，不比较 dtype 标签）。
fn values_equal(a: &Value, b: &Value) -> bool {
    match (peel_shared(a), peel_shared(b)) {
        (Value::Int(x, _), Value::Int(y, _)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Float32(x), Value::Float32(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        (Value::Vec(x), Value::Vec(y)) => {
            let xs = x.borrow();
            let ys = y.borrow();
            xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(p, q)| values_equal(p, q))
        }
        (
            Value::Enum { variant: va, fields: fa, .. },
            Value::Enum { variant: vb, fields: fb, .. },
        ) => {
            if va != vb {
                return false;
            }
            let xs = fa.borrow();
            let ys = fb.borrow();
            xs.len() == ys.len()
                && xs
                    .iter()
                    .zip(ys.iter())
                    .all(|((ka, p), (kb, q))| ka == kb && values_equal(p, q))
        }
        (x, y) => format!("{x:?}") == format!("{y:?}"),
    }
}

/// 断言同一源码在 VM 与解释器两条路径都成功且结果结构相等，返回 VM 侧结果。
fn assert_both_paths(src: &str) -> Value {
    let vm = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {e}\n源码: {src}"));
    let ip = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {e}\n源码: {src}"));
    assert!(
        values_equal(&vm, &ip),
        "VM 与解释器结果不一致\n源码: {src}\nVM: {vm:?}\n解释器: {ip:?}"
    );
    vm
}

#[test]
fn test_dual_path_str_to_utf16() {
    let v = assert_both_paths("str_to_utf16(\"hello\")");
    assert_eq!(as_i64_vec(&v), vec![104, 101, 108, 108, 111]);
}

#[test]
fn test_dual_path_utf16_to_str_public_name() {
    // `utf16_to_str` 是公开名：解释器侧此前只有 `_utf16_to_str` ⇒ TENTH_NO_VM 下
    // `undefined function 'utf16_to_str'`。
    let v = assert_both_paths("utf16_to_str(str_to_utf16(\"hello\"))");
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn test_dual_path_str_to_bytes() {
    let v = assert_both_paths("str_to_bytes(\"AB\")");
    assert_eq!(as_i64_vec(&v), vec![65, 66]);
}

#[test]
fn test_dual_path_bytes_to_str_public_name() {
    // `bytes_to_str` 公开名：解释器侧此前只有 `_bytes_to_str`。
    let v = assert_both_paths("bytes_to_str(str_to_bytes(\"hello\"))");
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn test_dual_path_to_utf8_public_alias() {
    // `to_utf8` 公开别名：解释器侧此前只有 `_to_utf8`。
    let v = assert_both_paths("to_utf8(\"AB\")");
    assert_eq!(as_i64_vec(&v), vec![65, 66]);
}

#[test]
fn test_dual_path_to_utf16_public_alias() {
    // `to_utf16` 公开别名：解释器侧此前只有 `_to_utf16`。
    let v = assert_both_paths("to_utf16(\"hello\")");
    assert_eq!(as_i64_vec(&v), vec![104, 101, 108, 108, 111]);
}

#[test]
fn test_dual_path_from_utf16_public_alias() {
    let v = assert_both_paths("from_utf16(str_to_utf16(\"hello\"))");
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn test_dual_path_utf16_chinese() {
    let v = assert_both_paths("str_to_utf16(\"你\")");
    assert_eq!(as_i64_vec(&v), vec![0x4F60]);
}

#[test]
fn test_dual_path_gbk_roundtrip() {
    let v = assert_both_paths("from_gbk(to_gbk(\"你好\"))");
    assert_eq!(as_str(&v), "你好");
    let v = assert_both_paths("to_gbk(\"你\")");
    assert_eq!(as_i64_vec(&v), vec![0xC4, 0xE3]);
}

#[test]
fn test_dual_path_base64_hex_url() {
    let v = assert_both_paths("base64_encode(str_to_bytes(\"Hello\"))");
    assert_eq!(as_str(&v), "SGVsbG8=");
    let v = assert_both_paths("hex_encode([255, 0, 128])");
    assert_eq!(as_str(&v), "ff0080");
    let v = assert_both_paths("url_encode(\"hello world\")");
    assert_eq!(as_str(&v), "hello%20world");
}

/// AUDIT-11.4.61 同族：**容器元素经 `Vec.push` 包装（Value::Shared）**后再交给
/// 编码 native。解释器侧读取元素时未 peel ⇒ 全部元素落 `_ => 0`（`bytes_to_str`
/// 静默返回 `"\0\0"`、`utf16_to_str` 静默返回 `"\0\0"`），与 VM 分叉且**无报错**。
#[test]
fn test_dual_path_bytes_to_str_from_pushed_vec() {
    let src = r#"
let v = Vec::new();
v.push(65);
v.push(66);
bytes_to_str(v)
"#;
    let v = assert_both_paths(src);
    assert_eq!(as_str(&v), "AB", "Vec.push 包装元素后 bytes_to_str 应得 \"AB\"");
}

#[test]
fn test_dual_path_utf16_to_str_from_pushed_vec() {
    let src = r#"
let v = Vec::new();
v.push(104);
v.push(105);
utf16_to_str(v)
"#;
    let v = assert_both_paths(src);
    assert_eq!(as_str(&v), "hi", "Vec.push 包装元素后 utf16_to_str 应得 \"hi\"");
}

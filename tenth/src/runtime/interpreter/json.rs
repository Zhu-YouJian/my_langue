//! JSON 编解码（安全版本，带 H-6 修复）。
//!
//! 从 `main.rs` 第 338-541 行迁移而来（注意：原 `interpreter.rs` 第 71-191 行
//! 是**旧版本**，缺 H-6 深度闸门和转义状态机修复）。
//!
//! 修复点（H-6）：
//! 1. `json_decode_string_depth` 加深度闸门（256），防止 `[[[...` 递归爆栈；
//! 2. `simple_json_split` 修复转义状态机，正确识别 `"a\"b"` 中的 `\"`；
//! 3. `json_unescape` 替代链式 `replace()`，正确处理反斜杠后接引号的情况。

use crate::runtime::value::Value;
use crate::hir::types::BaseType;
use std::rc::Rc;
use std::cell::RefCell;

pub fn json_encode_value(val: &Value) -> String {
    match val {
        Value::Int(n, _) => format!("{}", n),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")),
        Value::Unit => "null".into(),
        Value::Vec(v) => {
            let items: Vec<String> = v.borrow().iter().map(|v| json_encode_value(v)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Array(a) => {
            let items: Vec<String> = a.borrow().iter().map(|v| json_encode_value(v)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Map(map) => {
            let entries: Vec<String> = map.borrow().iter().map(|(k, v)| {
                format!("\"{}\": {}", k, json_encode_value(v))
            }).collect();
            format!("{{{}}}", entries.join(", "))
        }
        _ => "null".into(),
    }
}

pub fn json_encode_value_pretty(val: &Value, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    let inner_prefix = "  ".repeat(indent + 1);
    match val {
        Value::Int(n, _) => format!("{}", n),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")),
        Value::Unit => "null".into(),
        Value::Vec(v) => {
            if v.borrow().is_empty() { return "[]".into(); }
            let items: Vec<String> = v.borrow().iter().map(|v| format!("{}{}", inner_prefix, json_encode_value_pretty(v, indent + 1))).collect();
            format!("[\n{}\n{}]", items.join(",\n"), prefix)
        }
        Value::Array(a) => {
            if a.borrow().is_empty() { return "[]".into(); }
            let items: Vec<String> = a.borrow().iter().map(|v| format!("{}{}", inner_prefix, json_encode_value_pretty(v, indent + 1))).collect();
            format!("[\n{}\n{}]", items.join(",\n"), prefix)
        }
        Value::Map(map) => {
            if map.borrow().is_empty() { return "{}".into(); }
            let entries: Vec<String> = map.borrow().iter().map(|(k, v)| {
                format!("{}\"{}\": {}", inner_prefix, k, json_encode_value_pretty(v, indent + 1))
            }).collect();
            format!("{{\n{}\n{}}}", entries.join(",\n"), prefix)
        }
        _ => "null".into(),
    }
}

/// JSON 字符串解析最大嵌套深度。超过即返回 `Value::Unit`，避免恶意构造的
/// `[[[[...]]]` 递归爆栈（即便 build.rs 把栈扩到 64 MiB，几千层嵌套仍可爆栈）。
const JSON_MAX_DEPTH: usize = 256;

pub fn json_decode_string(s: &str) -> Value {
    json_decode_string_depth(s, 0)
}

fn json_decode_string_depth(s: &str, depth: usize) -> Value {
    // 安全：深度闸门。攻击者构造 `[[[...` 千层嵌套即可触发栈溢出 DoS。
    if depth > JSON_MAX_DEPTH {
        return Value::Unit;
    }
    let s = s.trim();
    if s == "null" { return Value::Unit; }
    if s == "true" { return Value::Bool(true); }
    if s == "false" { return Value::Bool(false); }
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        let inner = &s[1..s.len()-1];
        return Value::String(json_unescape(inner));
    }
    if let Ok(n) = s.parse::<i64>() { return Value::Int(n, BaseType::I32); }
    if let Ok(f) = s.parse::<f64>() { return Value::Float(f); }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len()-1];
        if inner.trim().is_empty() { return Value::Vec(Rc::new(RefCell::new(Vec::new()))); }
        let items: Vec<Value> = simple_json_split(inner, ',')
            .iter()
            .map(|s| json_decode_string_depth(s, depth + 1))
            .collect();
        return Value::Vec(Rc::new(RefCell::new(items)));
    }
    if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len()-1];
        if inner.trim().is_empty() {
            return Value::Map(Rc::new(RefCell::new(std::collections::HashMap::new())));
        }
        let mut map = std::collections::HashMap::new();
        let entries = simple_json_split(inner, ',');
        for entry in &entries {
            let parts = simple_json_split(entry, ':');
            if parts.len() >= 2 {
                let key = json_decode_string_depth(parts[0].trim(), depth + 1);
                if let Value::String(k) = key {
                    let val = json_decode_string_depth(parts[1].trim(), depth + 1);
                    map.insert(k, val);
                }
            }
        }
        return Value::Map(Rc::new(RefCell::new(map)));
    }
    Value::Unit
}

/// 解析 JSON 字符串字面量中的转义序列。支持 `\"`、`\\`、`\n`、`\t`、`\r`、`\/`、`\b`、`\f`。
/// 不支持的转义（如 `\uXXXX`）按字面保留，便于上层识别。
/// 此函数替代了原 `replace()` 链式调用——后者无法正确处理 `"a\\\"b"`（反斜杠后接引号）的情况。
fn json_unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('b') => out.push('\u{0008}'),
            Some('f') => out.push('\u{000C}'),
            Some('u') => {
                // \uXXXX — 取 4 位十六进制。失败则保留字面 `\u`。
                let mut hex = String::with_capacity(4);
                for _ in 0..4 {
                    if let Some(h) = chars.next() {
                        hex.push(h);
                    } else {
                        break;
                    }
                }
                if let Ok(code) = u32::from_str_radix(&hex, 16) {
                    if let Some(ch) = char::from_u32(code) {
                        out.push(ch);
                        continue;
                    }
                }
                out.push_str("\\u");
                out.push_str(&hex);
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn simple_json_split(s: &str, delimiter: char) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut prev_was_backslash = false;
    for c in s.chars() {
        if in_string {
            current.push(c);
            // 安全：修复转义状态机。原实现不识别 `\"`，导致 `"a\"b"` 中的 `\"`
            // 被误认为字符串结束，后续 `,` 被当作分隔符，解析结果错误。
            if prev_was_backslash {
                // 当前字符被反斜杠转义，不改变 in_string 状态
                prev_was_backslash = false;
            } else if c == '\\' {
                prev_was_backslash = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            prev_was_backslash = false;
            current.push(c);
            continue;
        }
        match c {
            '[' | '{' => { depth += 1; current.push(c); }
            ']' | '}' => { depth -= 1; current.push(c); }
            d if d == delimiter && depth == 0 => {
                result.push(current.trim().to_string());
                current = String::new();
            }
            _ => { current.push(c); }
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() { result.push(trimmed); }
    result
}

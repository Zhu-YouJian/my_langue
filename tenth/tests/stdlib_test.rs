use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

#[test]
fn test_string_len() {
    let src = "\"hello\".len()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(5)) => {}
        v => panic!("expected Some(Int(5)), got {:?}", v),
    }
}

#[test]
fn test_string_concat() {
    let src = "\"hello\" + \" world\"";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "hello world" => {}
        v => panic!("expected String(\"hello world\"), got {:?}", v),
    }
}

#[test]
fn test_vec_new_and_len() {
    let src = "Vec::new().len()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(0)) => {}
        v => panic!("expected Some(Int(0)), got {:?}", v),
    }
}

#[test]
fn test_hashmap_new_and_len() {
    let src = "HashMap::new().len()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(0)) => {}
        v => panic!("expected Some(Int(0)), got {:?}", v),
    }
}

#[test]
fn test_hashmap_get() {
    let src = "HashMap::new().get(\"missing\")";
    let result = run_code(src).unwrap();
    // get returns None → no matching variable → should return None/Unit
    assert!(result.is_none(), "expected None for missing key, got {:?}", result);
}

#[test]
fn test_read_file() {
    // Test that read_file builtin is registered and returns error for missing file
    // (actual file read tested via integration)
    let src = "read_file(\"nonexistent_file.th\")";
    let result = run_code(src);
    assert!(result.is_err(), "should fail for missing file");
}

#[test]
fn test_option_some() {
    let src = "Option::Some(42)";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Enum { enum_name, variant, fields }) => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "Some");
            assert!(fields.borrow().iter().any(|(n, _)| n == "_0"));
        }
        v => panic!("expected Option::Some, got {:?}", v),
    }
}

#[test]
fn test_option_none() {
    let src = "Option::None";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Enum { enum_name, variant, .. }) => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "None");
        }
        v => panic!("expected Option::None, got {:?}", v),
    }
}

// ── String method tests ──

#[test]
fn test_string_contains() {
    let src = r#""hello world".contains("world")"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Bool(true)) => {}
        v => panic!("expected Bool(true), got {:?}", v),
    }
}

#[test]
fn test_string_find() {
    let src = r#""hello world".find("world")"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(6)) => {}
        v => panic!("expected Int(6), got {:?}", v),
    }
}

#[test]
fn test_string_find_not_found() {
    let src = r#""hello".find("xyz")"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(-1)) => {}
        v => panic!("expected Int(-1), got {:?}", v),
    }
}

#[test]
fn test_string_starts_ends_with() {
    let src = r#""hello".starts_with("he") && "hello".ends_with("llo")"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Bool(true)) => {}
        v => panic!("expected Bool(true), got {:?}", v),
    }
}

#[test]
fn test_string_parse_int() {
    let src = r#""42".parse_int()"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42)) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

#[test]
fn test_string_parse_float() {
    let src = r#""3.14".parse_float()"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 3.14).abs() < 0.01),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_format_builtin() {
    let src = r#"format("{} + {} = {}", 1, 2, 3)"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "1 + 2 = 3" => {}
        v => panic!("expected String(\"1 + 2 = 3\"), got {:?}", v),
    }
}

// ── Vec method tests ──

#[test]
fn test_vec_contains() {
    let src = r#"
        let v = Vec::new();
        v.push(10);
        v.push(20);
        v.contains(10)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Bool(true)) => {}
        v => panic!("expected Bool(true), got {:?}", v),
    }
}

#[test]
fn test_vec_index_of() {
    let src = r#"
        let v = Vec::new();
        v.push("a");
        v.push("b");
        v.push("c");
        v.index_of("b")
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1)) => {}
        v => panic!("expected Int(1), got {:?}", v),
    }
}

#[test]
fn test_vec_remove() {
    let src = r#"
        let v = Vec::new();
        v.push(10);
        v.push(20);
        v.push(30);
        let removed = v.remove(1);
        v.len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(2)) => {}
        v => panic!("expected Int(2), got {:?}", v),
    }
}

#[test]
fn test_vec_join() {
    let src = r#"
        let v = Vec::new();
        v.push("hello");
        v.push("world");
        v.join(" ")
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "hello world" => {}
        v => panic!("expected String(\"hello world\"), got {:?}", v),
    }
}

#[test]
fn test_vec_reverse() {
    let src = r#"
        let v = Vec::new();
        v.push(1);
        v.push(2);
        v.push(3);
        let r = v.reverse();
        r.get(0)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Shared(rc)) => {
            let val = rc.borrow();
            match &*val {
                Value::Int(3) => {}
                v => panic!("expected Int(3), got {:?}", v),
            }
        }
        Some(Value::Int(3)) => {}
        v => panic!("expected 3, got {:?}", v),
    }
}

#[test]
fn test_vec_slice() {
    let src = r#"
        let v = Vec::new();
        v.push(10);
        v.push(20);
        v.push(30);
        v.push(40);
        let s = v.slice(1, 3);
        s.len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(2)) => {}
        v => panic!("expected Int(2), got {:?}", v),
    }
}

// ── HashMap method tests ──

#[test]
fn test_hashmap_contains_key() {
    let src = r#"
        let m = HashMap::new();
        m.insert("name", "Alice");
        m.contains_key("name")
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Bool(true)) => {}
        v => panic!("expected Bool(true), got {:?}", v),
    }
}

#[test]
fn test_hashmap_remove() {
    let src = r#"
        let m = HashMap::new();
        m.insert("key", "value");
        m.remove("key");
        m.len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(0)) => {}
        v => panic!("expected Int(0), got {:?}", v),
    }
}

#[test]
fn test_hashmap_keys() {
    let src = r#"
        let m = HashMap::new();
        m.insert("a", 1);
        m.insert("b", 2);
        m.keys().len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(2)) => {}
        v => panic!("expected Int(2), got {:?}", v),
    }
}

// ── Extended String method tests ──

#[test]
fn test_string_repeat() {
    let src = r#""ha".repeat(3)"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "hahaha" => {}
        v => panic!("expected String(\"hahaha\"), got {:?}", v),
    }
}

#[test]
fn test_string_chars() {
    let src = r#""abc".chars().len()"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("expected Int(3), got {:?}", v),
    }
}

#[test]
fn test_string_bytes() {
    let src = r#""ABC".bytes().len()"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("expected Int(3), got {:?}", v),
    }
}

#[test]
fn test_string_trim_start_end() {
    let src = r#""  hi  ".trim_start().trim_end()"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "hi" => {}
        v => panic!("expected String(\"hi\"), got {:?}", v),
    }
}

#[test]
fn test_string_strip_prefix() {
    let src = r#""hello.txt".strip_prefix("hello.")"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "txt" => {}
        v => panic!("expected String(\"txt\"), got {:?}", v),
    }
}

#[test]
fn test_string_strip_suffix() {
    let src = r#""hello.txt".strip_suffix(".txt")"#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "hello" => {}
        v => panic!("expected String(\"hello\"), got {:?}", v),
    }
}

// ── Extended Vec method tests ──

#[test]
fn test_vec_extend() {
    let src = r#"
        let a = Vec::new();
        a.push(1);
        a.push(2);
        let b = Vec::new();
        b.push(3);
        b.push(4);
        a.extend(b);
        a.len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(4)) => {}
        v => panic!("expected Int(4), got {:?}", v),
    }
}

#[test]
fn test_vec_sort() {
    let src = r#"
        let v = Vec::new();
        v.push(3);
        v.push(1);
        v.push(2);
        v.sort();
        v.get(0)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Shared(rc)) => {
            match &*rc.borrow() {
                Value::Int(1) => {}
                v => panic!("expected Int(1), got {:?}", v),
            }
        }
        Some(Value::Int(1)) => {}
        v => panic!("expected 1, got {:?}", v),
    }
}

#[test]
fn test_vec_dedup() {
    let src = r#"
        let v = Vec::new();
        v.push(1);
        v.push(1);
        v.push(2);
        v.push(2);
        v.dedup();
        v.len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(2)) => {}
        v => panic!("expected Int(2), got {:?}", v),
    }
}

#[test]
fn test_vec_first_last() {
    let src = r#"
        let v = Vec::new();
        v.push(10);
        v.push(20);
        v.push(30);
        let f = v.first();
        let l = v.last();
        format("{}-{}", f, l)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s.contains("10") && s.contains("30") => {}
        v => panic!("expected string with 10 and 30, got {:?}", v),
    }
}

#[test]
fn test_vec_flatten() {
    let src = r#"
        let v = Vec::new();
        let inner = Vec::new();
        inner.push(1);
        inner.push(2);
        v.push(inner);
        let inner2 = Vec::new();
        inner2.push(3);
        v.push(inner2);
        let flat = v.flatten();
        flat.len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("expected Int(3), got {:?}", v),
    }
}

#[test]
fn test_vec_chunks() {
    let src = r#"
        let v = Vec::new();
        v.push(1);
        v.push(2);
        v.push(3);
        v.push(4);
        v.push(5);
        let c = v.chunks(2);
        c.len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("expected Int(3), got {:?}", v),
    }
}

// ── Extended HashMap method tests ──

#[test]
fn test_hashmap_entries() {
    let src = r#"
        let m = HashMap::new();
        m.insert("x", 1);
        m.insert("y", 2);
        m.entries().len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(2)) => {}
        v => panic!("expected Int(2), got {:?}", v),
    }
}

#[test]
fn test_hashmap_merge() {
    let src = r#"
        let a = HashMap::new();
        a.insert("x", 1);
        let b = HashMap::new();
        b.insert("y", 2);
        a.merge(b);
        a.len()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(2)) => {}
        v => panic!("expected Int(2), got {:?}", v),
    }
}

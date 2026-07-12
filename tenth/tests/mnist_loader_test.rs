//! Test MNIST data loading: read_bytes native function and IDX format parsing.

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use std::io::Write;

/// Run source code through the interpreter with read_bytes support.
fn run_interp(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interp = Interpreter::new(&hir);
    match interp.execute_program(&hir) {
        Ok(Some(val)) => Ok(val),
        Ok(None) => Ok(Value::Unit),
        Err(e) => Err(e.to_string()),
    }
}

#[test]
fn test_read_bytes_native() {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join("tenth_test_read_bytes.bin");
    {
        let mut f = std::fs::File::create(&tmp_path).unwrap();
        f.write_all(&[0x00, 0x00, 0x08, 0x03, 0x00, 0x00, 0x00, 0x02]).unwrap();
    }

    let path_str = tmp_path.to_string_lossy().to_string();
    let src = format!(r#"
        let bytes = read_bytes("{}");
        bytes.len()
    "#, path_str.replace("\\", "\\\\"));

    let result = run_interp(&src).unwrap();
    match result {
        Value::Int(n, _) => assert_eq!(n, 8, "expected 8 bytes, got {}", n),
        _ => panic!("expected int, got {:?}", result),
    }

    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_read_bytes_big_endian_i32() {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join("tenth_test_idx.bin");
    {
        let mut f = std::fs::File::create(&tmp_path).unwrap();
        f.write_all(&[
            0x00, 0x00, 0x08, 0x03,  // magic: 2051
            0x00, 0x00, 0x00, 0x05,  // num_images: 5
            0x00, 0x00, 0x00, 0x02,  // rows: 2
            0x00, 0x00, 0x00, 0x03,  // cols: 3
        ]).unwrap();
    }

    let path_str = tmp_path.to_string_lossy().to_string();
    // Vec.get() returns the value directly; no .as_int() needed
    let src = format!(r#"
        let bytes = read_bytes("{}");
        let b0 = bytes.get(0);
        let b1 = bytes.get(1);
        let b2 = bytes.get(2);
        let b3 = bytes.get(3);
        b0 * 16777216 + b1 * 65536 + b2 * 256 + b3
    "#, path_str.replace("\\", "\\\\"));

    let result = run_interp(&src).unwrap();
    match result {
        Value::Int(n, _) => assert_eq!(n, 2051, "expected magic 2051, got {}", n),
        _ => panic!("expected int, got {:?}", result),
    }

    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_idx_parse_images_metadata() {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join("tenth_test_images.idx3");
    {
        let mut f = std::fs::File::create(&tmp_path).unwrap();
        f.write_all(&[
            0x00, 0x00, 0x08, 0x03,  // magic: 2051
            0x00, 0x00, 0x00, 0x02,  // num_images: 2
            0x00, 0x00, 0x00, 0x02,  // rows: 2
            0x00, 0x00, 0x00, 0x03,  // cols: 3
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
        ]).unwrap();
    }

    let path_str = tmp_path.to_string_lossy().to_string();
    let src = format!(r#"
        let bytes = read_bytes("{}");
        let magic = bytes.get(0) * 16777216 + bytes.get(1) * 65536 + bytes.get(2) * 256 + bytes.get(3);
        let num_images = bytes.get(4) * 16777216 + bytes.get(5) * 65536 + bytes.get(6) * 256 + bytes.get(7);
        let rows = bytes.get(8) * 16777216 + bytes.get(9) * 65536 + bytes.get(10) * 256 + bytes.get(11);
        magic + num_images + rows
    "#, path_str.replace("\\", "\\\\"));

    let result = run_interp(&src).unwrap();
    match result {
        Value::Int(n, _) => assert_eq!(n, 2051 + 2 + 2, "expected {}, got {}", 2051 + 2 + 2, n),
        _ => panic!("expected int, got {:?}", result),
    }

    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_idx_parse_pixel_data() {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join("tenth_test_pixels.idx3");
    {
        let mut f = std::fs::File::create(&tmp_path).unwrap();
        f.write_all(&[
            0x00, 0x00, 0x08, 0x03,  // magic: 2051
            0x00, 0x00, 0x00, 0x01,  // num_images: 1
            0x00, 0x00, 0x00, 0x01,  // rows: 1
            0x00, 0x00, 0x00, 0x03,  // cols: 3
            10, 20, 30,
        ]).unwrap();
    }

    let path_str = tmp_path.to_string_lossy().to_string();
    let src = format!(r#"
        let bytes = read_bytes("{}");
        let p0 = bytes.get(16);
        let p1 = bytes.get(17);
        let p2 = bytes.get(18);
        p0 + p1 + p2
    "#, path_str.replace("\\", "\\\\"));

    let result = run_interp(&src).unwrap();
    match result {
        Value::Int(n, _) => assert_eq!(n, 60, "expected pixel sum 60, got {}", n),
        _ => panic!("expected int, got {:?}", result),
    }

    let _ = std::fs::remove_file(&tmp_path);
}

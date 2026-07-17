//! Duration 类型集成测试（基本功核查第 49 项，路径 B：复用 struct 机制）。
//!
//! 端到端验证：
//! - struct Duration 字段访问（nanos）
//! - 构造：from_secs / from_millis / from_micros / from_nanos
//! - 访问：as_secs / as_millis / as_micros / as_nanos
//! - 算术：add / sub / mul_scalar / div_scalar
//! - 比较：eq / lt / le / gt / ge / ne
//! - 格式化：to_string（含负数、复合单位）
//! - 零值：zero / is_zero
//!
//! 参考模式：tests/date_test.rs 的内联函数 + run_code 模式
//! （运行时不支持 `use` 加载 .th 模块，故在此内联 std/duration.th 中的关键函数）。

use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

/// Run source through lexer → parser → HIR → interpreter.
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

/// 内联 std/duration.th 中的 Duration struct 定义和所有函数，供测试调用。
/// 与 tenth/std/duration.th 中的实现保持同步。
const DURATION_HELPERS: &str = r#"
    struct Duration { nanos: i64 }

    // 构造函数用 date_i64_add_days(0, N) 取得 I64 标记的常量，
    // 绕过运行时字面量 dtype 硬编码为 I32 的 pre-existing bug
    // （见 std/duration.th 文件头注释）。
    fn duration_from_secs(secs: i64) -> Duration {
        Duration { nanos: date_i64_add_days(0, 1000000000) * secs }
    }

    fn duration_from_millis(ms: i64) -> Duration {
        Duration { nanos: date_i64_add_days(0, 1000000) * ms }
    }

    fn duration_from_micros(us: i64) -> Duration {
        Duration { nanos: date_i64_add_days(0, 1000) * us }
    }

    fn duration_from_nanos(ns: i64) -> Duration {
        Duration { nanos: ns }
    }

    fn duration_as_secs(d: Duration) -> i64 {
        d.nanos / 1000000000i64
    }

    fn duration_as_millis(d: Duration) -> i64 {
        d.nanos / 1000000i64
    }

    fn duration_as_micros(d: Duration) -> i64 {
        d.nanos / 1000i64
    }

    fn duration_as_nanos(d: Duration) -> i64 {
        d.nanos
    }

    fn duration_add(d1: Duration, d2: Duration) -> Duration {
        Duration { nanos: d1.nanos + d2.nanos }
    }

    fn duration_sub(d1: Duration, d2: Duration) -> Duration {
        Duration { nanos: d1.nanos - d2.nanos }
    }

    fn duration_mul_scalar(d: Duration, s: i64) -> Duration {
        Duration { nanos: d.nanos * s }
    }

    fn duration_div_scalar(d: Duration, s: i64) -> Duration {
        Duration { nanos: d.nanos / s }
    }

    fn duration_eq(d1: Duration, d2: Duration) -> bool {
        d1.nanos == d2.nanos
    }

    fn duration_lt(d1: Duration, d2: Duration) -> bool {
        d1.nanos < d2.nanos
    }

    fn duration_le(d1: Duration, d2: Duration) -> bool {
        d1.nanos <= d2.nanos
    }

    fn duration_gt(d1: Duration, d2: Duration) -> bool {
        d1.nanos > d2.nanos
    }

    fn duration_ge(d1: Duration, d2: Duration) -> bool {
        d1.nanos >= d2.nanos
    }

    fn duration_ne(d1: Duration, d2: Duration) -> bool {
        d1.nanos != d2.nanos
    }

    fn duration_to_string(d: Duration) -> String {
        let abs_nanos = if d.nanos < 0 { -d.nanos } else { d.nanos };
        let sign = if d.nanos < 0 { "-" } else { "" };

        if abs_nanos < 1000i64 {
            return sign + to_string(abs_nanos) + "ns";
        } else if abs_nanos < 1000000i64 {
            let us = abs_nanos / 1000i64;
            let rem = abs_nanos % 1000i64;
            if rem == 0 {
                return sign + to_string(us) + "us";
            } else {
                return sign + to_string(abs_nanos) + "ns";
            }
        } else if abs_nanos < 1000000000i64 {
            let ms = abs_nanos / 1000000i64;
            let rem = abs_nanos % 1000000i64;
            if rem == 0 {
                return sign + to_string(ms) + "ms";
            } else {
                return sign + to_string(abs_nanos / 1000i64) + "us";
            }
        } else {
            let secs = abs_nanos / 1000000000i64;
            let rem_nanos = abs_nanos % 1000000000i64;
            if secs < 60 {
                if rem_nanos == 0 {
                    return sign + to_string(secs) + "s";
                } else {
                    let ms = rem_nanos / 1000000i64;
                    return sign + to_string(secs) + "." + to_string(ms) + "s";
                }
            } else if secs < 3600 {
                let m = secs / 60;
                let s = secs % 60;
                return sign + to_string(m) + "m" + to_string(s) + "s";
            } else {
                let h = secs / 3600;
                let m = (secs % 3600) / 60;
                return sign + to_string(h) + "h" + to_string(m) + "m";
            }
        }
    }

    fn duration_zero() -> Duration {
        Duration { nanos: 0 }
    }

    fn duration_is_zero(d: Duration) -> bool {
        d.nanos == 0
    }
"#;

/// 取解释器返回值中的 i64。
fn as_i64(v: Option<Value>) -> i64 {
    match v {
        Some(Value::Int(n, _)) => n,
        other => panic!("期望 Some(Int(_))，got {:?}", other),
    }
}

/// 取解释器返回值中的 String。
fn as_string(v: Option<Value>) -> String {
    match v {
        Some(Value::String(s)) => s,
        other => panic!("期望 Some(String(_))，got {:?}", other),
    }
}

/// 取解释器返回值中的 bool。
fn as_bool(v: Option<Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => b,
        other => panic!("期望 Some(Bool(_))，got {:?}", other),
    }
}

// ─── Test 1: duration_from_secs 构造与字段访问 ─────────────────────────────

#[test]
fn test_duration_from_secs() {
    // from_secs(5) → as_secs == 5 / as_nanos == 5_000_000_000
    let src = format!(
        "{}\n    let d = duration_from_secs(5); duration_as_secs(d)",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 5);

    let src = format!(
        "{}\n    let d = duration_from_secs(5); duration_as_nanos(d)",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 5000000000);
}

// ─── Test 2: duration_from_millis ──────────────────────────────────────────

#[test]
fn test_duration_from_millis() {
    // from_millis(500) → as_millis == 500 / as_micros == 500_000
    let src = format!(
        "{}\n    let d = duration_from_millis(500); duration_as_millis(d)",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 500);

    let src = format!(
        "{}\n    let d = duration_from_millis(500); duration_as_micros(d)",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 500000);
}

// ─── Test 3: duration_from_micros ─────────────────────────────────────────

#[test]
fn test_duration_from_micros() {
    // from_micros(100) → as_micros == 100 / as_nanos == 100_000
    let src = format!(
        "{}\n    let d = duration_from_micros(100); duration_as_micros(d)",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 100);

    let src = format!(
        "{}\n    let d = duration_from_micros(100); duration_as_nanos(d)",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 100000);
}

// ─── Test 4: duration_from_nanos ──────────────────────────────────────────

#[test]
fn test_duration_from_nanos() {
    // from_nanos(42) → as_nanos == 42
    let src = format!(
        "{}\n    duration_as_nanos(duration_from_nanos(42))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 42);

    // 直接访问 nanos 字段
    let src = format!(
        "{}\n    duration_from_nanos(42).nanos",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 42);
}

// ─── Test 5: duration_add ─────────────────────────────────────────────────

#[test]
fn test_duration_add() {
    // add(from_secs(3), from_secs(2)) → as_secs == 5
    let src = format!(
        "{}\n    duration_as_secs(duration_add(duration_from_secs(3), duration_from_secs(2)))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 5);

    // 跨单位相加：1s + 500ms → as_millis == 1500
    let src = format!(
        "{}\n    duration_as_millis(duration_add(duration_from_secs(1), duration_from_millis(500)))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 1500);
}

// ─── Test 6: duration_sub ─────────────────────────────────────────────────

#[test]
fn test_duration_sub() {
    // sub(from_secs(5), from_secs(3)) → as_secs == 2
    let src = format!(
        "{}\n    duration_as_secs(duration_sub(duration_from_secs(5), duration_from_secs(3)))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 2);

    // 反向相减：3 - 5 = -2 秒
    let src = format!(
        "{}\n    duration_as_secs(duration_sub(duration_from_secs(3), duration_from_secs(5)))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), -2);
}

// ─── Test 7: duration_mul_scalar ──────────────────────────────────────────

#[test]
fn test_duration_mul_scalar() {
    // mul_scalar(from_secs(3), 2) → as_secs == 6
    let src = format!(
        "{}\n    duration_as_secs(duration_mul_scalar(duration_from_secs(3), 2))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 6);

    // 标量 0：任意 Duration × 0 = 0
    let src = format!(
        "{}\n    duration_as_nanos(duration_mul_scalar(duration_from_secs(100), 0))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 0);

    // 负标量：3s × -1 = -3s
    let src = format!(
        "{}\n    duration_as_secs(duration_mul_scalar(duration_from_secs(3), -1))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), -3);
}

// ─── Test 8: duration_div_scalar ──────────────────────────────────────────

#[test]
fn test_duration_div_scalar() {
    // div_scalar(from_secs(10), 2) → as_secs == 5
    let src = format!(
        "{}\n    duration_as_secs(duration_div_scalar(duration_from_secs(10), 2))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 5);

    // 整数除法：10s / 3 = 3s（余 1s 被截断）
    let src = format!(
        "{}\n    duration_as_secs(duration_div_scalar(duration_from_secs(10), 3))",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 3);
}

// ─── Test 9: 比较 eq / ne ─────────────────────────────────────────────────

#[test]
fn test_duration_eq_ne() {
    // eq: 相同 Duration 相等
    let src = format!(
        "{}\n    duration_eq(duration_from_secs(5), duration_from_secs(5))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);

    // eq: 不同 Duration 不等
    let src = format!(
        "{}\n    duration_eq(duration_from_secs(5), duration_from_secs(3))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), false);

    // ne: 不同 Duration 不等
    let src = format!(
        "{}\n    duration_ne(duration_from_secs(5), duration_from_secs(3))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);

    // 跨单位相等：1s == 1000ms
    let src = format!(
        "{}\n    duration_eq(duration_from_secs(1), duration_from_millis(1000))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);
}

// ─── Test 10: 比较 lt / le / gt / ge ──────────────────────────────────────

#[test]
fn test_duration_lt_le_gt_ge() {
    // lt: 3s < 5s
    let src = format!(
        "{}\n    duration_lt(duration_from_secs(3), duration_from_secs(5))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);

    // lt: 5s < 3s 为假
    let src = format!(
        "{}\n    duration_lt(duration_from_secs(5), duration_from_secs(3))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), false);

    // le: 3s <= 3s（相等）
    let src = format!(
        "{}\n    duration_le(duration_from_secs(3), duration_from_secs(3))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);

    // gt: 5s > 3s
    let src = format!(
        "{}\n    duration_gt(duration_from_secs(5), duration_from_secs(3))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);

    // ge: 5s >= 5s（相等）
    let src = format!(
        "{}\n    duration_ge(duration_from_secs(5), duration_from_secs(5))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);

    // 跨单位比较：500ms < 1s
    let src = format!(
        "{}\n    duration_lt(duration_from_millis(500), duration_from_secs(1))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);
}

// ─── Test 11: to_string 基本格式 ──────────────────────────────────────────

#[test]
fn test_duration_to_string_basic() {
    // to_string(from_secs(5)) == "5s"
    let src = format!(
        "{}\n    duration_to_string(duration_from_secs(5))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "5s");

    // to_string(from_millis(500)) == "500ms"
    let src = format!(
        "{}\n    duration_to_string(duration_from_millis(500))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "500ms");

    // to_string(from_micros(100)) == "100us"
    let src = format!(
        "{}\n    duration_to_string(duration_from_micros(100))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "100us");

    // to_string(from_nanos(42)) == "42ns"
    let src = format!(
        "{}\n    duration_to_string(duration_from_nanos(42))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "42ns");

    // to_string(zero()) == "0ns"
    let src = format!(
        "{}\n    duration_to_string(duration_zero())",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "0ns");
}

// ─── Test 12: to_string 复合单位（分/时）────────────────────────────────

#[test]
fn test_duration_to_string_compound() {
    // 125 秒 = 2 分 5 秒 → "2m5s"
    let src = format!(
        "{}\n    duration_to_string(duration_from_secs(125))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "2m5s");

    // 60 秒 = 1 分 0 秒 → "1m0s"
    let src = format!(
        "{}\n    duration_to_string(duration_from_secs(60))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "1m0s");

    // 3700 秒 = 1 小时 1 分（秒省略）→ "1h1m"
    let src = format!(
        "{}\n    duration_to_string(duration_from_secs(3700))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "1h1m");

    // 3600 秒 = 1 小时 0 分 → "1h0m"
    let src = format!(
        "{}\n    duration_to_string(duration_from_secs(3600))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "1h0m");
}

// ─── Test 13: to_string 负数 ──────────────────────────────────────────────

#[test]
fn test_duration_to_string_negative() {
    // from_secs(-3) → "-3s"
    let src = format!(
        "{}\n    duration_to_string(duration_from_secs(-3))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "-3s");

    // from_millis(-500) → "-500ms"
    let src = format!(
        "{}\n    duration_to_string(duration_from_millis(-500))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "-500ms");

    // from_nanos(-42) → "-42ns"
    let src = format!(
        "{}\n    duration_to_string(duration_from_nanos(-42))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "-42ns");

    // 负数复合单位：from_secs(-125) → "-2m5s"
    let src = format!(
        "{}\n    duration_to_string(duration_from_secs(-125))",
        DURATION_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "-2m5s");
}

// ─── Test 14: zero / is_zero ──────────────────────────────────────────────

#[test]
fn test_duration_zero_is_zero() {
    // zero() → is_zero == true
    let src = format!(
        "{}\n    duration_is_zero(duration_zero())",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);

    // zero().nanos == 0
    let src = format!(
        "{}\n    duration_zero().nanos",
        DURATION_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 0);

    // from_secs(5) 不是零
    let src = format!(
        "{}\n    duration_is_zero(duration_from_secs(5))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), false);

    // sub 自身 → 零
    let src = format!(
        "{}\n    duration_is_zero(duration_sub(duration_from_secs(5), duration_from_secs(5)))",
        DURATION_HELPERS
    );
    assert_eq!(as_bool(run_code(&src).unwrap()), true);
}

// ─── Test 15: 往返一致性：from_secs → as_secs ─────────────────────────────

#[test]
fn test_duration_roundtrip() {
    // from_secs → as_secs 往返
    for n in [0i64, 1, 5, 60, 125, 3600, -3, -125] {
        let src = format!(
            "{}\n    duration_as_secs(duration_from_secs({}))",
            DURATION_HELPERS, n
        );
        assert_eq!(as_i64(run_code(&src).unwrap()), n,
            "from_secs({}) → as_secs roundtrip failed", n);
    }

    // from_millis → as_millis 往返
    for n in [0i64, 1, 500, 1000, -500] {
        let src = format!(
            "{}\n    duration_as_millis(duration_from_millis({}))",
            DURATION_HELPERS, n
        );
        assert_eq!(as_i64(run_code(&src).unwrap()), n,
            "from_millis({}) → as_millis roundtrip failed", n);
    }

    // from_nanos → as_nanos 往返
    for n in [0i64, 1, 42, 999, 1000, -42] {
        let src = format!(
            "{}\n    duration_as_nanos(duration_from_nanos({}))",
            DURATION_HELPERS, n
        );
        assert_eq!(as_i64(run_code(&src).unwrap()), n,
            "from_nanos({}) → as_nanos roundtrip failed", n);
    }
}

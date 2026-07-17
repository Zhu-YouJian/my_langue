//! Date 类型集成测试（Wave 3 第 8 项，路径 B：复用 struct 机制）。
//!
//! 端到端验证：
//! - 5 个 date native（date_to_unix_days / date_from_unix_days / date_add_days /
//!   date_diff_days / date_day_of_week）在解释器路径下的正确性
//! - struct Date 字段访问（year/month/day）
//! - 算法正确性：闰年、跨年、公元前、星期、往返一致性
//!
//! 参考模式：tests/stdlib_test.rs 中 `test_lr_schedule_*` 的内联函数 + run_code 模式
//! （运行时不支持 `use` 加载 .th 模块，故在此内联 std/date.th 中的关键函数）。

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

/// 内联 std/date.th 中的 Date struct 定义和辅助函数，供测试调用。
/// 与 tenth/std/date.th 中的实现保持同步。
const DATE_HELPERS: &str = r#"
    struct Date { year: i64, month: i64, day: i64 }

    fn date_new(year: i64, month: i64, day: i64) -> Date {
        Date { year: year, month: month, day: day }
    }

    fn date_from_days(days: i64) -> Date {
        let (year, month, day) = date_from_unix_days(days);
        Date { year: year, month: month, day: day }
    }

    fn date_to_days(d: Date) -> i64 {
        date_to_unix_days(d.year, d.month, d.day)
    }

    fn date_add_days(d: Date, delta: i64) -> Date {
        let new_days = date_to_days(d) + delta;
        date_from_days(new_days)
    }

    fn date_diff(d1: Date, d2: Date) -> i64 {
        date_diff_days(date_to_days(d1), date_to_days(d2))
    }

    fn date_weekday(d: Date) -> i64 {
        date_day_of_week(date_to_days(d))
    }

    fn pad2(n: i64) -> String {
        if n < 0 { to_string(n) }
        else if n < 10 { "0" + to_string(n) }
        else { to_string(n) }
    }

    fn date_to_string(d: Date) -> String {
        to_string(d.year) + "-" + pad2(d.month) + "-" + pad2(d.day)
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

// ─── Test 1: date_to_unix_days 基本正确性 ─────────────────────────────────

#[test]
fn test_date_to_unix_days_epoch() {
    // 1970-01-01 = Unix epoch = day 0
    let src = format!("{}\n    date_to_unix_days(1970, 1, 1)", DATE_HELPERS);
    assert_eq!(as_i64(run_code(&src).unwrap()), 0);
}

#[test]
fn test_date_to_unix_days_known_dates() {
    let src = format!("{}\n    date_to_unix_days(2026, 7, 17)", DATE_HELPERS);
    // 56*365 + 14 leap days + (Jan31+Feb28+Mar31+Apr30+May31+Jun30 + 16) = 20454 + 197 = 20651
    assert_eq!(as_i64(run_code(&src).unwrap()), 20651);

    let src = format!("{}\n    date_to_unix_days(2000, 1, 1)", DATE_HELPERS);
    assert_eq!(as_i64(run_code(&src).unwrap()), 10957);

    let src = format!("{}\n    date_to_unix_days(1900, 1, 1)", DATE_HELPERS);
    assert_eq!(as_i64(run_code(&src).unwrap()), -25567);
}

#[test]
fn test_date_to_unix_days_leap_day() {
    // 2024-02-29（闰日）
    let src = format!("{}\n    date_to_unix_days(2024, 2, 29)", DATE_HELPERS);
    assert_eq!(as_i64(run_code(&src).unwrap()), 19782);
}

#[test]
fn test_date_to_unix_days_invalid_args_returns_err() {
    // 字符串参数应返回 Err
    let src = format!("{}\n    date_to_unix_days(\"2026\", 7, 17)", DATE_HELPERS);
    let result = run_code(&src);
    assert!(result.is_err(), "期望字符串参数返回 Err，got {:?}", result);
}

// ─── Test 2: date_from_unix_days 返回 Tuple ───────────────────────────────

#[test]
fn test_date_from_unix_days_epoch() {
    // days=0 → (1970, 1, 1)
    let src = format!(
        "{}\n    let (y, m, d) = date_from_unix_days(0); y + m * 10000 + d * 1000000",
        DATE_HELPERS
    );
    // 1970 + 1*10000 + 1*1000000 = 1011970
    assert_eq!(as_i64(run_code(&src).unwrap()), 1011970);
}

#[test]
fn test_date_from_unix_days_pre_epoch() {
    // days=-1 → (1969, 12, 31)
    let src = format!(
        "{}\n    let (y, m, d) = date_from_unix_days(-1); y * 10000 + m * 100 + d",
        DATE_HELPERS
    );
    // 1969 * 10000 + 12 * 100 + 31 = 19691231
    assert_eq!(as_i64(run_code(&src).unwrap()), 19691231);
}

// ─── Test 3: date_to_unix_days ↔ date_from_unix_days 往返一致性 ──────────

#[test]
fn test_roundtrip_modern_dates() {
    for (y, m, d) in [
        (1970, 1, 1),
        (2000, 2, 29),
        (2024, 2, 29),
        (2026, 7, 17),
        (1999, 12, 31),
        (2000, 1, 1),
        (2100, 2, 28),
    ] {
        let src = format!(
            "{}\n    let (y, m, d) = date_from_unix_days(date_to_unix_days({}, {}, {})); y * 10000 + m * 100 + d",
            DATE_HELPERS, y, m, d
        );
        let expected = y * 10000 + m * 100 + d;
        let got = as_i64(run_code(&src).unwrap());
        assert_eq!(got, expected,
            "roundtrip failed for {}-{}-{}: expected {}, got {}", y, m, d, expected, got);
    }
}

#[test]
fn test_roundtrip_pre_1970_dates() {
    for (y, m, d) in [
        (1969, 12, 31),
        (1900, 1, 1),
        (1776, 7, 4),
        (1492, 10, 12),
        (1, 1, 1),
    ] {
        let src = format!(
            "{}\n    let (y, m, d) = date_from_unix_days(date_to_unix_days({}, {}, {})); y * 10000 + m * 100 + d",
            DATE_HELPERS, y, m, d
        );
        let expected = y * 10000 + m * 100 + d;
        let got = as_i64(run_code(&src).unwrap());
        assert_eq!(got, expected,
            "roundtrip failed for {}-{}-{}: expected {}, got {}", y, m, d, expected, got);
    }
}

// ─── Test 4: struct Date 构造与字段访问 ───────────────────────────────────

#[test]
fn test_date_new_field_access() {
    let src = format!(
        "{}\n    let d = date_new(2026, 7, 17); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 20260717);
}

#[test]
fn test_date_to_days_via_struct() {
    // struct Date → date_to_days → native date_to_unix_days
    let src = format!(
        "{}\n    let d = date_new(2026, 7, 17); date_to_days(d)",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 20651);
}

// ─── Test 5: date_add_days 跨月/跨年/减天数 ───────────────────────────────

#[test]
fn test_date_add_days_basic() {
    // 2026-07-17 + 10 = 2026-07-27
    let src = format!(
        "{}\n    let d = date_add_days(date_new(2026, 7, 17), 10); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    let result = run_code(&src);
    if let Err(e) = &result {
        eprintln!("DEBUG test_date_add_days_basic error: {}", e);
    }
    assert_eq!(as_i64(result.unwrap()), 20260727);
}

#[test]
fn test_date_add_days_cross_month() {
    // 2026-07-31 + 1 = 2026-08-01
    let src = format!(
        "{}\n    let d = date_add_days(date_new(2026, 7, 31), 1); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 20260801);
}

#[test]
fn test_date_add_days_cross_year() {
    // 2026-12-31 + 1 = 2027-01-01
    let src = format!(
        "{}\n    let d = date_add_days(date_new(2026, 12, 31), 1); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 20270101);
}

#[test]
fn test_date_add_days_negative_delta() {
    // 2026-01-01 - 1 = 2025-12-31
    let src = format!(
        "{}\n    let d = date_add_days(date_new(2026, 1, 1), -1); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 20251231);
}

#[test]
fn test_date_add_days_epoch_boundary() {
    // 1970-01-01 - 1 = 1969-12-31
    let src = format!(
        "{}\n    let d = date_add_days(date_new(1970, 1, 1), -1); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 19691231);
}

// ─── Test 6: date_diff_days 日期差 ────────────────────────────────────────

#[test]
fn test_date_diff_within_month() {
    // 2026-07-17 - 2026-07-07 = 10
    let src = format!(
        "{}\n    date_diff(date_new(2026, 7, 17), date_new(2026, 7, 7))",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 10);
}

#[test]
fn test_date_diff_across_year_non_leap() {
    // 2026-12-31 - 2026-01-01 = 364
    let src = format!(
        "{}\n    date_diff(date_new(2026, 12, 31), date_new(2026, 1, 1))",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 364);
}

#[test]
fn test_date_diff_across_leap_year() {
    // 2024-12-31 - 2024-01-01 = 365 (2024 闰年)
    let src = format!(
        "{}\n    date_diff(date_new(2024, 12, 31), date_new(2024, 1, 1))",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 365);
}

#[test]
fn test_date_diff_negative() {
    // 2026-07-07 - 2026-07-17 = -10
    let src = format!(
        "{}\n    date_diff(date_new(2026, 7, 7), date_new(2026, 7, 17))",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), -10);
}

// ─── Test 7: date_day_of_week 星期几 ──────────────────────────────────────

#[test]
fn test_date_day_of_week_thursday_1970() {
    // 1970-01-01 = 周四 (4)
    let src = format!("{}\n    date_day_of_week(0)", DATE_HELPERS);
    assert_eq!(as_i64(run_code(&src).unwrap()), 4);
}

#[test]
fn test_date_day_of_week_friday_2026() {
    // 2026-07-17 = 周五 (5)
    let src = format!(
        "{}\n    date_weekday(date_new(2026, 7, 17))",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 5);
}

#[test]
fn test_date_day_of_week_saturday_2000() {
    // 2000-01-01 = 周六 (6)
    let src = format!(
        "{}\n    date_weekday(date_new(2000, 1, 1))",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 6);
}

#[test]
fn test_date_day_of_week_saturday_1949() {
    // 1949-10-01 = 周六 (6)
    let src = format!(
        "{}\n    date_weekday(date_new(1949, 10, 1))",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 6);
}

#[test]
fn test_date_day_of_week_negative_days() {
    // days=-1 (1969-12-31) = 周三 (3)
    let src = format!("{}\n    date_day_of_week(-1)", DATE_HELPERS);
    assert_eq!(as_i64(run_code(&src).unwrap()), 3);
}

// ─── Test 8: date_to_string 格式化 ────────────────────────────────────────

#[test]
fn test_date_to_string_basic() {
    let src = format!(
        "{}\n    date_to_string(date_new(2026, 7, 17))",
        DATE_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "2026-07-17");
}

#[test]
fn test_date_to_string_zero_pad_month_and_day() {
    let src = format!(
        "{}\n    date_to_string(date_new(2026, 1, 1))",
        DATE_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "2026-01-01");

    let src = format!(
        "{}\n    date_to_string(date_new(2026, 1, 9))",
        DATE_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "2026-01-09");

    let src = format!(
        "{}\n    date_to_string(date_new(2026, 10, 1))",
        DATE_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "2026-10-01");

    let src = format!(
        "{}\n    date_to_string(date_new(2026, 12, 31))",
        DATE_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "2026-12-31");
}

#[test]
fn test_date_to_string_pre_epoch() {
    // 1900-01-01
    let src = format!(
        "{}\n    date_to_string(date_new(1900, 1, 1))",
        DATE_HELPERS
    );
    assert_eq!(as_string(run_code(&src).unwrap()), "1900-01-01");
}

// ─── Test 9: 闰年/平年边界 ────────────────────────────────────────────────

#[test]
fn test_leap_year_2024_02_29_roundtrip() {
    // 2024-02-29 (闰日) 往返
    let src = format!(
        "{}\n    let d = date_from_days(date_to_days(date_new(2024, 2, 29))); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 20240229);
}

#[test]
fn test_non_leap_2100_02_28_roundtrip() {
    // 2100-02-28 (2100 是 400 年边界外的平年) 往返
    let src = format!(
        "{}\n    let d = date_from_days(date_to_days(date_new(2100, 2, 28))); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 21000228);
}

#[test]
fn test_leap_year_2000_02_29_roundtrip() {
    // 2000-02-29 (2000 是 400 年边界上的闰年) 往返
    let src = format!(
        "{}\n    let d = date_from_days(date_to_days(date_new(2000, 2, 29))); d.year * 10000 + d.month * 100 + d.day",
        DATE_HELPERS
    );
    assert_eq!(as_i64(run_code(&src).unwrap()), 20000229);
}

//! 日期算法：将 Unix 天数转换为 (年, 月, 日)。
//!
//! 从 `main.rs` 第 316-336 行迁移而来（注释更完整），用于
//! `time_date` / `time_datetime` 原生函数的实现。

/// 将自 0000-03-01 起的天数（已偏移到 1970-01-01 epoch）转换为 (年, 月, 日)。
///
/// 算法来自 Howard Hinnant 的 date_algorithms。
pub fn days_to_date(days: u64) -> (u64, u64, u64) {
    // L-3: 防 `days + 719468` 溢出。days 来自 SystemTime（自 1970-01-01 起的秒数 / 86400），
    // 物理上不可能接近 u64::MAX，但显式校验以杜绝 UB（debug build 会 panic，release 会回绕）。
    // 常量 719468 = Howard Hinnant 算法的 epoch 偏移（1970-01-01 → 0000-03-01）。
    const EPOCH_OFFSET: u64 = 719468;
    if days > u64::MAX - EPOCH_OFFSET {
        // 溢出时返回零值（年/月/日 = 0），调用方据此可识别异常。
        return (0, 0, 0);
    }
    let z = days + EPOCH_OFFSET;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365*yoe + yoe/4 - yoe/100);
    let mp = (5*doy + 2) / 153;
    let d = doy - (153*mp+2)/5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// 将公历日期 (year, month, day) 转换为 Unix days（自 1970-01-01 起的天数）。
///
/// 算法来自 Howard Hinnant 的 date_algorithms（`civil_from_days` 的逆向：
/// `days_from_civil`），与 `days_to_date` 对称。返回 `i64`：
/// - 正数表示 1970-01-01 之后的天数
/// - 0 表示 1970-01-01 当天
/// - 负数表示 1970-01-01 之前的天数（公元前/中世纪等）
///
/// 输入约定：
/// - `year`：任意 i64（负数表示公元前，使用天文年份计数：1 BC = 0，2 BC = -1，...）
/// - `month`：1..=12
/// - `day`：1..=31（实际范围由月/闰年约束）
///
/// 边界处理：与 Howard Hinnant 算法一致，month <= 2 视为前一年的 13/14 月，
/// 使闰日（2/29）成为一年的最后一天，简化闰年判定。
pub fn date_to_days(year: i64, month: i64, day: i64) -> i64 {
    // 将 1/2 月视为前一年的 13/14 月：让闰日落在"年"的末尾。
    // 这样一年内 doy 单调递增，闰年只需检查 yoe/4 - yoe/100。
    let y = if month <= 2 { year - 1 } else { year };
    // era：400 年周期。i64 整除向零截断，y>=0 时正常；y<0 时 (y-399)/400 给出向下取整的 era。
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    // yoe：era 内的年份（0..399）。
    let yoe = y - era * 400;
    // doy：从 3 月 1 日 起的年内序日（3/1=0, 3/2=1, ..., 2/28=364 or 2/29=365 平闰年自适应）。
    // 153*month + 2 是 Howard Hinnant 的月份长度近似（5 个月 = 153 天）。
    let month_shifted = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * month_shifted + 2) / 5 + day - 1;
    // doe：era 内的天数（0..146096）。
    let doe = yoe * 365 + yoe/4 - yoe/100 + doy;
    // 减去 719468：从 0000-03-01 偏移到 1970-01-01 epoch。
    era * 146097 + doe - 719468
}

/// 将 Unix days（i64，可为负数）转换为公历日期 (year, month, day)。
///
/// `days_to_date` 的 i64 对偶版本，支持 1970 之前的日期。
/// 算法与 `days_to_date` 同源（Howard Hinnant date_algorithms），
/// 仅把 `u64` 换成 `i64` 以支持负数输入。
pub fn days_to_date_i64(days: i64) -> (i64, i64, i64) {
    // 加 719468：从 1970-01-01 epoch 偏移到 0000-03-01（算法内部年起点）。
    let z = days + 719468;
    // era：400 年周期。i64 整除对负数向零截断；用 (z-146096)/146097 处理负 z 的向下取整。
    let era = if z >= 0 { z / 146097 } else { (z - 146096) / 146097 };
    let doe = z - era * 146097; // 0..=146096
    let yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365; // 0..=399
    let y = yoe + era * 400;
    let doy = doe - (365*yoe + yoe/4 - yoe/100); // 0..=365
    let mp = (5*doy + 2) / 153; // 0..=11
    let d = doy - (153*mp+2)/5 + 1; // 1..=31
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // 1..=12
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── date_to_days 基本正确性 ──
    #[test]
    fn test_epoch_zero() {
        // 1970-01-01 = Unix epoch = day 0
        assert_eq!(date_to_days(1970, 1, 1), 0);
    }

    #[test]
    fn test_known_dates() {
        // 与 Python datetime.date.toordinal() - 719163 对照（Python ordinal 以公元 1 年 1 月 1 日 = 1）
        // 2026-07-17：(56*365 + 14 leap days) + (Jan31+Feb28+Mar31+Apr30+May31+Jun30 + 16) = 20454 + 197 = 20651
        assert_eq!(date_to_days(2026, 7, 17), 20651);
        // 2000-01-01：date(2000,1,1).toordinal() - 719163 = 10957
        assert_eq!(date_to_days(2000, 1, 1), 10957);
        // 1900-01-01：date(1900,1,1).toordinal() - 719163 = -25567
        assert_eq!(date_to_days(1900, 1, 1), -25567);
    }

    #[test]
    fn test_leap_year_2024_02_29() {
        // 闰日：2024-02-29
        assert_eq!(date_to_days(2024, 2, 29), 19782);
    }

    #[test]
    fn test_year_boundary_dec31_to_jan1() {
        // 2026-12-31 + 1 = 2027-01-01
        let d1 = date_to_days(2026, 12, 31);
        let d2 = date_to_days(2027, 1, 1);
        assert_eq!(d2 - d1, 1);
    }

    #[test]
    fn test_month_before_march_treated_as_prior_year() {
        // 1 月、2 月应被算法视为前一年的 13/14 月。
        // 验证：2024-01-01 与 2023-13-01（虚拟）应得到相同 days；
        // 由于我们不让用户传 month=13，这里验证 2024-01-01 与 2023-12-01 相差 31 天。
        let d_jan = date_to_days(2024, 1, 1);
        let d_dec = date_to_days(2023, 12, 1);
        assert_eq!(d_jan - d_dec, 31);
    }

    // ── date_to_days ↔ days_to_date_i64 往返一致性 ──
    #[test]
    fn test_roundtrip_modern_dates() {
        for (y, m, d) in [
            (1970, 1, 1),
            (2000, 2, 29), // 闰日
            (2024, 2, 29),
            (2026, 7, 17),
            (1999, 12, 31),
            (2000, 1, 1),
            (2100, 2, 28), // 2100 是 400 年边界外的平年（被 100 整除但不被 400 整除）
        ] {
            let days = date_to_days(y, m, d);
            let (y2, m2, d2) = days_to_date_i64(days);
            assert_eq!((y2, m2, d2), (y, m, d),
                "roundtrip failed for {}-{}-{}: got {}-{}-{}", y, m, d, y2, m2, d2);
        }
    }

    #[test]
    fn test_roundtrip_pre_1970_dates() {
        for (y, m, d) in [
            (1969, 12, 31), // epoch 前一天
            (1900, 1, 1),
            (1776, 7, 4), // 美国独立日
            (1492, 10, 12),
            (1, 1, 1), // 公元 1 年 1 月 1 日
        ] {
            let days = date_to_days(y, m, d);
            let (y2, m2, d2) = days_to_date_i64(days);
            assert_eq!((y2, m2, d2), (y, m, d),
                "roundtrip failed for {}-{}-{}: got {}-{}-{}", y, m, d, y2, m2, d2);
        }
    }

    #[test]
    fn test_roundtrip_bce_dates() {
        // 天文年份计数：1 BC = 0，2 BC = -1，...
        for (y, m, d) in [
            (-1, 12, 31), // 2 BC 12-31
            (-1, 1, 1),   // 2 BC 01-01
            (-100, 6, 15),
            (-1000, 3, 1),
        ] {
            let days = date_to_days(y, m, d);
            let (y2, m2, d2) = days_to_date_i64(days);
            assert_eq!((y2, m2, d2), (y, m, d),
                "roundtrip failed for {}-{}-{}: got {}-{}-{}", y, m, d, y2, m2, d2);
        }
    }

    // ── 与现有 u64 版本 days_to_date 一致性（1970 后）──
    #[test]
    fn test_consistency_with_u64_version() {
        for days_u in [0u64, 1, 30, 365, 10957, 19782, 20683, 50000, 100000] {
            let (y1, m1, d1) = days_to_date(days_u);
            let (y2, m2, d2) = days_to_date_i64(days_u as i64);
            assert_eq!((y1 as i64, m1 as i64, d1 as i64), (y2, m2, d2),
                "i64/u64 mismatch at days={}: u64=({},{},{}) i64=({},{},{})",
                days_u, y1, m1, d1, y2, m2, d2);
        }
    }

    // ── 星期计算基础：1970-01-01 是周四 ──
    #[test]
    fn test_1970_01_01_is_thursday() {
        // Unix days = 0 → 周四（用 0=周日, 4=周四 表示）
        let days = date_to_days(1970, 1, 1);
        let weekday = ((days + 4) % 7 + 7) % 7; // 0=周日
        assert_eq!(weekday, 4, "1970-01-01 应为周四 (4)");
    }

    #[test]
    fn test_day_of_week_known_dates() {
        // 2026-07-17 是周五 → 5
        assert_eq!(((date_to_days(2026, 7, 17) + 4) % 7 + 7) % 7, 5);
        // 2000-01-01 是周六 → 6
        assert_eq!(((date_to_days(2000, 1, 1) + 4) % 7 + 7) % 7, 6);
        // 1949-10-01 是周六 → 6
        assert_eq!(((date_to_days(1949, 10, 1) + 4) % 7 + 7) % 7, 6);
    }
}

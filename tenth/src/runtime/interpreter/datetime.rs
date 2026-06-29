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

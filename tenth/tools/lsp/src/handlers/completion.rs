use std::collections::HashSet;

use super::Handler;
use crate::lsp_types::*;
use tenth::lexer::lexer::Lexer;
use tenth::parser::ast::{ExprKind, ItemKind, StmtKind};
use tenth::parser::parser::Parser;

pub struct CompletionHandler;

impl Handler for CompletionHandler {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value {
        let uri = params
            .and_then(|p| p.get("textDocument"))
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or("");

        let position = params
            .and_then(|p| p.get("position"))
            .and_then(|pos| {
                let line = pos.get("line")?.as_u64()? as u32;
                let character = pos.get("character")?.as_u64()? as u32;
                Some(Position { line, character })
            });

        let source = crate::document_store::get_content_or_disk_global(uri)
            .unwrap_or_default();
        let is_method_context = position
            .map(|pos| is_after_dot(&source, pos))
            .unwrap_or(false);

        let items = if is_method_context {
            method_completion_items(&source)
        } else {
            full_completion_items(&source)
        };

        serde_json::to_value(items).unwrap()
    }
}

/// Read the source file content from a file URI.
/// Returns empty string if the file cannot be read.
fn read_source(uri: &str) -> String {
    // Handle file:// URIs
    let path = if let Some(rest) = uri.strip_prefix("file:///") {
        // On Windows, the path after file:/// is like /C:/...
        // strip the leading slash if it looks like a drive letter
        if rest.len() > 2 && rest.chars().nth(1) == Some(':') {
            &rest[1..]
        } else {
            rest
        }
    } else if let Some(rest) = uri.strip_prefix("file://") {
        rest
    } else {
        uri
    };

    std::fs::read_to_string(path).unwrap_or_default()
}

/// Check if the cursor position is immediately after a `.` character,
/// indicating a method call or field access context.
fn is_after_dot(source: &str, position: Position) -> bool {
    let lines: Vec<&str> = source.lines().collect();
    let line_idx = position.line as usize;
    if line_idx >= lines.len() {
        return false;
    }
    let line = lines[line_idx];
    let char_idx = position.character as usize;
    if char_idx == 0 || char_idx > line.len() {
        return false;
    }
    // Check the character just before the cursor
    let before = &line[..char_idx];
    before.ends_with('.')
}

/// Build completion items for method/field context (after a `.`).
/// Includes methods from impl blocks and struct fields.
fn method_completion_items(source: &str) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    if let Some(program) = parse_source(source) {
        let mut method_names: HashSet<String> = HashSet::new();

        for item in &program.items {
            if let ItemKind::Impl { functions, .. } = &item.kind {
                for func in functions {
                    if let ItemKind::Function { name, .. } = &func.kind {
                        if method_names.insert(name.name.clone()) {
                            items.push(CompletionItem {
                                label: name.name.clone(),
                                kind: CompletionItemKind::Method,
                                detail: Some(format!("method {}", name.name)),
                            documentation: None,
                            insert_text: None,
                        });
                        }
                    }
                }
            }
        }
    }

    // Fallback: if no methods found, still return empty — the client
    // will show no completions which is correct for unknown types
    items
}

/// Build the full set of completion items: static keywords/builtins/types
/// plus user-defined symbols and local variables from the parsed program.
fn full_completion_items(source: &str) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    // Add user-defined symbols from the parsed program
    if let Some(program) = parse_source(source) {
        add_user_defined_items(&mut items, &program);
        add_local_variables(&mut items, &program);
    }

    // Add static keywords, builtins, and types as fallback
    add_keywords(&mut items);
    add_builtins(&mut items);
    add_types(&mut items);

    items
}

/// Try to parse the source file. Returns Some(Program) on success,
/// None on failure (caller falls back to static list).
fn parse_source(source: &str) -> Option<tenth::parser::ast::Program> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().ok()?;
    let mut parser = Parser::new(tokens);
    // Use parse_program_with_recovery for fault tolerance in LSP context
    let (program, _errors) = parser.parse_program_with_recovery();
    Some(program)
}

/// Extract user-defined functions, structs, and enums from the AST
/// and add them as completion items.
fn add_user_defined_items(items: &mut Vec<CompletionItem>, program: &tenth::parser::ast::Program) {
    let mut seen_funcs: HashSet<String> = HashSet::new();
    let mut seen_structs: HashSet<String> = HashSet::new();
    let mut seen_enums: HashSet<String> = HashSet::new();

    for item in &program.items {
        match &item.kind {
            ItemKind::Function { name, .. } => {
                if seen_funcs.insert(name.name.clone()) {
                    items.push(CompletionItem {
                        label: name.name.clone(),
                        kind: CompletionItemKind::Function,
                        detail: Some(format!("fn {}", name.name)),
                        documentation: None,
                        insert_text: None,
                    });
                }
            }
            ItemKind::StructDef { name, .. } => {
                if seen_structs.insert(name.name.clone()) {
                    items.push(CompletionItem {
                        label: name.name.clone(),
                        kind: CompletionItemKind::Class,
                        detail: Some(format!("struct {}", name.name)),
                        documentation: None,
                        insert_text: None,
                    });
                }
            }
            ItemKind::EnumDef { name, .. } => {
                if seen_enums.insert(name.name.clone()) {
                    items.push(CompletionItem {
                        label: name.name.clone(),
                        kind: CompletionItemKind::Class,
                        detail: Some(format!("enum {}", name.name)),
                        documentation: None,
                        insert_text: None,
                    });
                }
            }
            ItemKind::Impl { functions, .. } => {
                for func in functions {
                    if let ItemKind::Function { name, .. } = &func.kind {
                        // Impl methods are added as Function (not Method) in
                        // top-level completions; they appear as Method only in
                        // dot-completion context.
                        if seen_funcs.insert(name.name.clone()) {
                            items.push(CompletionItem {
                                label: name.name.clone(),
                                kind: CompletionItemKind::Function,
                                detail: Some(format!("fn {}", name.name)),
                                documentation: None,
                                insert_text: None,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Extract local variables from the parsed program: function parameters,
/// `let` bindings, `for` loop variables, and closure parameters.
fn add_local_variables(items: &mut Vec<CompletionItem>, program: &tenth::parser::ast::Program) {
    let mut seen: HashSet<String> = HashSet::new();
    for item in &program.items {
        if let ItemKind::Function { params, body, .. } = &item.kind {
            for p in params {
                collect_local_var(items, &mut seen, &p.name.name, "参数");
            }
            walk_expr_for_locals(items, &mut seen, body);
        }
    }
}

/// Add a single local variable completion item (deduplicated).
fn collect_local_var(
    items: &mut Vec<CompletionItem>,
    seen: &mut HashSet<String>,
    name: &str,
    detail: &str,
) {
    if name.is_empty() || seen.contains(name) {
        return;
    }
    seen.insert(name.to_string());
    items.push(CompletionItem {
        label: name.to_string(),
        kind: CompletionItemKind::Variable,
        detail: Some(detail.to_string()),
        documentation: None,
        insert_text: None,
    });
}

/// Best-effort AST walk over an expression, collecting local bindings in nested
/// blocks / closures / loops. Unknown expression kinds are skipped safely.
fn walk_expr_for_locals(
    items: &mut Vec<CompletionItem>,
    seen: &mut HashSet<String>,
    expr: &tenth::parser::ast::Expr,
) {
    match &expr.kind {
        ExprKind::Block(stmts) => {
            for s in stmts {
                walk_stmt_for_locals(items, seen, s);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            walk_expr_for_locals(items, seen, left);
            walk_expr_for_locals(items, seen, right);
        }
        ExprKind::CustomBinary { left, right, .. } => {
            walk_expr_for_locals(items, seen, left);
            walk_expr_for_locals(items, seen, right);
        }
        ExprKind::Unary { expr: inner, .. } => walk_expr_for_locals(items, seen, inner),
        ExprKind::Call { func, args } => {
            walk_expr_for_locals(items, seen, func);
            for a in args {
                walk_expr_for_locals(items, seen, a);
            }
        }
        ExprKind::GenericCall { func, args, .. } => {
            walk_expr_for_locals(items, seen, func);
            for a in args {
                walk_expr_for_locals(items, seen, a);
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            walk_expr_for_locals(items, seen, receiver);
            for a in args {
                walk_expr_for_locals(items, seen, a);
            }
        }
        ExprKind::Index { target, indices } => {
            walk_expr_for_locals(items, seen, target);
            for idx in indices {
                if let tenth::parser::ast::IndexExpr::Single(e) = idx {
                    walk_expr_for_locals(items, seen, e);
                }
            }
        }
        ExprKind::Field { target, .. } => walk_expr_for_locals(items, seen, target),
        ExprKind::TensorLiteral(rows) => {
            for row in rows {
                for e in row {
                    walk_expr_for_locals(items, seen, e);
                }
            }
        }
        ExprKind::ArrayLiteral(es) => {
            for e in es {
                walk_expr_for_locals(items, seen, e);
            }
        }
        ExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                walk_expr_for_locals(items, seen, s);
            }
            if let Some(e) = end {
                walk_expr_for_locals(items, seen, e);
            }
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            walk_expr_for_locals(items, seen, cond);
            walk_expr_for_locals(items, seen, then_branch);
            if let Some(e) = else_branch {
                walk_expr_for_locals(items, seen, e);
            }
        }
        ExprKind::Closure { params, body } => {
            for (ident, _) in params {
                collect_local_var(items, seen, &ident.name, "闭包参数");
            }
            walk_expr_for_locals(items, seen, body);
        }
        ExprKind::Assign { target, value } => {
            walk_expr_for_locals(items, seen, target);
            walk_expr_for_locals(items, seen, value);
        }
        ExprKind::AssignOp { target, value, .. } => {
            walk_expr_for_locals(items, seen, target);
            walk_expr_for_locals(items, seen, value);
        }
        ExprKind::StructLiteral { fields, .. } => {
            for (_, e) in fields {
                walk_expr_for_locals(items, seen, e);
            }
        }
        ExprKind::EnumLiteral { fields, .. } => {
            for (_, e) in fields {
                walk_expr_for_locals(items, seen, e);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            walk_expr_for_locals(items, seen, scrutinee);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    walk_expr_for_locals(items, seen, g);
                }
                walk_expr_for_locals(items, seen, &arm.body);
            }
        }
        ExprKind::Ref(e)
        | ExprKind::MutRef(e)
        | ExprKind::Deref(e)
        | ExprKind::Move(e)
        | ExprKind::Lossy(e)
        | ExprKind::TryBlock(e)
        | ExprKind::Await(e)
        | ExprKind::Spawn(e) => {
            walk_expr_for_locals(items, seen, e);
        }
        ExprKind::Yield(Some(e)) => walk_expr_for_locals(items, seen, e),
        ExprKind::NamedArg { value, .. } => walk_expr_for_locals(items, seen, value),
        _ => {}
    }
}

/// Best-effort AST walk over a statement, collecting local bindings.
fn walk_stmt_for_locals(
    items: &mut Vec<CompletionItem>,
    seen: &mut HashSet<String>,
    stmt: &tenth::parser::ast::Stmt,
) {
    match &stmt.kind {
        StmtKind::Let { names, init, .. } => {
            for n in names {
                collect_local_var(items, seen, &n.name, "变量");
            }
            if let Some(e) = init {
                walk_expr_for_locals(items, seen, e);
            }
        }
        StmtKind::Expr(e) => walk_expr_for_locals(items, seen, e),
        StmtKind::Return(Some(e)) => walk_expr_for_locals(items, seen, e),
        StmtKind::Break { value: Some(e), .. } => walk_expr_for_locals(items, seen, e),
        StmtKind::While { cond, body, .. } => {
            walk_expr_for_locals(items, seen, cond);
            walk_stmt_for_locals(items, seen, body);
        }
        StmtKind::DoWhile {
            body, condition, ..
        } => {
            walk_stmt_for_locals(items, seen, body);
            walk_expr_for_locals(items, seen, condition);
        }
        StmtKind::For { var, iter, body, .. } => {
            collect_local_var(items, seen, &var.name, "循环变量");
            walk_expr_for_locals(items, seen, iter);
            walk_stmt_for_locals(items, seen, body);
        }
        StmtKind::Loop { body, .. } => {
            for s in body {
                walk_stmt_for_locals(items, seen, s);
            }
        }
        _ => {}
    }
}

fn add_keywords(items: &mut Vec<CompletionItem>) {
    let keywords = [
        "fn", "let", "mut", "if", "else", "while", "for", "in", "return",
        "struct", "enum", "impl", "trait", "import", "tensor", "true", "false",
        "match", "mod", "use", "pub", "const", "ref", "move", "loop",
        "break", "continue",
    ];
    for kw in &keywords {
        items.push(CompletionItem {
            label: kw.to_string(),
            kind: CompletionItemKind::Keyword,
            detail: Some(format!("keyword {}", kw)),
            documentation: None,
            insert_text: None,
        });
    }
}

/// 内置 native 符号集，来源：`tenth/std/prelude.th`（权威内置函数清单，
/// "always available, no import needed"）。
fn add_builtins(items: &mut Vec<CompletionItem>) {
    let builtins: &[(&str, &str)] = &[
        // ── 输出 ──
        ("print", "输出到 stdout（不换行）"),
        ("println", "输出到 stdout（换行）"),
        ("eprint", "输出到 stderr（不换行）"),
        ("eprintln", "输出到 stderr（换行）"),
        // ── 张量构造 ──
        ("tensor", "从数据创建张量"),
        ("zeros", "创建零张量"),
        ("ones", "创建全一张量"),
        ("zeros_f32", "创建 f32 零张量"),
        ("ones_f32", "创建 f32 全一张量"),
        ("rand", "随机张量（f64）"),
        ("randn", "正态随机张量（f64）"),
        ("rand_f32", "随机张量（f32）"),
        ("randn_f32", "正态随机张量（f32）"),
        // ── 自动微分 ──
        ("param", "声明可学习参数"),
        ("grad", "计算张量梯度"),
        ("stop_grad", "停止梯度传播"),
        ("new_grad", "创建新梯度上下文"),
        ("start_grad", "开始梯度记录"),
        ("backward", "执行反向传播"),
        ("zero_grad", "清零梯度"),
        ("cross_entropy", "交叉熵损失"),
        // ── 张量比较 / 选择 ──
        ("select", "条件选择原语 (cond ? then : else)"),
        ("tensor_gt", "张量比较 >（返回 0.0/1.0 张量）"),
        ("tensor_lt", "张量比较 <"),
        ("tensor_ge", "张量比较 >="),
        ("tensor_le", "张量比较 <="),
        ("tensor_eq", "张量比较 =="),
        ("tensor_ne", "张量比较 !="),
        // ── 集合 / 智能指针 ──
        ("Vec::new", "创建动态数组"),
        ("HashMap::new", "创建哈希表"),
        ("Box::new", "堆分配 Box"),
        ("Rc::new", "引用计数 Rc"),
        ("Arc::new", "原子引用计数 Arc"),
        ("Pin::new", "固定 Pin"),
        ("Weak::new", "从 Rc/Arc 创建弱引用"),
        ("weak_upgrade", "弱引用取强引用 -> Option"),
        ("weak_strong_count", "当前强引用数"),
        ("weak_weak_count", "当前弱引用数"),
        // ── 文件 I/O ──
        ("read_file", "读取文件为字符串"),
        ("write_bytes", "写入字节到文件"),
        ("read_bytes", "读取文件为字节"),
        ("save_weights", "保存模型权重"),
        ("load_weights", "加载模型权重"),
        ("path_join", "拼接路径"),
        ("path_exists", "路径是否存在"),
        ("path_is_file", "是否为文件"),
        ("path_is_dir", "是否为目录"),
        ("mkdir", "创建目录"),
        ("list_dir", "列出目录内容"),
        ("file_size", "文件大小"),
        ("remove_file", "删除文件"),
        ("copy_file", "复制文件"),
        ("rename_file", "重命名文件"),
        // ── 时间 ──
        ("time_now", "当前 Unix 时间戳 (f64 秒)"),
        ("time_now_ms", "当前 Unix 时间戳 (毫秒)"),
        ("time_date", "当前日期 YYYY-MM-DD"),
        ("time_time", "当前时间 HH:MM:SS"),
        ("time_datetime", "当前日期时间"),
        ("time_sleep_ms", "休眠指定毫秒"),
        // ── 日期 ──
        ("date_to_unix_days", "公历日期 -> Unix days"),
        ("date_from_unix_days", "Unix days -> (年,月,日)"),
        ("date_i64_add_days", "Unix days + delta"),
        ("date_diff_days", "两个 Unix days 之差"),
        ("date_day_of_week", "星期几 (0=周日..6=周六)"),
        // ── 随机 ──
        ("random_int", "[lo, hi] 随机整数"),
        ("random_float", "[0, 1) 随机浮点"),
        // ── 数学 ──
        ("abs", "绝对值"),
        ("sqrt", "平方根"),
        ("sin", "正弦"),
        ("cos", "余弦"),
        ("ln", "自然对数"),
        ("pow", "幂"),
        ("exp", "指数"),
        ("math_tan", "正切"),
        ("math_asin", "反正弦"),
        ("math_acos", "反余弦"),
        ("math_atan", "反正切"),
        ("math_atan2", "二参数反正切"),
        ("math_sinh", "双曲正弦"),
        ("math_cosh", "双曲余弦"),
        ("math_tanh", "双曲正切"),
        ("math_log10", "以 10 为底对数"),
        ("math_log2", "以 2 为底对数"),
        ("math_floor", "向下取整"),
        ("math_ceil", "向上取整"),
        ("math_round", "四舍五入"),
        // ── CLI ──
        ("cli_args_count", "命令行参数个数"),
        ("cli_arg", "按索引取命令行参数"),
        // ── JSON ──
        ("json_encode", "编码为 JSON 字符串"),
        ("json_encode_pretty", "美化 JSON 编码"),
        ("json_decode", "解析 JSON 字符串"),
        // ── I/O / 环境 ──
        ("read_line", "从 stdin 读一行 -> Result<String>"),
        ("env_get", "获取环境变量 -> Result<String>"),
        ("env_set", "设置环境变量"),
        ("exit", "以退出码终止进程"),
        // ── 静默失败防护 ──
        ("or_die", "Result/Option 显式解包，Err/None 则 panic"),
        ("assume_ok", "声明保证成功，直接取内部值"),
        // ── TCP ──
        ("tcp_connect", "连接 TCP -> Result<i64>"),
        ("tcp_read", "读最多 n 字节 -> Result<Vec<i64>>"),
        ("tcp_write", "写字节 -> Result<i64>"),
        ("tcp_close", "关闭连接"),
        ("tcp_set_timeout", "设置读写超时"),
        ("tcp_listen", "监听地址 -> Result<i64>"),
        ("tcp_accept", "接受连接 -> Result<i64>"),
        ("tcp_listener_close", "关闭监听器"),
        // ── UDP ──
        ("udp_bind", "绑定 UDP 套接字 -> Result<i64>"),
        ("udp_recv_from", "接收字节 + 对端地址"),
        ("udp_send_to", "发送字节到地址"),
        ("udp_close", "关闭 UDP 套接字"),
        ("udp_set_timeout", "设置 UDP 超时"),
        // ── 子进程 ──
        ("command_new", "创建子进程命令 -> Result<i64>"),
        ("command_arg", "添加命令参数"),
        ("command_run", "执行并等待 -> Result<i64>"),
        ("command_output", "执行并捕获 stdout -> Result<String>"),
        // ── HTTP ──
        ("http_get", "HTTP GET 请求 -> Result<String>"),
        ("http_post", "HTTP POST 请求 -> Result<String>"),
        // ── 异步 I/O ──
        ("async_sleep_ms", "异步休眠 -> Future<Unit>"),
        ("async_tcp_read", "异步 TCP 读 -> Future<Result<Vec<i64>>>"),
        ("async_tcp_write", "异步 TCP 写 -> Future<Result<i64>>"),
        // ── 正则 ──
        ("regex_compile", "编译正则 -> Result<i64>"),
        ("regex_match", "完整匹配检查 -> bool"),
        ("regex_find", "找首个匹配 -> String"),
        ("regex_find_all", "找全部匹配 -> Vec<String>"),
        ("regex_replace", "正则替换 -> String"),
        ("regex_split", "正则切分 -> Vec<String>"),
        // ── 文本编码（B批）──
        ("unicode_nfc", "Unicode NFC 规范化"),
        ("unicode_nfd", "Unicode NFD 规范化"),
        ("str_to_utf16", "字符串 -> UTF-16 码元数组"),
        ("utf16_to_str", "UTF-16 码元数组 -> 字符串"),
        ("str_to_bytes", "字符串 -> UTF-8 字节数组"),
        ("bytes_to_str", "UTF-8 字节数组 -> 字符串"),
        ("to_utf8", "str_to_bytes 别名"),
        ("to_utf16", "str_to_utf16 别名"),
        ("from_utf16", "utf16_to_str 别名"),
        ("to_gbk", "字符串 -> GBK 字节数组"),
        ("from_gbk", "GBK 字节数组 -> 字符串"),
        ("base64_encode", "字节数组 -> Base64 字符串"),
        ("base64_decode", "Base64 -> Result<Vec>"),
        ("hex_encode", "字节数组 -> 十六进制字符串"),
        ("hex_decode", "十六进制 -> Result<Vec>"),
        ("url_encode", "字符串 -> URL 编码"),
        ("url_decode", "URL 编码 -> Result<String>"),
        // ── 哈希 ──
        ("sha256", "SHA-256 哈希 (hex 输出)"),
        ("sha512", "SHA-512 哈希 (hex 输出)"),
        ("md5", "MD5 哈希（不安全，仅校验和）"),
        ("sha256_str", "SHA-256 字符串便捷版"),
        ("sha512_str", "SHA-512 字符串便捷版"),
        ("md5_str", "MD5 字符串便捷版"),
        // ── 断言 / 运行时限制 ──
        ("assert", "断言（失败则运行时错误）"),
        ("assert_eq", "断言相等"),
        ("with_step_limit", "以步数限制执行闭包"),
        ("with_timeout_ms", "以毫秒超时执行闭包"),
        ("is_timeout", "判断错误是否为超时"),
        // ── 通用工具 ──
        ("len", "返回集合长度"),
        ("shape", "返回张量 shape"),
        ("reshape", "重塑张量"),
        ("range", "创建迭代范围"),
        ("format", "格式化字符串"),
        ("parse_int", "解析整数"),
        ("parse_float", "解析浮点"),
        ("max", "最大值"),
        ("min", "最小值"),
        ("argmax", "最大值的索引"),
    ];
    for (name, doc) in builtins {
        items.push(CompletionItem {
            label: name.to_string(),
            kind: CompletionItemKind::Function,
            detail: Some(doc.to_string()),
            documentation: None,
            insert_text: None,
        });
    }
}

fn add_types(items: &mut Vec<CompletionItem>) {
    let types = [
        ("i64", "64-bit signed integer"),
        ("f64", "64-bit floating point"),
        ("bool", "Boolean type"),
        ("String", "UTF-8 string type"),
        ("Tensor", "N-dimensional tensor"),
        ("Vec", "Dynamic array"),
        ("Option", "Optional value"),
        ("Result", "Result type"),
    ];
    for (name, doc) in &types {
        items.push(CompletionItem {
            label: name.to_string(),
            kind: CompletionItemKind::Class,
            detail: Some(doc.to_string()),
            documentation: None,
            insert_text: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_after_dot_true() {
        // 光标在 "x." 之后（character=2，正好在 '.' 之后的位置）
        let src = "x.";
        let pos = Position { line: 0, character: 2 };
        assert!(
            is_after_dot(src, pos),
            "expected is_after_dot=true for cursor right after '.'"
        );
    }

    #[test]
    fn test_is_after_dot_false_in_identifier() {
        // 光标在标识符中间，前面没有 '.'
        let src = "let variable = 0;";
        // 光标在 character 5（"varia|ble" 中）
        let pos = Position { line: 0, character: 5 };
        assert!(
            !is_after_dot(src, pos),
            "expected is_after_dot=false when cursor is mid-identifier"
        );
    }

    #[test]
    fn test_is_after_dot_false_at_line_start() {
        // 光标在行首（character=0），前面没有 '.'
        let src = "fn main() -> i32 { 0 }";
        let pos = Position { line: 0, character: 0 };
        assert!(
            !is_after_dot(src, pos),
            "expected is_after_dot=false at line start"
        );
    }

    #[test]
    fn test_is_after_dot_false_for_line_out_of_range() {
        // 行号超出范围应返回 false
        let src = "let x = 0;";
        let pos = Position { line: 100, character: 0 };
        assert!(
            !is_after_dot(src, pos),
            "expected is_after_dot=false when line out of range"
        );
    }

    #[test]
    fn test_is_after_dot_true_with_more_text() {
        // "obj.method" 中光标正好在 '.' 之后（character=4，即 "obj.|method"）
        let src = "obj.method";
        let pos = Position { line: 0, character: 4 };
        assert!(
            is_after_dot(src, pos),
            "expected is_after_dot=true when cursor is right after '.'"
        );
    }

    #[test]
    fn test_full_completion_items_contains_keywords() {
        // 完整补全列表应包含关键字
        let items = full_completion_items("");
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        // 至少应包含 "fn" 这个关键字
        assert!(
            labels.contains(&"fn"),
            "expected completions to contain 'fn' keyword, got: {:?}",
            labels.iter().take(10).collect::<Vec<_>>()
        );
        // 应包含 "let" 关键字
        assert!(
            labels.contains(&"let"),
            "expected completions to contain 'let' keyword"
        );
    }

    #[test]
    fn test_full_completion_contains_prelude_builtins() {
        // 补全应包含 prelude 内置 native 符号（来自 tenth/std/prelude.th）
        let items = full_completion_items("");
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        // 覆盖多个类别：静默失败防护 / TCP / 哈希 / 断言 / 张量
        for expect in &["or_die", "assume_ok", "tcp_connect", "sha256", "assert_eq", "tensor_gt", "weak_upgrade", "http_get", "base64_encode", "start_grad"] {
            assert!(
                labels.contains(expect),
                "expected completion to contain '{}', got {} items (first 20: {:?})",
                expect,
                labels.len(),
                labels.iter().take(20).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_full_completion_contains_local_variables() {
        // let 绑定 / for 循环变量 / 闭包参数 / 函数参数应作为 Variable 补全
        let src = r#"
fn compute(a: i64, b: i64) -> i64 {
    let sum = a + b;
    for i in range(0, 10) {
        sum = sum + i;
    }
    let f = |x: i64| x * 2;
    sum
}
"#;
        let items = full_completion_items(src);
        let vars: Vec<&CompletionItem> = items
            .iter()
            .filter(|i| matches!(i.kind, CompletionItemKind::Variable))
            .collect();
        let labels: Vec<&str> = vars.iter().map(|i| i.label.as_str()).collect();
        for expect in &["a", "b", "sum", "i", "x", "f"] {
            assert!(
                labels.contains(expect),
                "expected local variable '{}' in completions, got: {:?}",
                expect,
                labels
            );
        }
        // 局部变量必须是 Variable 类型
        for v in &vars {
            assert_eq!(v.kind, CompletionItemKind::Variable);
        }
    }

    #[test]
    fn test_full_completion_contains_user_functions_and_locals() {
        // 用户定义函数 + 局部变量同时出现
        let src = "fn add(a: i32, b: i32) -> i32 { let c = a + b; c }";
        let items = full_completion_items(src);
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(labels.contains(&"add"), "user function 'add' missing");
        assert!(labels.contains(&"a"), "parameter 'a' missing");
        assert!(labels.contains(&"c"), "local 'c' missing");
    }
}

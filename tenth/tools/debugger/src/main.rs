//! Tenth 调试器（CLI，基于解释器路径）。
//!
//! M4.4：断点（按行）+ 单步 + 变量查看 + 继续。
//!
//! 原理：tree-walk 解释器逐语句执行——`Interpreter::debug_hook` 在每个
//! `HirStmt` 执行前被调用（`stmt.span.line` 提供源码行号，`interp.vars` 提供
//! 带名字的变量表）。钩子在命中断点/单步时阻塞等待用户命令（同步交互式），
//! 因此无需 VM 的 suspend/resume 机制，也不改变被调程序语义（仅观察，不注入）。
//!
//! 用法：
//!   tenth-debug <file.th> [--bp N]...
//!   交互命令：n(next) c(continue) p <var>(print) l(list) b [line](break)
//!             d <line>(delete) q(quit) h(help)

use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};
use std::path::Path;

use tenth::error::{TenthError, TenthResult};
use tenth::hir::hir::HirProgram;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::hir::hir::HirStmt;

/// 调试器状态（hook 闭包捕获）。
struct Debugger {
    source_lines: Vec<String>,
    breakpoints: BTreeSet<usize>,
    stepping: bool,
    /// 是否已停止（quit 后不再暂停）。
    stopped: bool,
}

impl Debugger {
    fn new(source: &str, initial_bps: &[usize]) -> Self {
        Debugger {
            source_lines: source.lines().map(|l| l.to_string()).collect(),
            breakpoints: initial_bps.iter().copied().collect(),
            stepping: false,
            stopped: false,
        }
    }

    fn source_line(&self, line: usize) -> String {
        if line == 0 || line > self.source_lines.len() {
            String::new()
        } else {
            self.source_lines[line - 1].trim().to_string()
        }
    }

    /// 打印当前位置与源码行。
    fn show_location(&self, line: usize) {
        println!("\n── 停在第 {} 行 ──", line);
        if line > 0 {
            let start = line.saturating_sub(2);
            let end = (line + 1).min(self.source_lines.len());
            for i in start..end {
                let marker = if i + 1 == line { ">" } else { " " };
                println!("{} {:>4} | {}", marker, i + 1, self.source_lines[i]);
            }
        }
        println!("──────────────");
    }
}

/// 打印一个变量的值（取当前作用域顶层绑定）。
fn print_var(interp: &Interpreter, name: &str) {
    match interp.vars.get(name) {
        Some(stack) => {
            if let Some((depth, val)) = stack.last() {
                println!("  {} : {} = {}", name, val.type_of(), val);
                if *depth > 0 {
                    println!("  (scope depth {})", depth);
                }
            } else {
                println!("  undefined variable: {}", name);
            }
        }
        None => println!("  undefined variable: {}", name),
    }
}

fn print_help() {
    println!("Tenth 调试器命令:");
    println!("  n / next        单步到下一个语句");
    println!("  c / continue    继续运行到下一个断点");
    println!("  p <name>        查看变量值");
    println!("  l [N]           列出当前行附近 N 行（默认 10）");
    println!("  b               列出所有断点");
    println!("  b <line>        在第 line 行设置断点");
    println!("  d <line>        删除第 line 行断点");
    println!("  q / quit        退出调试器（中止执行）");
    println!("  h / help        显示此帮助");
}

/// lex → parse → lower → HIR（与主编译器 source_to_hir 对齐，含 std 搜索路径）。
fn source_to_hir(source: &str) -> TenthResult<HirProgram> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;

    let mut search_paths = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        search_paths.push(cwd.to_string_lossy().to_string());
    }
    if let Ok(exe_dir) = std::env::current_exe()
        .map(|p| p.parent().map(|d| d.to_path_buf()).unwrap_or_default())
    {
        let std_near_exe = exe_dir.join("std");
        if std_near_exe.exists() {
            search_paths.push(std_near_exe.to_string_lossy().to_string());
        }
    }
    let std_dev = Path::new("tenth/std");
    if std_dev.exists() {
        if let Some(parent) = std_dev.parent() {
            search_paths.push(parent.to_string_lossy().to_string());
        }
        search_paths.push(std_dev.to_string_lossy().to_string());
    }
    let std_local = Path::new("std");
    if std_local.exists() {
        if let Some(parent) = std_local.parent() {
            let parent_str = parent.to_string_lossy().to_string();
            if !search_paths.iter().any(|p| *p == parent_str) {
                search_paths.push(parent_str);
            }
        }
        let std_str = std_local.to_string_lossy().to_string();
        if !search_paths.iter().any(|p| *p == std_str) {
            search_paths.push(std_str);
        }
    }

    let mut lowerer = Lowerer::with_search_paths(search_paths);
    lowerer.lower_program(&program)
}

fn main() {
    #[cfg(windows)]
    unsafe {
        unsafe extern "C" {
            fn SetConsoleOutputCP(code_page: u32) -> i32;
        }
        SetConsoleOutputCP(65001);
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: tenth-debug <file.th> [--bp N]...");
        eprintln!("  --bp N   启动前在第 N 行设置断点（可多次指定）");
        std::process::exit(2);
    }
    let path = &args[1];
    let mut initial_bps = Vec::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--bp" | "-b" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        initial_bps.push(n);
                        i += 2;
                        continue;
                    }
                }
                eprintln!("--bp 需要行号参数");
                std::process::exit(2);
            }
            other => {
                eprintln!("未知参数: {}", other);
                std::process::exit(2);
            }
        }
    }

    if let Err(e) = run_debug(path, &initial_bps) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_debug(path: &str, initial_bps: &[usize]) -> TenthResult<()> {
    let source = tenth::error::read_source(path)?;
    let hir = match source_to_hir(&source) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e.display_with_source(Some(&source)));
            return Ok(());
        }
    };
    for w in &hir.warnings {
        eprintln!("{}", w.display_with_source(Some(&source)));
    }

    let mut dbg = Debugger::new(&source, initial_bps);
    if !initial_bps.is_empty() {
        println!("断点: {:?}", initial_bps);
    }

    let mut interp = Interpreter::new(&hir);
    interp.set_debug_hook(Some(Box::new(move |interp: &mut Interpreter, stmt: &HirStmt| {
        hook(interp, stmt, &mut dbg)
    })));

    // 从第一行开始单步（便于交互调试脚本型程序）。
    let result = interp.execute_program(&hir);
    match result {
        Ok(_) => {
            println!("\n程序执行完毕。");
            Ok(())
        }
        Err(e) => {
            // 调试器退出是主动中止，不算错误。
            if e.to_string().contains("[debugger] 退出") {
                println!("\n调试器退出。");
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

/// 调试钩子：每个语句执行前调用。
fn hook(interp: &mut Interpreter, stmt: &HirStmt, dbg: &mut Debugger) -> TenthResult<()> {
    if dbg.stopped {
        return Ok(());
    }
    let line = stmt.span.line;
    let hit = dbg.breakpoints.contains(&line) || dbg.stepping;
    if !hit {
        return Ok(());
    }

    dbg.stepping = false;
    dbg.show_location(line);
    println!("提示: 输入 h 查看命令");

    let stdin = io::stdin();
    loop {
        print!("(tenth-dbg) ");
        io::stdout().flush().ok();
        let mut input = String::new();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => {
                // EOF → 继续运行到结束
                dbg.stepping = false;
                return Ok(());
            }
            Ok(_) => {}
            Err(e) => return Err(runtime_err(&format!("读取命令失败: {}", e))),
        }
        let cmd = input.trim();
        if cmd.is_empty() {
            continue;
        }
        let mut parts = cmd.split_whitespace();
        let op = parts.next().unwrap_or("");
        match op {
            "n" | "next" => {
                dbg.stepping = true;
                return Ok(());
            }
            "c" | "continue" => {
                dbg.stepping = false;
                return Ok(());
            }
            "p" | "print" => {
                let name = parts.next().unwrap_or("");
                if name.is_empty() {
                    println!("用法: p <变量名>");
                } else {
                    print_var(interp, name);
                }
            }
            "l" | "list" => {
                let n: usize = parts
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10)
                    .min(40);
                if line > 0 {
                    let start = line.saturating_sub(n / 2);
                    let end = (line + n / 2).min(dbg.source_lines.len());
                    for i in start..=end {
                        if i == 0 {
                            continue;
                        }
                        let marker = if i == line { ">" } else { " " };
                        println!("{} {:>4} | {}", marker, i, dbg.source_lines[i - 1]);
                    }
                }
            }
            "b" | "break" => {
                match parts.next() {
                    Some(line_str) => match line_str.parse::<usize>() {
                        Ok(n) if n >= 1 => {
                            dbg.breakpoints.insert(n);
                            println!("断点已设置在行 {}", n);
                        }
                        _ => println!("行号无效: {}", line_str),
                    },
                    None => {
                        if dbg.breakpoints.is_empty() {
                            println!("(无断点)");
                        } else {
                            for bp in &dbg.breakpoints {
                                println!("  行 {}: {}", bp, dbg.source_line(*bp));
                            }
                        }
                    }
                }
            }
            "d" | "delete" => {
                match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                    Some(n) => {
                        if dbg.breakpoints.remove(&n) {
                            println!("断点已删除: 行 {}", n);
                        } else {
                            println!("行 {} 没有断点", n);
                        }
                    }
                    None => println!("用法: d <行号>"),
                }
            }
            "q" | "quit" => {
                dbg.stopped = true;
                return Err(runtime_err("[debugger] 退出"));
            }
            "h" | "help" => print_help(),
            _ => {
                println!("未知命令: {}（输入 h 查看帮助）", op);
            }
        }
    }
}

fn runtime_err(msg: &str) -> TenthError {
    TenthError::RuntimeError {
        line: None,
        col: None,
        message: msg.to_string(),
    }
}

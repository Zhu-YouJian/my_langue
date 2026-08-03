//! Tenth 性能分析器（CLI，基于 VM 路径）。
//!
//! M4.4：函数级计时（inclusive）+ 调用计数，输出 top-N 热点报告。
//!
//! 原理：VM 的 `profile_hook` 在函数入口/出口时发出 `VmProfileEvent`
//! （Enter/Exit + chunk_idx）。剖析器用**栈式 inclusive 计时**：Enter 压栈
//! （记录进入时间）、Exit 弹栈（把经过时间累积到该函数）、程序结束冲刷剩余
//! 栈——嵌套调用下各函数的 inclusive 时间精确归属（父函数时间 = 子函数时间 +
//! 自身时间），调用计数按 Enter 累加。递归（同 chunk）由帧深度变化检测覆盖。
//!
//! 用法：
//!   tenth-prof <file.th> [--top N]
//!   --top N   报告只显示前 N 个热点函数（默认 10）

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::error::{TenthError, TenthResult};
use tenth::hir::hir::HirProgram;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::{Vm, VmProfileEvent, VmProfileKind};

/// 每个函数的剖析统计。
#[derive(Debug, Default, Clone)]
struct FuncStat {
    name: String,
    calls: u64,
    total_ns: u128,
}

/// 剖析状态（hook 闭包与报告代码共享）。
struct ProfilerState {
    /// 活跃函数调用栈：(函数名, 进入时间)。Enter 压栈、Exit 弹栈。
    /// 栈式保证 inclusive 时间正确归属——Exit 后恢复调用者，不把时间误记给被调函数。
    stack: Vec<(String, Instant)>,
    /// 统计表：函数名 → 统计。
    stats: HashMap<String, FuncStat>,
}

impl ProfilerState {
    fn new() -> Self {
        ProfilerState {
            stack: Vec::new(),
            stats: HashMap::new(),
        }
    }

    /// 处理一个剖析事件。
    /// - Enter：压栈（记录进入时间），调用计数 +1；
    /// - Exit：弹栈，把 `now - 进入时间` 累积到该函数（inclusive 时间）。
    fn on_event(&mut self, vm: &Vm, event: VmProfileEvent, now: Instant) {
        match event.kind {
            VmProfileKind::Enter => {
                let name = vm
                    .chunk_name_at(event.chunk_idx)
                    .unwrap_or_else(|| format!("__chunk_{}", event.chunk_idx));
                self.stack.push((name.clone(), now));
                let s = self.stats.entry(name.clone()).or_default();
                s.calls += 1;
                if s.name.is_empty() {
                    s.name = name;
                }
            }
            VmProfileKind::Exit => {
                if let Some((name, t0)) = self.stack.pop() {
                    let dt = now.duration_since(t0).as_nanos();
                    self.stats.entry(name).or_default().total_ns += dt;
                }
                // 栈空时忽略（防御：事件不平衡时不清零、不 panic）。
            }
        }
    }

    /// 程序结束：冲刷调用栈上所有未返回的函数（含顶层入口函数）。
    fn finish(&mut self, end: Instant) {
        for (name, t0) in self.stack.drain(..) {
            let dt = end.duration_since(t0).as_nanos();
            self.stats.entry(name).or_default().total_ns += dt;
        }
    }
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
        eprintln!("用法: tenth-prof <file.th> [--top N]");
        eprintln!("  --top N   只显示前 N 个热点函数（默认 10）");
        std::process::exit(2);
    }
    let path = &args[1];
    let mut top_n: usize = 10;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--top" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        top_n = n.max(1);
                        i += 2;
                        continue;
                    }
                }
                eprintln!("--top 需要正整数参数");
                std::process::exit(2);
            }
            other => {
                eprintln!("未知参数: {}", other);
                std::process::exit(2);
            }
        }
    }

    if let Err(e) = run_prof(path, top_n) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_prof(path: &str, top_n: usize) -> TenthResult<()> {
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

    // 构建 VM（与主编译器 vm_execute 对齐；剖析走字节码解释循环，绕过 JIT）。
    let mut vm = Vm::new();
    natives::register_all_natives(&mut vm);
    let global_names: HashSet<String> = hir.globals.iter().map(|g| g.name.clone()).collect();
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new_with_globals(global_names.clone());
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
                vm.set_global(
                    func.name.clone(),
                    Value::FnRef {
                        name: func.name.clone(),
                        params: func.params.clone(),
                        return_type: func.return_type.clone(),
                        captures: vec![],
                    },
                );
            }
            Err(_) => {
                // 单函数编译失败 → VM 无法执行此程序（无副作用的编译期失败）。
                return Err(TenthError::RuntimeError {
                    line: None,
                    col: None,
                    message: format!(
                        "VM 无法编译函数 '{}'——剖析器基于 VM chunk 入口，无法剖析此程序。",
                        func.name
                    ),
                });
            }
        }
    }

    let state = Rc::new(RefCell::new(ProfilerState::new()));
    let state_hook = Rc::clone(&state);
    vm.set_profile_hook(Some(Box::new(
        move |vm: &mut Vm, event: VmProfileEvent| -> TenthResult<()> {
            state_hook.borrow_mut().on_event(vm, event, Instant::now());
            Ok(())
        },
    )));

    // 执行（含全局初始化与 main）。
    let run_start = Instant::now();
    let result = run_vm(&mut vm, &hir, &global_names);
    let run_end = Instant::now();
    state.borrow_mut().finish(run_end);

    report(&state, top_n, run_end.duration_since(run_start).as_nanos());
    result.map(|_| ())
}

/// 执行程序（全局初始化 + main / main_expr），返回执行结果（错误由调用方处理）。
fn run_vm(vm: &mut Vm, hir: &HirProgram, global_names: &HashSet<String>) -> TenthResult<Value> {
    if !hir.globals.is_empty() {
        let gcompiler = BytecodeCompiler::new();
        let (gchunk, gclosures) = match gcompiler.compile_globals(&hir.globals) {
            Ok(x) => x,
            Err(_) => {
                return Err(TenthError::RuntimeError {
                    line: None,
                    col: None,
                    message: "全局初始化编译失败（VM 无法剖析此程序）".into(),
                });
            }
        };
        vm.add_fn("__global_init".into(), gchunk);
        for (name, closure_chunk) in gclosures {
            vm.add_fn(name, closure_chunk);
        }
        vm.call("__global_init")?;
    }

    if vm.has_fn("main") {
        vm.call("main")
    } else if let Some(expr) = &hir.main_expr {
        let compiler = BytecodeCompiler::new_with_globals(global_names.clone());
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
                vm.call("main")
            }
            Err(_) => Err(TenthError::RuntimeError {
                line: None,
                col: None,
                message: "VM 无法编译 main（剖析器基于 VM chunk 入口，无法剖析此程序）".into(),
            }),
        }
    } else if hir.functions.is_empty() {
        Ok(Value::Unit)
    } else {
        Err(TenthError::RuntimeError {
            line: None,
            col: None,
            message: "VM: main 未编译（剖析器基于 VM chunk 入口，无法剖析此程序）".into(),
        })
    }
}

/// 输出热点报告。`wall_ns` 为真实墙钟运行时间（inclusive 区间相互嵌套重叠，
/// 各函数占比相对墙钟时间计算，占比和可超 100%——这是 flat profile 的正常现象）。
fn report(state: &Rc<RefCell<ProfilerState>>, top_n: usize, wall_ns: u128) {
    let st = state.borrow();
    if st.stats.is_empty() {
        println!("(无剖析数据)");
        return;
    }
    let mut entries: Vec<&FuncStat> = st.stats.values().collect();
    entries.sort_by(|a, b| b.total_ns.cmp(&a.total_ns).then(a.name.cmp(&b.name)));
    entries.truncate(top_n);

    println!("Tenth Profiler — 热点报告（前 {} / {} 个函数）", entries.len(), st.stats.len());
    println!("墙钟总耗时: {:.2} ms", wall_ns as f64 / 1e6);
    println!("{:<4} {:<28} {:>10} {:>12} {:>8}", "排名", "函数", "调用次数", "耗时(ms)", "占比");
    for (i, stat) in entries.iter().enumerate() {
        let pct = if wall_ns > 0 {
            stat.total_ns as f64 / wall_ns as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "{:<4} {:<28} {:>10} {:>12.3} {:>7.1}%",
            i + 1,
            stat.name,
            stat.calls,
            stat.total_ns as f64 / 1e6,
            pct
        );
    }
}

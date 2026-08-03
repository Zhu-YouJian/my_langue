mod commands;
mod engine;
mod manifest;
mod pkg;
mod resolver;
mod version;

use std::env;
use std::process;

fn print_help() {
    println!("Tenth Package Manager (tenthpm) v0.2.0");
    println!();
    println!("USAGE:");
    println!("    tenthpm <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    init [NAME]           Create a new Tenth project");
    println!("    build                 Compile-check all source files");
    println!("    run                   Build and run src/main.th");
    println!("    test                  Run all test files in tests/");
    println!("    add <PKG> [VER]       Add a dependency (name/path/git-url)");
    println!("    remove <PKG>          Remove a dependency");
    println!("    list                  List all dependencies");
    println!("    clean [--deps|--all]  Remove build artifacts");
    println!("    publish [--registry <dir>]  Package into .tenthpkg (optionally publish to a local registry dir)");
    println!("    install <PKG> [--registry <dir>]  Install a package (git-url/local-path/.tenthpkg/registry-name)");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help            Print this help message");
    println!("    --deps                Also clean deps/ (with clean)");
    println!("    --all                 Clean everything (with clean)");
    println!("    --registry <dir>      Local registry directory (with publish/install)");
    println!();
    println!("DEPENDENCY FORMATS:");
    println!("    tenthpm add mylib           Registry dependency (version *)");
    println!("    tenthpm add mylib 1.0.0     Registry dependency (version 1.0.0)");
    println!("    tenthpm add ./mylib         Local path dependency");
    println!("    tenthpm add https://...     Git dependency (cloned to deps/)");
    println!();
    println!("VERSION CONSTRAINTS:");
    println!("    *                        Any version");
    println!("    1.2.3                    Exact version");
    println!("    ^1.2.3                   >=1.2.3 <2.0.0 (caret)");
    println!("    >=1.2.0,<2.0.0           Range (comma = AND)");
    println!();
    println!("RESOLUTION:");
    println!("    Transitive dependencies are resolved from path/git deps' Tenth.toml.");
    println!("    Conflicts and missing deps are reported loudly (never silently).");
    println!("    The resolved graph is locked in Tenth.lock.");
}

/// 从参数中提取 `--registry <dir>` 的值。
fn registry_arg(args: &[String]) -> Option<String> {
    args.iter()
        .position(|a| a == "--registry")
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        process::exit(0);
    }

    let command = &args[1];

    let result = match command.as_str() {
        "init" => {
            let name = args.get(2).map(|s| s.as_str());
            commands::init::init(name)
        }
        "build" => commands::build::build(),
        "test" => commands::test_cmd::test(),
        "run" => commands::run::run(),
        "add" => {
            let package = args.get(2);
            if package.is_none() {
                eprintln!("Error: `add` requires a package name, path, or git URL");
                process::exit(1);
            }
            let version = args.get(3).map(|s| s.as_str());
            commands::add::add(package.unwrap(), version)
        }
        "remove" | "rm" => {
            let package = args.get(2);
            if package.is_none() {
                eprintln!("Error: `remove` requires a package name");
                process::exit(1);
            }
            commands::remove::remove(package.unwrap())
        }
        "list" | "ls" => commands::list::list(),
        "clean" => {
            let deps_too = args.iter().any(|a| a == "--deps");
            let all = args.iter().any(|a| a == "--all");
            commands::clean::clean(deps_too, all)
        }
        "publish" => {
            let registry = registry_arg(&args);
            commands::publish::publish(registry.as_deref())
        }
        "install" => {
            let package = args.get(2);
            if package.is_none() {
                eprintln!("Error: `install` requires a package name, git URL, or .tenthpkg file");
                process::exit(1);
            }
            let registry = registry_arg(&args);
            commands::install::install(package.unwrap(), registry.as_deref())
        }
        "-h" | "--help" => {
            print_help();
            process::exit(0);
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            print_help();
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

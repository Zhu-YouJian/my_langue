mod commands;
mod engine;
mod manifest;

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
    println!("    publish               Package the project into .tenthpkg");
    println!("    install <PKG>         Install a package (git-url/local-path)");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help            Print this help message");
    println!("    --deps                Also clean deps/ (with clean)");
    println!("    --all                 Clean everything (with clean)");
    println!();
    println!("DEPENDENCY FORMATS:");
    println!("    tenthpm add mylib           Registry dependency (version *)");
    println!("    tenthpm add mylib 1.0.0     Registry dependency (version 1.0.0)");
    println!("    tenthpm add ./mylib         Local path dependency");
    println!("    tenthpm add https://...     Git dependency (cloned to deps/)");
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
        "publish" => commands::publish::publish(),
        "install" => {
            let package = args.get(2);
            if package.is_none() {
                eprintln!("Error: `install` requires a package name or git URL");
                process::exit(1);
            }
            commands::install::install(package.unwrap())
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

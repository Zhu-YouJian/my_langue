mod commands;
mod manifest;

use std::env;
use std::process;

fn print_help() {
    println!("Tenth Package Manager (tenthpm) v0.1.0");
    println!();
    println!("USAGE:");
    println!("    tenthpm <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    init [NAME]       Create a new Tenth project");
    println!("    build             Build the project");
    println!("    test              Run tests");
    println!("    run               Build and run the project");
    println!("    add <PKG> [VER]   Add a dependency");
    println!("    publish           Publish the package to registry");
    println!("    install <PKG>     Install a package from registry");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help        Print this help message");
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
                eprintln!("Error: `add` requires a package name");
                process::exit(1);
            }
            let version = args.get(3).map(|s| s.as_str());
            commands::add::add(package.unwrap(), version)
        }
        "publish" => commands::publish::publish(),
        "install" => {
            let package = args.get(2);
            if package.is_none() {
                eprintln!("Error: `install` requires a package name");
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

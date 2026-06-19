use std::path::Path;

use crate::engine;
use crate::manifest::Manifest;

/// Run all test files under tests/.
///
/// Each .th file in tests/ is executed in-process. A test passes if the
/// program exits without error. Test files can use `assert(condition, message)`
/// or simply panic/error to signal failure.
pub fn test() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    let tests_dir = Path::new("tests");
    if !tests_dir.exists() {
        println!("未找到 tests/ 目录，没有可运行的测试。");
        return Ok(());
    }

    let test_files = engine::collect_th_files(tests_dir)?;

    if test_files.is_empty() {
        println!("tests/ 下没有测试文件。");
        return Ok(());
    }

    println!(
        "Running tests for `{}` v{}",
        manifest.package.name, manifest.package.version
    );
    println!();

    let mut passed = 0;
    let mut failed = 0;

    for file in &test_files {
        let rel = file.strip_prefix(".").unwrap_or(file);
        print!("  test {} ... ", rel.display());

        match engine::run_file(file) {
            Ok(()) => {
                println!("ok");
                passed += 1;
            }
            Err(e) => {
                println!("FAILED");
                eprintln!("    {}", e);
                failed += 1;
            }
        }
    }

    println!();
    if failed > 0 {
        println!("Test result: FAILED. {} passed; {} failed.", passed, failed);
        Err(format!("{} 个测试失败", failed).into())
    } else {
        println!(
            "Test result: ok. {} passed; {} failed.",
            passed, failed
        );
        Ok(())
    }
}

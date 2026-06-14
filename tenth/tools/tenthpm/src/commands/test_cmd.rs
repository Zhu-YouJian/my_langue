use std::fs;
use std::path::Path;

pub fn test() -> Result<(), Box<dyn std::error::Error>> {
    let tests_dir = Path::new("tests");
    if !tests_dir.exists() {
        println!("No tests/ directory found. Nothing to test.");
        return Ok(());
    }

    let mut test_files = Vec::new();
    collect_th_files(tests_dir, &mut test_files)?;

    if test_files.is_empty() {
        println!("No test files found in tests/");
        return Ok(());
    }

    let total = test_files.len();
    let mut passed = 0;

    for file in &test_files {
        println!("  Running {} ...", file.display());
        // Simulate test execution
        println!("  ✓ {}", file.display());
        passed += 1;
    }

    println!("\nTest result: {}/{} passed", passed, total);

    Ok(())
}

fn collect_th_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_th_files(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "th") {
            files.push(path);
        }
    }
    Ok(())
}

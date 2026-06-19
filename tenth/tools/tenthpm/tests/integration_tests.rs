use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Find the tenthpm binary path.
fn tenthpm_binary() -> PathBuf {
    // The binary is at tenth/tools/tenthpm/target/debug/tenthpm[.exe]
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir.join("target").join("debug");
    
    #[cfg(windows)]
    let bin = target_dir.join("tenthpm.exe");
    #[cfg(not(windows))]
    let bin = target_dir.join("tenthpm");
    
    if bin.exists() {
        return bin;
    }
    
    // Fallback: check workspace target
    let workspace_target = manifest_dir
        .join("..")
        .join("..")
        .join("..")
        .join("target")
        .join("debug");
    
    #[cfg(windows)]
    let bin = workspace_target.join("tenthpm.exe");
    #[cfg(not(windows))]
    let bin = workspace_target.join("tenthpm");
    
    bin
}

/// Create a temporary test directory.
fn create_test_dir(name: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let test_dir = manifest_dir.join("target").join("test-tmp").join(name);
    
    // Clean up if exists
    if test_dir.exists() {
        fs::remove_dir_all(&test_dir).ok();
    }
    
    fs::create_dir_all(&test_dir).unwrap();
    test_dir
}

/// Write a file with UTF-8 content (no BOM).
fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut file = fs::File::create(path).unwrap();
    file.write_all(content.as_bytes()).unwrap();
}

/// Run tenthpm with the given arguments in the specified directory.
fn run_tenthpm(cwd: &Path, args: &[&str]) -> (bool, String) {
    let bin = tenthpm_binary();
    let output = Command::new(&bin)
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("Failed to execute tenthpm");
    
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    
    let combined = if stdout.is_empty() {
        stderr
    } else if stderr.is_empty() {
        stdout
    } else {
        format!("{}\n{}", stdout, stderr)
    };
    
    (output.status.success(), combined)
}

// ============ init 命令测试 ============

#[test]
fn test_init_creates_project_structure() {
    let test_dir = create_test_dir("init_basic");
    
    let (success, output) = run_tenthpm(&test_dir, &["init", "myproject"]);
    assert!(success, "init failed: {}", output);
    assert!(output.contains("Created Tenth project `myproject`"));
    
    // Check directory structure
    let project_dir = test_dir.join("myproject");
    assert!(project_dir.exists(), "Project directory not created");
    assert!(project_dir.join("src").exists(), "src/ directory not created");
    assert!(project_dir.join("src/main.th").exists(), "src/main.th not created");
    assert!(project_dir.join("Tenth.toml").exists(), "Tenth.toml not created");
    assert!(project_dir.join("tests").exists(), "tests/ directory not created");
}

#[test]
fn test_init_manifest_content() {
    let test_dir = create_test_dir("init_manifest");
    
    run_tenthpm(&test_dir, &["init", "testpkg"]);
    
    let manifest_path = test_dir.join("testpkg").join("Tenth.toml");
    let content = fs::read_to_string(&manifest_path).unwrap();
    
    assert!(content.contains("name = \"testpkg\""));
    assert!(content.contains("version = \"0.1.0\""));
    assert!(content.contains("edition = \"2024\""));
}

#[test]
fn test_init_main_th_content() {
    let test_dir = create_test_dir("init_main_th");
    
    run_tenthpm(&test_dir, &["init", "hello"]);
    
    let main_path = test_dir.join("hello").join("src").join("main.th");
    let content = fs::read_to_string(&main_path).unwrap();
    
    assert!(content.contains("fn main()"));
    assert!(content.contains("println"));
    assert!(content.contains("Hello from hello!"));
}

// ============ build 命令测试 ============

#[test]
fn test_build_success() {
    let test_dir = create_test_dir("build_success");
    
    run_tenthpm(&test_dir, &["init", "buildtest"]);
    
    let project_dir = test_dir.join("buildtest");
    let (success, output) = run_tenthpm(&project_dir, &["build"]);
    
    assert!(success, "build failed: {}", output);
    assert!(output.contains("Build finished successfully"));
    
    // Check lock file was created
    assert!(project_dir.join("Tenth.lock").exists(), "Tenth.lock not created");
}

#[test]
fn test_build_no_manifest() {
    let test_dir = create_test_dir("build_no_manifest");
    
    let (success, output) = run_tenthpm(&test_dir, &["build"]);
    
    assert!(!success, "build should fail without Tenth.toml");
    assert!(output.contains("Tenth.toml"));
}

// ============ run 命令测试 ============

#[test]
fn test_run_hello_world() {
    let test_dir = create_test_dir("run_hello");
    
    run_tenthpm(&test_dir, &["init", "runtest"]);
    
    let project_dir = test_dir.join("runtest");
    let (success, output) = run_tenthpm(&project_dir, &["run"]);
    
    assert!(success, "run failed: {}", output);
    assert!(output.contains("Hello from runtest!"));
}

// ============ test 命令测试 ============

#[test]
fn test_test_command_with_tests() {
    let test_dir = create_test_dir("test_with");
    
    run_tenthpm(&test_dir, &["init", "testpkg"]);
    
    let project_dir = test_dir.join("testpkg");
    
    // Create a test file
    write_file(
        &project_dir.join("tests").join("test_basic.th"),
        "fn main() { println(\"test ok\"); }",
    );
    
    let (success, output) = run_tenthpm(&project_dir, &["test"]);
    
    assert!(success, "test failed: {}", output);
    assert!(output.contains("test ok"));
    assert!(output.contains("1 passed"));
}

#[test]
fn test_test_command_no_tests_dir() {
    let test_dir = create_test_dir("test_no_dir");
    
    run_tenthpm(&test_dir, &["init", "notest"]);
    
    let project_dir = test_dir.join("notest");
    
    // Remove tests/ directory
    fs::remove_dir_all(project_dir.join("tests")).ok();
    
    let (success, output) = run_tenthpm(&project_dir, &["test"]);
    
    assert!(success, "test should succeed with no tests dir");
    assert!(output.contains("没有可运行的测试") || output.contains("tests/"));
}

// ============ add 命令测试 ============

#[test]
fn test_add_registry_dependency() {
    let test_dir = create_test_dir("add_registry");
    
    run_tenthpm(&test_dir, &["init", "addtest"]);
    
    let project_dir = test_dir.join("addtest");
    let (success, output) = run_tenthpm(&project_dir, &["add", "mylib", "1.0.0"]);
    
    assert!(success, "add failed: {}", output);
    assert!(output.contains("Added dependency `mylib`"));
    
    // Check manifest was updated
    let manifest = fs::read_to_string(project_dir.join("Tenth.toml")).unwrap();
    assert!(manifest.contains("mylib"));
    assert!(manifest.contains("1.0.0"));
    
    // Check lock file was updated
    let lock = fs::read_to_string(project_dir.join("Tenth.lock")).unwrap();
    assert!(lock.contains("mylib"));
}

#[test]
fn test_add_registry_no_version() {
    let test_dir = create_test_dir("add_no_version");
    
    run_tenthpm(&test_dir, &["init", "addtest2"]);
    
    let project_dir = test_dir.join("addtest2");
    let (success, output) = run_tenthpm(&project_dir, &["add", "somelib"]);
    
    assert!(success, "add failed: {}", output);
    
    let manifest = fs::read_to_string(project_dir.join("Tenth.toml")).unwrap();
    assert!(manifest.contains("somelib"));
    assert!(manifest.contains("\"*\""));
}

#[test]
fn test_add_path_dependency() {
    let test_dir = create_test_dir("add_path");
    
    // Create a library project
    run_tenthpm(&test_dir, &["init", "mylib"]);
    // Create a main project
    run_tenthpm(&test_dir, &["init", "mainapp"]);
    
    let project_dir = test_dir.join("mainapp");
    let lib_path = test_dir.join("mylib");
    let lib_path_str = lib_path.to_str().unwrap();
    
    let (success, output) = run_tenthpm(&project_dir, &["add", lib_path_str]);
    
    assert!(success, "add path failed: {}", output);
    assert!(output.contains("Added dependency"));
    
    // Check manifest has path dependency
    let manifest = fs::read_to_string(project_dir.join("Tenth.toml")).unwrap();
    assert!(manifest.contains("mylib"));
    assert!(manifest.contains("path"));
}

// ============ remove 命令测试 ============

#[test]
fn test_remove_dependency() {
    let test_dir = create_test_dir("remove_dep");
    
    run_tenthpm(&test_dir, &["init", "remtest"]);
    
    let project_dir = test_dir.join("remtest");
    
    // Add a dependency first
    run_tenthpm(&project_dir, &["add", "toremove", "1.0.0"]);
    
    // Verify it was added
    let manifest = fs::read_to_string(project_dir.join("Tenth.toml")).unwrap();
    assert!(manifest.contains("toremove"));
    
    // Remove it
    let (success, output) = run_tenthpm(&project_dir, &["remove", "toremove"]);
    
    assert!(success, "remove failed: {}", output);
    assert!(output.contains("Removed dependency `toremove`"));
    
    // Verify it was removed
    let manifest = fs::read_to_string(project_dir.join("Tenth.toml")).unwrap();
    assert!(!manifest.contains("toremove"));
}

#[test]
fn test_remove_nonexistent() {
    let test_dir = create_test_dir("remove_nonexist");
    
    run_tenthpm(&test_dir, &["init", "remtest2"]);
    
    let project_dir = test_dir.join("remtest2");
    
    let (success, _) = run_tenthpm(&project_dir, &["remove", "nonexistent"]);
    
    assert!(!success, "remove should fail for nonexistent dependency");
}

// ============ list 命令测试 ============

#[test]
fn test_list_empty() {
    let test_dir = create_test_dir("list_empty");
    
    run_tenthpm(&test_dir, &["init", "listtest"]);
    
    let project_dir = test_dir.join("listtest");
    let (success, output) = run_tenthpm(&project_dir, &["list"]);
    
    assert!(success, "list failed: {}", output);
    assert!(output.contains("no dependencies"));
}

#[test]
fn test_list_with_dependencies() {
    let test_dir = create_test_dir("list_with_deps");
    
    run_tenthpm(&test_dir, &["init", "listtest2"]);
    
    let project_dir = test_dir.join("listtest2");
    run_tenthpm(&project_dir, &["add", "lib1", "1.0.0"]);
    run_tenthpm(&project_dir, &["add", "lib2", "2.0.0"]);
    
    let (success, output) = run_tenthpm(&project_dir, &["list"]);
    
    assert!(success, "list failed: {}", output);
    assert!(output.contains("lib1"));
    assert!(output.contains("1.0.0"));
    assert!(output.contains("lib2"));
    assert!(output.contains("2.0.0"));
    assert!(output.contains("Total: 2"));
}

// ============ clean 命令测试 ============

#[test]
fn test_clean_removes_lock() {
    let test_dir = create_test_dir("clean_lock");
    
    run_tenthpm(&test_dir, &["init", "cleantest"]);
    
    let project_dir = test_dir.join("cleantest");
    
    // Build to create lock file
    run_tenthpm(&project_dir, &["build"]);
    assert!(project_dir.join("Tenth.lock").exists());
    
    // Clean
    let (success, output) = run_tenthpm(&project_dir, &["clean"]);
    
    assert!(success, "clean failed: {}", output);
    assert!(!project_dir.join("Tenth.lock").exists(), "Tenth.lock should be removed");
}

#[test]
fn test_clean_nothing_to_clean() {
    let test_dir = create_test_dir("clean_nothing");
    
    run_tenthpm(&test_dir, &["init", "cleantest2"]);
    
    let project_dir = test_dir.join("cleantest2");
    
    let (success, output) = run_tenthpm(&project_dir, &["clean"]);
    
    assert!(success, "clean should succeed even with nothing to clean");
    assert!(output.contains("Nothing to clean"));
}

// ============ publish 命令测试 ============

#[test]
fn test_publish_validation_fails() {
    let test_dir = create_test_dir("publish_fail");
    
    run_tenthpm(&test_dir, &["init", "pubtest"]);
    
    let project_dir = test_dir.join("pubtest");
    
    // Publish without description/license should fail
    let (success, output) = run_tenthpm(&project_dir, &["publish"]);
    
    assert!(!success, "publish should fail without description/license");
    assert!(output.contains("description") || output.contains("license"));
}

#[test]
fn test_publish_success() {
    let test_dir = create_test_dir("publish_success");
    
    run_tenthpm(&test_dir, &["init", "pubtest2"]);
    
    let project_dir = test_dir.join("pubtest2");
    
    // Add description and license to manifest
    let manifest = r#"[package]
name = "pubtest2"
version = "0.1.0"
edition = "2024"
authors = []
description = "A test package"
license = "MIT"

[dependencies]
"#;
    write_file(&project_dir.join("Tenth.toml"), manifest);
    
    let (success, output) = run_tenthpm(&project_dir, &["publish"]);
    
    assert!(success, "publish failed: {}", output);
    assert!(output.contains("Published"));
    
    // Check .tenthpkg file was created
    let pkg_file = project_dir.join("pubtest2-0.1.0.tenthpkg");
    assert!(pkg_file.exists(), ".tenthpkg file not created");
}

// ============ manifest 测试 ============

#[test]
fn test_manifest_validation() {
    use tenthpm::manifest::Manifest;
    
    let mut m = Manifest::new("testpkg");
    assert!(m.validate(false).is_ok());
    
    m.package.name = "".to_string();
    assert!(m.validate(false).is_err());
    
    m.package.name = "testpkg".to_string();
    m.package.version = "invalid".to_string();
    assert!(m.validate(false).is_err());
    
    m.package.version = "1.0.0".to_string();
    assert!(m.validate(false).is_ok());
    assert!(m.validate(true).is_err()); // No description/license
}

#[test]
fn test_manifest_serialize_deserialize() {
    use tenthpm::manifest::Manifest;
    
    let m = Manifest::new("serde_test");
    let toml_str = toml::to_string(&m).unwrap();
    let m2: Manifest = toml::from_str(&toml_str).unwrap();
    
    assert_eq!(m.package.name, m2.package.name);
    assert_eq!(m.package.version, m2.package.version);
}

// ============ lockfile 测试 ============

#[test]
fn test_lockfile_from_manifest() {
    use tenthpm::manifest::{Dependency, Lockfile, Manifest};
    use std::collections::HashMap;
    
    let mut deps = HashMap::new();
    deps.insert("lib1".to_string(), Dependency {
        version: "1.0.0".to_string(),
        path: None,
        git: None,
    });
    deps.insert("lib2".to_string(), Dependency {
        version: "2.0.0".to_string(),
        path: None,
        git: Some("https://github.com/example/lib2".to_string()),
    });
    
    let manifest = Manifest {
        package: tenthpm::manifest::PackageInfo {
            name: "test".to_string(),
            version: "0.1.0".to_string(),
            edition: "2024".to_string(),
            authors: vec![],
            description: None,
            license: None,
        },
        dependencies: deps,
    };
    
    let lockfile = Lockfile::from_manifest(&manifest);
    assert_eq!(lockfile.packages.len(), 2);
    assert_eq!(lockfile.version, 1);
}

// ============ 帮助命令测试 ============

#[test]
fn test_help_output() {
    let test_dir = create_test_dir("help");
    
    let (success, output) = run_tenthpm(&test_dir, &["--help"]);
    
    assert!(success);
    assert!(output.contains("Tenth Package Manager"));
    assert!(output.contains("USAGE"));
    assert!(output.contains("COMMANDS"));
    assert!(output.contains("init"));
    assert!(output.contains("build"));
    assert!(output.contains("run"));
    assert!(output.contains("test"));
    assert!(output.contains("add"));
    assert!(output.contains("remove"));
    assert!(output.contains("list"));
    assert!(output.contains("clean"));
    assert!(output.contains("publish"));
    assert!(output.contains("install"));
}

#[test]
fn test_no_args_shows_help() {
    let test_dir = create_test_dir("no_args");
    
    let (success, output) = run_tenthpm(&test_dir, &[]);
    
    assert!(success);
    assert!(output.contains("Tenth Package Manager"));
}

#[test]
fn test_unknown_command() {
    let test_dir = create_test_dir("unknown_cmd");
    
    let (success, _) = run_tenthpm(&test_dir, &["nonexistent"]);
    
    assert!(!success);
}

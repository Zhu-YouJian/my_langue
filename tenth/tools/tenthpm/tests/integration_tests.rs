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

// ============ 安全校验函数单元测试 ============
//
// 以下测试覆盖 AUDIT C-1 修复引入的安全校验函数（manifest.rs）。
// 这些函数是防御路径穿越、命令注入、敏感目录删除等攻击的关键代码，
// 必须有测试守护。`is_valid_version`/`fnv1a_64` 为私有函数，通过
// `Manifest::validate`/`Lockfile::from_manifest` 间接测试（见下方）。

mod safety_tests {
    use super::*;
    use tenthpm::manifest::*;
    use std::sync::Mutex;

    // 环境变量测试需要串行运行以避免竞态。
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    // 在不同 edition 下 set_var/remove_var 的 unsafe 性不同，统一包装。
    fn set_env_safe(key: &str, value: &str) {
        #[allow(unused_unsafe)]
        unsafe { std::env::set_var(key, value); }
    }
    fn remove_env_safe(key: &str) {
        #[allow(unused_unsafe)]
        unsafe { std::env::remove_var(key); }
    }

    // ---- validate_package_name ----

    #[test]
    fn test_validate_package_name_valid() {
        assert!(validate_package_name("my_lib").is_ok());
        assert!(validate_package_name("my-lib").is_ok());
        assert!(validate_package_name("MyLib123").is_ok());
        assert!(validate_package_name("lib.v2").is_ok());
        assert!(validate_package_name("a").is_ok());
    }

    #[test]
    fn test_validate_package_name_empty() {
        assert!(validate_package_name("").is_err());
    }

    #[test]
    fn test_validate_package_name_dot_and_dotdot() {
        assert!(validate_package_name(".").is_err());
        assert!(validate_package_name("..").is_err());
    }

    #[test]
    fn test_validate_package_name_path_separators() {
        assert!(validate_package_name("a/b").is_err());
        assert!(validate_package_name("a\\b").is_err());
        assert!(validate_package_name("/abs").is_err());
        assert!(validate_package_name("trailing/").is_err());
    }

    #[test]
    fn test_validate_package_name_windows_reserved() {
        // 大小写不敏感
        for name in ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1", "con", "Aux", "com9", "lpt2"] {
            assert!(
                validate_package_name(name).is_err(),
                "应拒绝 Windows 保留名: {}",
                name
            );
        }
    }

    #[test]
    fn test_validate_package_name_control_and_whitespace() {
        assert!(validate_package_name("a\tb").is_err());
        assert!(validate_package_name("a\nb").is_err());
        assert!(validate_package_name("a b").is_err());
        assert!(validate_package_name("a\0b").is_err());
        assert!(validate_package_name("a\u{007f}b").is_err());
    }

    #[test]
    fn test_validate_package_name_leading_and_trailing_dot() {
        assert!(validate_package_name(".hidden").is_err());
        assert!(validate_package_name("trailing.").is_err());
        assert!(validate_package_name("..sneaky").is_err());
    }

    // ---- is_git_url ----

    #[test]
    fn test_is_git_url_default_protocols() {
        let _guard = ENV_LOCK.lock().unwrap();
        remove_env_safe("TENTH_ALLOW_INSECURE_GIT");
        // 安全默认：仅 https://
        assert!(is_git_url("https://github.com/x/y"));
        assert!(!is_git_url("http://github.com/x/y"));
        assert!(!is_git_url("git://github.com/x/y"));
        assert!(!is_git_url("ssh://git@github.com/x/y"));
        assert!(!is_git_url("not a url"));
        assert!(!is_git_url("ftp://example.com/x"));
        assert!(!is_git_url(""));
    }

    #[test]
    fn test_is_git_url_insecure_mode() {
        let _guard = ENV_LOCK.lock().unwrap();
        set_env_safe("TENTH_ALLOW_INSECURE_GIT", "1");
        // opt-in 后放行所有 git 协议 + .git 后缀
        assert!(is_git_url("https://github.com/x/y"));
        assert!(is_git_url("http://github.com/x/y"));
        assert!(is_git_url("git://github.com/x/y"));
        assert!(is_git_url("ssh://git@github.com/x/y"));
        assert!(is_git_url("https://github.com/x/y.git"));
        // 仍拒绝非 URL
        assert!(!is_git_url("not a url"));
        remove_env_safe("TENTH_ALLOW_INSECURE_GIT");
    }

    // ---- extract_package_name ----

    #[test]
    fn test_extract_package_name_various() {
        assert_eq!(
            extract_package_name("https://github.com/user/mylib"),
            Some("mylib".to_string())
        );
        assert_eq!(
            extract_package_name("https://github.com/user/mylib.git"),
            Some("mylib".to_string())
        );
        assert_eq!(
            extract_package_name("https://github.com/user/mylib?rev=abc"),
            Some("mylib".to_string())
        );
        assert_eq!(
            extract_package_name("https://github.com/user/mylib#section"),
            Some("mylib".to_string())
        );
        // 无 '/' 的字符串：返回原字符串（调用方须再用 validate_package_name 校验）
        assert_eq!(extract_package_name("not a url"), Some("not a url".to_string()));
        // 末尾空段
        assert_eq!(extract_package_name("https://"), None);
        // 危险输入：query 内的 ".." 应被剥离，得到 "y"
        assert_eq!(extract_package_name("https://x/y?.."), Some("y".to_string()));
    }

    // ---- safe_package_name_from_git ----

    #[test]
    fn test_safe_package_name_from_git_valid() {
        assert_eq!(
            safe_package_name_from_git("https://github.com/user/mylib").unwrap(),
            "mylib"
        );
        assert_eq!(
            safe_package_name_from_git("https://github.com/user/mylib.git").unwrap(),
            "mylib"
        );
    }

    #[test]
    fn test_safe_package_name_from_git_path_traversal() {
        // 经典路径穿越攻击：URL 末段是 ".."
        assert!(safe_package_name_from_git("https://attacker.invalid/..").is_err());
        // 多段路径：末段是 ".." 也应被拒绝
        assert!(safe_package_name_from_git("https://attacker.invalid/foo/..").is_err());
        // 末段是 "." 应被拒绝
        assert!(safe_package_name_from_git("https://attacker.invalid/.").is_err());
        // 默认拒绝非 https 协议
        assert!(safe_package_name_from_git("http://github.com/x/y").is_err());
        assert!(safe_package_name_from_git("git://github.com/x/y").is_err());
        // Windows 保留名作为包名
        assert!(safe_package_name_from_git("https://github.com/x/CON").is_err());
        // 末段以 "." 开头（隐藏目录）应被拒绝
        assert!(safe_package_name_from_git("https://github.com/x/.hidden").is_err());
        // 末段含空格应被拒绝
        assert!(safe_package_name_from_git("https://github.com/x/a b").is_err());
        // 注意：合法的多段 URL（末段是普通名）应通过
        assert!(safe_package_name_from_git("https://github.com/x/a/b").is_ok());
    }

    // ---- safe_to_remove_dir ----

    #[test]
    fn test_safe_to_remove_dir_rejects_dangerous() {
        // 根目录（字符串形式直接拦截）
        assert!(safe_to_remove_dir(Path::new("/")).is_err());
        assert!(safe_to_remove_dir(Path::new("\\")).is_err());
        assert!(safe_to_remove_dir(Path::new("C:\\")).is_err());
        assert!(safe_to_remove_dir(Path::new("C:/")).is_err());
        // 空路径
        assert!(safe_to_remove_dir(Path::new("")).is_err());
    }

    #[test]
    fn test_safe_to_remove_dir_rejects_sensitive_subdirs() {
        // 在临时目录下创建 .ssh / .aws / .config 子目录，模拟敏感目录
        let tmp = create_test_dir("safety_remove_sensitive");
        for sub in [".ssh", ".aws", ".config"] {
            let dir = tmp.join(sub);
            fs::create_dir_all(&dir).unwrap();
            assert!(
                safe_to_remove_dir(&dir).is_err(),
                "应拒绝删除敏感目录: {}",
                dir.display()
            );
        }
    }

    #[test]
    fn test_safe_to_remove_dir_rejects_home() {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_default();
        if !home.is_empty() && Path::new(&home).exists() {
            assert!(
                safe_to_remove_dir(Path::new(&home)).is_err(),
                "应拒绝删除用户主目录"
            );
        }
    }

    #[test]
    fn test_safe_to_remove_dir_accepts_normal() {
        let tmp = create_test_dir("safety_remove_normal");
        // 普通临时目录应可删除
        assert!(safe_to_remove_dir(&tmp).is_ok());
        // 子目录也应可删除
        let sub = tmp.join("subdir");
        fs::create_dir_all(&sub).unwrap();
        assert!(safe_to_remove_dir(&sub).is_ok());
    }

    // ---- ensure_within ----

    #[test]
    fn test_ensure_within_inside() {
        let root = create_test_dir("safety_within_root");
        // 已存在的子目录
        let target = root.join("subdir");
        fs::create_dir_all(&target).unwrap();
        assert!(ensure_within(&root, &target).is_ok());
        // 尚不存在的子目录（首次 clone 场景）
        let target_new = root.join("newdep");
        assert!(ensure_within(&root, &target_new).is_ok());
    }

    #[test]
    fn test_ensure_within_outside() {
        let root1 = create_test_dir("safety_within_root1");
        let root2 = create_test_dir("safety_within_root2");
        // root2 中的文件不在 root1 内
        let outside = root2.join("file.txt");
        write_file(&outside, "x");
        assert!(ensure_within(&root1, &outside).is_err());
    }

    #[test]
    fn test_ensure_within_traversal() {
        let root = create_test_dir("safety_within_traversal");
        // 路径穿越：root/../escape 应被拒绝
        // 创建 escape 文件以便 canonicalize 能成功解析
        let parent = root.parent().unwrap();
        let escape = parent.join("escape_file_for_test");
        write_file(&escape, "x");
        let traversal_path = root.join("..").join("escape_file_for_test");
        assert!(ensure_within(&root, &traversal_path).is_err());
        // 清理
        fs::remove_file(&escape).ok();
    }
}

// ============ Manifest/Lockfile 单元测试补充 ============

#[test]
fn test_manifest_validate_is_valid_version() {
    use tenthpm::manifest::Manifest;

    // is_valid_version 为私有函数，通过 Manifest::validate 间接测试。
    // 合法 X.Y.Z 格式
    let mut m = Manifest::new("ver_test");
    m.package.version = "1.0.0".to_string();
    assert!(m.validate(false).is_ok());
    m.package.version = "0.0.0".to_string();
    assert!(m.validate(false).is_ok());
    m.package.version = "10.20.30".to_string();
    assert!(m.validate(false).is_ok());

    // 非法格式
    m.package.version = "1.0".to_string();
    assert!(m.validate(false).is_err());
    m.package.version = "1".to_string();
    assert!(m.validate(false).is_err());
    m.package.version = "1.2.3.4".to_string();
    assert!(m.validate(false).is_err());
    m.package.version = "a.b.c".to_string();
    assert!(m.validate(false).is_err());
    m.package.version = "1.0.0-alpha".to_string();
    assert!(m.validate(false).is_err());
}

#[test]
fn test_lockfile_from_manifest_with_path_dep() {
    use tenthpm::manifest::{Dependency, Lockfile, Manifest, PackageInfo};
    use std::collections::HashMap;

    // 创建临时依赖目录，含 Tenth.toml（用于 checksum 计算）
    let dep_dir = create_test_dir("lockfile_path_dep/lib_x");
    let dep_manifest = "[package]\nname = \"lib_x\"\nversion = \"1.2.3\"\nedition = \"2024\"\nauthors = []\n[dependencies]\n";
    write_file(&dep_dir.join("Tenth.toml"), dep_manifest);

    let mut deps = HashMap::new();
    deps.insert(
        "lib_x".to_string(),
        Dependency {
            version: "1.2.3".to_string(),
            path: Some(dep_dir.to_string_lossy().to_string()),
            git: None,
        },
    );

    let manifest = Manifest {
        package: PackageInfo {
            name: "main".to_string(),
            version: "0.1.0".to_string(),
            edition: "2024".to_string(),
            authors: vec![],
            description: None,
            license: None,
        },
        dependencies: deps,
    };

    let lockfile = Lockfile::from_manifest(&manifest);
    assert_eq!(lockfile.packages.len(), 1);
    assert_eq!(lockfile.version, 1);

    let pkg = &lockfile.packages[0];
    assert_eq!(pkg.name, "lib_x");
    assert_eq!(pkg.version, "1.2.3");
    // path 依赖应计算 checksum（fnv1a_64，私有函数，通过此字段间接验证）
    assert!(pkg.checksum.is_some(), "path 依赖应有 checksum");
    let checksum = pkg.checksum.as_ref().unwrap();
    assert_eq!(checksum.len(), 16, "fnv1a_64 应返回 16 位十六进制");
    assert!(
        checksum.chars().all(|c| c.is_ascii_hexdigit()),
        "checksum 应为十六进制字符串"
    );
    // source 应为 "path:..."
    assert!(pkg.source.as_ref().unwrap().starts_with("path:"));
}

#[test]
fn test_lockfile_from_manifest_empty() {
    use tenthpm::manifest::{Lockfile, Manifest};

    let manifest = Manifest::new("empty_proj");
    let lockfile = Lockfile::from_manifest(&manifest);
    assert!(lockfile.packages.is_empty());
    assert_eq!(lockfile.version, 1);
}

#[test]
fn test_manifest_save_load_roundtrip() {
    use tenthpm::manifest::{Dependency, Manifest, PackageInfo};
    use std::collections::HashMap;

    let tmp = create_test_dir("manifest_roundtrip");
    let manifest_path = tmp.join("Tenth.toml");

    let mut deps = HashMap::new();
    deps.insert(
        "dep1".to_string(),
        Dependency {
            version: "1.0.0".to_string(),
            path: None,
            git: Some("https://github.com/x/dep1".to_string()),
        },
    );
    deps.insert(
        "dep2".to_string(),
        Dependency {
            version: "2.0.0".to_string(),
            path: Some("./local/dep2".to_string()),
            git: None,
        },
    );

    let original = Manifest {
        package: PackageInfo {
            name: "roundtrip".to_string(),
            version: "0.2.1".to_string(),
            edition: "2024".to_string(),
            authors: vec!["tester".to_string()],
            description: Some("test desc".to_string()),
            license: Some("MIT".to_string()),
        },
        dependencies: deps,
    };

    original.save_to_file(&manifest_path).unwrap();
    let loaded = Manifest::load_from_file(&manifest_path).unwrap();

    assert_eq!(loaded.package.name, "roundtrip");
    assert_eq!(loaded.package.version, "0.2.1");
    assert_eq!(loaded.package.edition, "2024");
    assert_eq!(loaded.package.authors, vec!["tester".to_string()]);
    assert_eq!(loaded.package.description, Some("test desc".to_string()));
    assert_eq!(loaded.package.license, Some("MIT".to_string()));
    assert_eq!(loaded.dependencies.len(), 2);

    let dep1 = loaded.dependencies.get("dep1").unwrap();
    assert_eq!(dep1.version, "1.0.0");
    assert_eq!(dep1.git.as_deref(), Some("https://github.com/x/dep1"));
    assert!(dep1.path.is_none());

    let dep2 = loaded.dependencies.get("dep2").unwrap();
    assert_eq!(dep2.version, "2.0.0");
    assert_eq!(dep2.path.as_deref(), Some("./local/dep2"));
    assert!(dep2.git.is_none());
}

// ============ 未覆盖命令的集成测试 ============

#[test]
fn test_remove_git_dependency_offline() {
    // 离线模拟 git 依赖删除：手动构造 deps/libname/ 与 Tenth.toml 中的 git 依赖，
    // 不实际执行 git clone。
    let test_dir = create_test_dir("remove_git_dep");
    run_tenthpm(&test_dir, &["init", "remgit"]);
    let project_dir = test_dir.join("remgit");

    // 模拟 git clone 结果：在 deps/clonedlib/ 下放置文件
    let deps_dir = project_dir.join("deps");
    let cloned_dir = deps_dir.join("clonedlib");
    fs::create_dir_all(cloned_dir.join("src")).unwrap();
    write_file(
        &cloned_dir.join("Tenth.toml"),
        "[package]\nname = \"clonedlib\"\nversion = \"0.1.0\"\nedition = \"2024\"\nauthors = []\n[dependencies]\n",
    );
    write_file(&cloned_dir.join("src").join("main.th"), "fn main() {}\n");

    // 在 Tenth.toml 中添加 git 依赖条目
    let manifest = r#"[package]
name = "remgit"
version = "0.1.0"
edition = "2024"
authors = []

[dependencies]
clonedlib = { version = "*", git = "https://github.com/example/clonedlib" }
"#;
    write_file(&project_dir.join("Tenth.toml"), manifest);

    // 创建 lockfile
    let lock = "version = 1\n[[packages]]\nname = \"clonedlib\"\nversion = \"*\"\nsource = \"git:https://github.com/example/clonedlib\"\n";
    write_file(&project_dir.join("Tenth.lock"), lock);

    assert!(cloned_dir.exists(), "前置条件：deps/clonedlib 应存在");

    let (success, output) = run_tenthpm(&project_dir, &["remove", "clonedlib"]);
    assert!(success, "remove git dep failed: {}", output);
    assert!(output.contains("Removed dependency `clonedlib`"));
    assert!(output.contains("Removed `deps/clonedlib`"));

    // 验证 deps/clonedlib/ 已被删除
    assert!(
        !cloned_dir.exists(),
        "remove 应删除 git 依赖的 deps/clonedlib 目录"
    );

    // 验证 manifest 中已无 clonedlib
    let manifest_after = fs::read_to_string(project_dir.join("Tenth.toml")).unwrap();
    assert!(!manifest_after.contains("clonedlib"));
}

#[test]
fn test_clean_deps_flag() {
    let test_dir = create_test_dir("clean_deps");
    run_tenthpm(&test_dir, &["init", "cleandeps"]);
    let project_dir = test_dir.join("cleandeps");

    // 创建 deps/ 目录和文件
    let deps_dir = project_dir.join("deps");
    fs::create_dir_all(deps_dir.join("somepkg")).unwrap();
    write_file(
        &deps_dir.join("somepkg").join("Tenth.toml"),
        "[package]\nname = \"somepkg\"\nversion = \"1.0.0\"\nedition = \"2024\"\nauthors = []\n[dependencies]\n",
    );

    // 创建 lock 文件
    write_file(&project_dir.join("Tenth.lock"), "version = 1\npackages = []\n");

    assert!(deps_dir.exists());
    assert!(project_dir.join("Tenth.lock").exists());

    let (success, output) = run_tenthpm(&project_dir, &["clean", "--deps"]);
    assert!(success, "clean --deps failed: {}", output);
    assert!(output.contains("Removed Tenth.lock"));
    assert!(output.contains("Removed deps/"));

    assert!(!project_dir.join("Tenth.lock").exists());
    assert!(!deps_dir.exists(), "deps/ 应被 clean --deps 删除");
}

#[test]
fn test_install_local_path() {
    let test_dir = create_test_dir("install_local");
    // 创建本地库（被安装方）
    run_tenthpm(&test_dir, &["init", "locallib"]);
    // 创建主项目（安装方）
    run_tenthpm(&test_dir, &["init", "mainproj"]);

    let lib_path = test_dir.join("locallib");
    let project_dir = test_dir.join("mainproj");

    let lib_path_str = lib_path.to_str().unwrap();
    let (success, output) = run_tenthpm(&project_dir, &["install", lib_path_str]);
    assert!(success, "install local path failed: {}", output);
    assert!(output.contains("Installed"));

    // 验证文件被复制到 deps/locallib/
    let installed_dir = project_dir.join("deps").join("locallib");
    assert!(installed_dir.exists(), "deps/locallib 应存在");
    assert!(installed_dir.join("Tenth.toml").exists());
    assert!(installed_dir.join("src").join("main.th").exists());

    // 验证 manifest 中有 path 依赖
    let manifest = fs::read_to_string(project_dir.join("Tenth.toml")).unwrap();
    assert!(manifest.contains("locallib"));
    assert!(manifest.contains("path"));

    // 验证 lockfile 有 locallib 条目
    let lock = fs::read_to_string(project_dir.join("Tenth.lock")).unwrap();
    assert!(lock.contains("locallib"));
}

#[test]
fn test_install_rejects_path_traversal_url() {
    // 防御性测试：install 应拒绝路径穿越攻击 URL
    // （不应实际执行 git clone，因为 safe_package_name_from_git 会先拦截）
    let test_dir = create_test_dir("install_traversal");
    run_tenthpm(&test_dir, &["init", "travroot"]);
    let project_dir = test_dir.join("travroot");

    // 记录安装前的父目录条目数，用于检测是否有意外文件被创建
    let parent = project_dir.parent().unwrap();
    let before_count = fs::read_dir(parent).map(|d| d.count()).unwrap_or(0);

    let (success, output) =
        run_tenthpm(&project_dir, &["install", "https://attacker.invalid/.."]);

    assert!(
        !success,
        "install 应拒绝路径穿越 URL，但成功了。output: {}",
        output
    );
    // 验证 deps/ 目录未被创建（install 在 safe_package_name_from_git 阶段就失败）
    assert!(
        !project_dir.join("deps").exists(),
        "deps/ 不应被创建（install 应在包名校验阶段失败）"
    );
    // 验证父目录条目数未增加（无逃逸文件）
    let after_count = fs::read_dir(parent).map(|d| d.count()).unwrap_or(0);
    assert_eq!(
        before_count, after_count,
        "父目录条目数不应变化（无逃逸文件被创建）"
    );
}

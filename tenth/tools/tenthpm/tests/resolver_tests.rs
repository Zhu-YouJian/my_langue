//! M4.1 tenthpm 完整：依赖解析（传递依赖/版本冲突/循环检测/锁文件）+ 发布流程
//! （打包/发布到本地 registry/从 registry 与 .tenthpkg 安装）集成测试。
//!
//! 护城河红线验证：依赖冲突 / 缺失必须响亮报错（命令失败 + 错误消息），
//! 绝不静默选择错误版本或静默忽略。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ============ 工具（与 integration_tests.rs 一致，独立测试 crate 需重复） ============

fn tenthpm_binary() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target_dir = manifest_dir.join("target").join("debug");

    #[cfg(windows)]
    let bin = target_dir.join("tenthpm.exe");
    #[cfg(not(windows))]
    let bin = target_dir.join("tenthpm");

    if bin.exists() {
        return bin;
    }

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

fn create_test_dir(name: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let test_dir = manifest_dir.join("target").join("test-tmp").join(name);
    if test_dir.exists() {
        fs::remove_dir_all(&test_dir).ok();
    }
    fs::create_dir_all(&test_dir).unwrap();
    test_dir
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(path, content).unwrap();
}

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

/// 生成 package 部分无依赖的 manifest。
fn pkg_toml(name: &str, version: &str) -> String {
    format!(
        "[package]\nname = \"{}\"\nversion = \"{}\"\nedition = \"2024\"\nauthors = []\n[dependencies]\n",
        name, version
    )
}

/// 生成带依赖的 manifest；path 依赖用内联表 `{ version, path }`。
fn pkg_toml_with_deps(
    name: &str,
    version: &str,
    deps: &[(&str, &str, Option<&str>)],
) -> String {
    let mut s = format!(
        "[package]\nname = \"{}\"\nversion = \"{}\"\nedition = \"2024\"\nauthors = []\n[dependencies]\n",
        name, version
    );
    for (n, v, p) in deps {
        match p {
            Some(path) => s.push_str(&format!(
                "{} = {{ version = \"{}\", path = \"{}\" }}\n",
                n, v, path
            )),
            // Dependency 是结构体，registry 依赖也必须写内联表
            None => s.push_str(&format!("{} = {{ version = \"{}\" }}\n", n, v)),
        }
    }
    s
}

/// 可发布的 manifest（含 description/license）。
fn publishable_toml(name: &str, version: &str) -> String {
    format!(
        "[package]\nname = \"{}\"\nversion = \"{}\"\nedition = \"2024\"\nauthors = []\ndescription = \"test package\"\nlicense = \"MIT\"\n[dependencies]\n",
        name, version
    )
}

// ============ 传递依赖解析 + 锁文件 ============

#[test]
fn test_build_resolves_transitive_and_locks() {
    let test_dir = create_test_dir("m4_transitive");

    // lib_c ← lib_b ← app（全部 path 依赖，相对项目根）
    write_file(
        &test_dir.join("lib_c").join("Tenth.toml"),
        &pkg_toml("lib_c", "0.3.0"),
    );
    write_file(
        &test_dir.join("lib_c").join("src").join("lib_c.th"),
        "fn lib_c_hello() -> str { \"c\" }\n",
    );
    write_file(
        &test_dir.join("lib_b").join("Tenth.toml"),
        &pkg_toml_with_deps("lib_b", "0.2.0", &[("lib_c", "*", Some("../lib_c"))]),
    );
    write_file(
        &test_dir.join("lib_b").join("src").join("lib_b.th"),
        "fn lib_b_hello() -> str { \"b\" }\n",
    );
    let app = test_dir.join("app");
    write_file(
        &app.join("Tenth.toml"),
        &pkg_toml_with_deps("app", "0.1.0", &[("lib_b", "*", Some("../lib_b"))]),
    );
    write_file(&app.join("src").join("main.th"), "fn main() { println(\"app ok\"); }\n");

    let (success, output) = run_tenthpm(&app, &["build"]);
    assert!(success, "build 失败: {}", output);
    assert!(output.contains("lib_b"), "应解析直接依赖 lib_b: {}", output);
    assert!(output.contains("lib_c"), "应解析传递依赖 lib_c: {}", output);

    let lock = fs::read_to_string(app.join("Tenth.lock")).unwrap();
    assert!(lock.contains("lib_b"), "锁文件应含 lib_b: {}", lock);
    assert!(lock.contains("lib_c"), "锁文件应含传递依赖 lib_c: {}", lock);
    assert!(
        lock.contains("dependencies"),
        "锁文件应记录依赖列表: {}",
        lock
    );
}

#[test]
fn test_lockfile_records_transitive_dependencies() {
    // 直接验证锁文件里 lib_b 的 dependencies 字段含 lib_c
    let test_dir = create_test_dir("m4_lock_trans");
    write_file(
        &test_dir.join("lib_c").join("Tenth.toml"),
        &pkg_toml("lib_c", "0.3.0"),
    );
    write_file(
        &test_dir.join("lib_b").join("Tenth.toml"),
        &pkg_toml_with_deps("lib_b", "0.2.0", &[("lib_c", "*", Some("../lib_c"))]),
    );
    write_file(
        &test_dir.join("app").join("Tenth.toml"),
        &pkg_toml_with_deps("app", "0.1.0", &[("lib_b", "*", Some("../lib_b"))]),
    );

    // 直接调用 resolver API（lib 测试）
    use tenthpm::manifest::Manifest;
    use tenthpm::resolver;

    let app = test_dir.join("app");
    let m = Manifest::load_from_file(&app.join("Tenth.toml")).unwrap();
    // path 依赖相对项目根（app 目录）解析
    let res = resolver::resolve(&m, &app).unwrap();
    let lib_b = res.packages.iter().find(|p| p.name == "lib_b").unwrap();
    assert_eq!(lib_b.dependencies, vec!["lib_c".to_string()]);

    let lock = tenthpm::manifest::Lockfile::from_resolution(&res);
    let lock_b = lock.packages.iter().find(|p| p.name == "lib_b").unwrap();
    assert_eq!(lock_b.dependencies, vec!["lib_c".to_string()]);
}

// ============ 版本冲突响亮报错 ============

#[test]
fn test_build_conflict_reported_loudly() {
    let test_dir = create_test_dir("m4_conflict");

    // lib 有两个版本：libv1=1.0.0，libv2=2.0.0
    write_file(&test_dir.join("libv1").join("Tenth.toml"), &pkg_toml("lib", "1.0.0"));
    write_file(&test_dir.join("libv1").join("src").join("lib.th"), "fn f() {}\n");
    write_file(&test_dir.join("libv2").join("Tenth.toml"), &pkg_toml("lib", "2.0.0"));
    write_file(&test_dir.join("libv2").join("src").join("lib.th"), "fn f() {}\n");

    // app_a 需要 lib ^1.0.0（指向 libv1），app_b 需要 lib ^2.0.0（指向 libv2）
    write_file(
        &test_dir.join("app_a").join("Tenth.toml"),
        &pkg_toml_with_deps("app_a", "0.1.0", &[("lib", "^1.0.0", Some("../libv1"))]),
    );
    write_file(&test_dir.join("app_a").join("src").join("a.th"), "fn a() {}\n");
    write_file(
        &test_dir.join("app_b").join("Tenth.toml"),
        &pkg_toml_with_deps("app_b", "0.1.0", &[("lib", "^2.0.0", Some("../libv2"))]),
    );
    write_file(&test_dir.join("app_b").join("src").join("b.th"), "fn b() {}\n");

    let root = test_dir.join("root");
    write_file(
        &root.join("Tenth.toml"),
        &pkg_toml_with_deps("root", "0.1.0", &[
            ("app_a", "*", Some("../app_a")),
            ("app_b", "*", Some("../app_b")),
        ]),
    );
    write_file(&root.join("src").join("main.th"), "fn main() {}\n");

    let (success, output) = run_tenthpm(&root, &["build"]);
    assert!(!success, "版本冲突应导致 build 失败");
    assert!(
        output.contains("版本冲突"),
        "应响亮报版本冲突: {}",
        output
    );
    assert!(output.contains("lib"), "冲突消息应提及包名 lib: {}", output);
    assert!(
        output.contains("app_a") && output.contains("app_b"),
        "冲突消息应包含依赖链: {}",
        output
    );
}

#[test]
fn test_resolver_conflict_direct_api() {
    // 直接 API 验证：registry-only 互斥约束也应报错
    let test_dir = create_test_dir("m4_registry_conflict");
    write_file(
        &test_dir.join("Tenth.toml"),
        &pkg_toml_with_deps("root", "0.1.0", &[("lib", ">=2.0.0,<1.0.0", None)]),
    );

    use tenthpm::manifest::Manifest;
    use tenthpm::resolver;

    let m = Manifest::load_from_file(&test_dir.join("Tenth.toml")).unwrap();
    let err = resolver::resolve(&m, &test_dir).unwrap_err();
    assert!(err.contains("版本冲突"), "registry-only 冲突应报错: {}", err);
}

// ============ 循环依赖检测 ============

#[test]
fn test_build_cycle_detected_loudly() {
    let test_dir = create_test_dir("m4_cycle");

    write_file(
        &test_dir.join("a").join("Tenth.toml"),
        &pkg_toml_with_deps("a", "0.1.0", &[("b", "*", Some("../b"))]),
    );
    write_file(&test_dir.join("a").join("src").join("main.th"), "fn main() {}\n");
    write_file(
        &test_dir.join("b").join("Tenth.toml"),
        &pkg_toml_with_deps("b", "0.1.0", &[("a", "*", Some("../a"))]),
    );
    write_file(&test_dir.join("b").join("src").join("main.th"), "fn main() {}\n");

    let root = test_dir.join("root");
    write_file(
        &root.join("Tenth.toml"),
        &pkg_toml_with_deps("root", "0.1.0", &[("a", "*", Some("../a"))]),
    );
    write_file(&root.join("src").join("main.th"), "fn main() {}\n");

    let (success, output) = run_tenthpm(&root, &["build"]);
    assert!(!success, "循环依赖应导致 build 失败");
    assert!(
        output.contains("循环依赖"),
        "应响亮报循环依赖: {}",
        output
    );
}

// ============ 缺失依赖响亮报错 ============

#[test]
fn test_build_missing_path_dep_loud() {
    let test_dir = create_test_dir("m4_missing");
    let root = test_dir.join("root");
    write_file(
        &root.join("Tenth.toml"),
        &pkg_toml_with_deps("root", "0.1.0", &[("ghost", "*", Some("./does_not_exist"))]),
    );
    write_file(&root.join("src").join("main.th"), "fn main() {}\n");

    let (success, output) = run_tenthpm(&root, &["build"]);
    assert!(!success, "缺失依赖应导致 build 失败");
    assert!(output.contains("不存在"), "应响亮报缺失依赖: {}", output);
}

// ============ add 冲突时原子性（不落盘） ============

#[test]
fn test_add_conflict_fails_atomically() {
    let test_dir = create_test_dir("m4_add_atomic");

    // lib@1.0.0 与 libv2@2.0.0 是同一包名的两个版本
    write_file(&test_dir.join("lib").join("Tenth.toml"), &pkg_toml("lib", "1.0.0"));
    write_file(&test_dir.join("libv2").join("Tenth.toml"), &pkg_toml("lib", "2.0.0"));
    write_file(
        &test_dir.join("app_a").join("Tenth.toml"),
        &pkg_toml_with_deps("app_a", "0.1.0", &[("lib", "^2.0.0", Some("../libv2"))]),
    );
    write_file(&test_dir.join("app_a").join("src").join("a.th"), "fn a() {}\n");

    // 项目：直接 path 依赖 lib@1.0.0（约束 ^1.0.0），且 path 依赖 app_a（需 lib ^2.0.0）→ 已冲突
    let app = test_dir.join("app");
    write_file(
        &app.join("Tenth.toml"),
        &pkg_toml_with_deps("app", "0.1.0", &[
            ("lib", "^1.0.0", Some("../lib")),
            ("app_a", "*", Some("../app_a")),
        ]),
    );
    write_file(&app.join("src").join("main.th"), "fn main() {}\n");

    let manifest_before = fs::read_to_string(app.join("Tenth.toml")).unwrap();

    // add 一个无关 registry 依赖 → 解析发现既有冲突 → 应失败且不落盘
    let (success, output) = run_tenthpm(&app, &["add", "someother", "1.0.0"]);
    assert!(!success, "存在版本冲突时 add 应失败: {}", output);
    assert!(output.contains("版本冲突"), "add 失败应报冲突: {}", output);

    let manifest_after = fs::read_to_string(app.join("Tenth.toml")).unwrap();
    assert_eq!(
        manifest_before, manifest_after,
        "add 失败时不应修改 manifest（原子性）"
    );
    assert!(
        !manifest_after.contains("someother"),
        "add 失败时不应写入新依赖"
    );
}

// ============ 发布 → 安装 .tenthpkg 闭环 ============

#[test]
fn test_publish_then_install_pkg_file_closure() {
    let test_dir = create_test_dir("m4_pkg_closure");

    // 库项目（可发布）
    let lib = test_dir.join("mylib");
    write_file(&lib.join("Tenth.toml"), &publishable_toml("mylib", "1.0.0"));
    write_file(&lib.join("src").join("main.th"), "fn main() { println(\"lib\"); }\n");

    let (success, output) = run_tenthpm(&lib, &["publish"]);
    assert!(success, "publish 失败: {}", output);
    let pkg_file = lib.join("mylib-1.0.0.tenthpkg");
    assert!(pkg_file.exists(), ".tenthpkg 未生成");

    // 主项目从 .tenthpkg 安装
    let app = test_dir.join("app");
    write_file(&app.join("Tenth.toml"), &pkg_toml("app", "0.1.0"));
    write_file(&app.join("src").join("main.th"), "fn main() {}\n");

    let pkg_str = pkg_file.to_str().unwrap().to_string();
    let (success, output) = run_tenthpm(&app, &["install", &pkg_str]);
    assert!(success, "install .tenthpkg 失败: {}", output);
    assert!(output.contains("Installed"), "应输出 Installed: {}", output);

    // 解包到 deps/mylib
    assert!(
        app.join("deps").join("mylib").join("Tenth.toml").exists(),
        "应解包 Tenth.toml"
    );
    assert!(
        app.join("deps").join("mylib").join("src").join("main.th").exists(),
        "应解包 src/main.th"
    );

    // manifest 有 path 依赖
    let manifest = fs::read_to_string(app.join("Tenth.toml")).unwrap();
    assert!(manifest.contains("mylib"));
    assert!(manifest.contains("deps/mylib"));

    // 锁文件有 mylib
    let lock = fs::read_to_string(app.join("Tenth.lock")).unwrap();
    assert!(lock.contains("mylib"), "锁文件应含 mylib: {}", lock);

    // 安装后 build 通过（依赖解析闭环）
    let (success, output) = run_tenthpm(&app, &["build"]);
    assert!(success, "安装后 build 失败: {}", output);
}

// ============ 本地 registry 发布/安装 ============

#[test]
fn test_registry_publish_and_install() {
    let test_dir = create_test_dir("m4_registry_flow");
    let reg = test_dir.join("reg");

    // 库发布到本地 registry
    let lib = test_dir.join("reglib");
    write_file(&lib.join("Tenth.toml"), &publishable_toml("reglib", "2.1.0"));
    write_file(&lib.join("src").join("main.th"), "fn main() { println(\"reglib\"); }\n");

    let reg_str = reg.to_str().unwrap().to_string();
    let (success, output) = run_tenthpm(&lib, &["publish", "--registry", &reg_str]);
    assert!(success, "publish --registry 失败: {}", output);
    assert!(
        reg.join("reglib-2.1.0.tenthpkg").exists(),
        "registry 目录应有归档"
    );
    assert!(
        output.contains("registry"),
        "应输出发布到 registry 的信息: {}",
        output
    );

    // 主项目从 registry 安装
    let app = test_dir.join("regapp");
    write_file(&app.join("Tenth.toml"), &pkg_toml("regapp", "0.1.0"));
    write_file(&app.join("src").join("main.th"), "fn main() {}\n");

    let (success, output) = run_tenthpm(&app, &["install", "reglib", "--registry", &reg_str]);
    assert!(success, "registry install 失败: {}", output);
    assert!(
        app.join("deps").join("reglib").join("Tenth.toml").exists(),
        "应从 registry 解包到 deps/reglib"
    );

    // build 通过（闭环）
    let (success, output) = run_tenthpm(&app, &["build"]);
    assert!(success, "registry 安装后 build 失败: {}", output);
}

#[test]
fn test_registry_install_missing_loud() {
    let test_dir = create_test_dir("m4_registry_missing");
    let reg = test_dir.join("empty_reg");
    fs::create_dir_all(&reg).unwrap();

    let app = test_dir.join("app");
    write_file(&app.join("Tenth.toml"), &pkg_toml("app", "0.1.0"));
    write_file(&app.join("src").join("main.th"), "fn main() {}\n");

    let reg_str = reg.to_str().unwrap().to_string();
    let (success, output) = run_tenthpm(&app, &["install", "ghost", "--registry", &reg_str]);
    assert!(!success, "缺失包应响亮报错");
    assert!(
        output.contains("未找到包"),
        "应报未找到包: {}",
        output
    );
}

// ============ 版本约束解析（API 层） ============

#[test]
fn test_version_req_api() {
    use tenthpm::version::{reqs_conflict, Version, VersionReq};

    let r = VersionReq::parse("^1.2.3").unwrap();
    assert!(r.matches(&Version::new(1, 5, 0)));
    assert!(!r.matches(&Version::new(2, 0, 0)));

    let r2 = VersionReq::parse(">=1.0.0,<2.0.0").unwrap();
    assert!(r2.matches(&Version::new(1, 9, 9)));
    assert!(!r2.matches(&Version::new(2, 0, 0)));

    // 冲突检测
    assert!(reqs_conflict(&[
        VersionReq::parse("^1.0.0").unwrap(),
        VersionReq::parse("^2.0.0").unwrap(),
    ]));
    assert!(!reqs_conflict(&[
        VersionReq::parse("^1.0.0").unwrap(),
        VersionReq::parse(">=1.5.0").unwrap(),
    ]));
}

// ============ 归档安全（API 层） ============

#[test]
fn test_pkg_archive_security() {
    use tenthpm::pkg;

    // 路径穿越归档应被拒绝
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(b"TENTHPKG\0");
    let ms = publishable_toml("safe", "1.0.0");
    data.extend_from_slice(&(ms.len() as u32).to_le_bytes());
    data.extend_from_slice(ms.as_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    let p = "../evil.th";
    data.extend_from_slice(&(p.len() as u32).to_le_bytes());
    data.extend_from_slice(p.as_bytes());
    let c = b"x";
    data.extend_from_slice(&(c.len() as u32).to_le_bytes());
    data.extend_from_slice(c);

    let err = pkg::parse_archive(&data).unwrap_err();
    assert!(err.contains("非法段"), "应拒绝路径穿越: {}", err);

    // 损坏归档（长度越界）应被拒绝而非 panic
    let mut bad: Vec<u8> = Vec::new();
    bad.extend_from_slice(b"TENTHPKG\0");
    bad.extend_from_slice(&9999u32.to_le_bytes());
    let err = pkg::parse_archive(&bad).unwrap_err();
    assert!(err.contains("越界"), "损坏归档应报错而非 panic: {}", err);

    // 非法 magic
    let err = pkg::parse_archive(b"NOTPKG!!!").unwrap_err();
    assert!(err.contains("magic"));
}

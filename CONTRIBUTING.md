# Contributing to Tenth

感谢考虑为 Tenth 做贡献！

## 行为准则

本项目遵循 [Contributor Covenant](CODE_OF_CONDUCT.md)。请友善交流。

## 如何贡献

### 报告 Bug

1. 在 GitHub Issues 中搜索是否已有相同报告
2. 提供：Tenth 版本、操作系统、复现步骤、期望 vs 实际行为
3. 贴上相关代码片段（用 \`\`\`tenth 代码块）

### 提交代码

1. **Fork** 本仓库
2. 从 `main` 分支创建你的特性分支：`git checkout -b feat/my-feature`
3. 写代码 + 测试
4. 确保全部测试通过：`cargo test --manifest-path tenth/Cargo.toml`
5. 提交：`git commit -m "feat: description"`
6. 推送到你的 fork：`git push origin feat/my-feature`
7. 发起 **Pull Request** 到 `main` 分支

### Commit 规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
feat:     新功能
fix:      Bug 修复
docs:     文档变更
refactor: 重构（无功能变更）
test:     测试相关
chore:    构建/工具
```

示例：`feat: add for-in loop syntax`

### 代码风格

- **Rust**：遵循 `cargo fmt` 和 `cargo clippy`
- **Tenth (.th)**：用 `//` 或 `/* */` 注释，缩进 4 空格
- 新功能必须有对应的测试覆盖

### 目录约定

| 做什么 | 放哪里 |
|---|---|
| 新语言特性（parser/HIR/interpreter） | `tenth/src/` |
| 新标准库函数 | `tenth/std/` |
| 新示例 | `Tenth实例/` |
| 新 TapeOp / backward | `tenth/src/runtime/autodiff.rs` |
| 文档更新 | `docs/` |

### 开发环境

详见 [DEPS.md](DEPS.md)。核心依赖只有 Rust ≥1.95。

```bash
# 安装 Rust
curl -o rustup-init.exe https://win.rustup.rs/x86_64
rustup-init.exe -y --default-toolchain stable

# 编译 + 测试
cargo build --manifest-path tenth/Cargo.toml --release
cargo test --manifest-path tenth/Cargo.toml
```

#!/usr/bin/env bash
# package.sh — Tenth release 产物打包脚本（Linux / macOS / CI 通用）
#
# 用法（仓库根目录执行）：
#   bash scripts/release/package.sh
#   bash scripts/release/package.sh --skip-build
#   bash scripts/release/package.sh -v 1.0.0
#
# 行为：
#   1. 构建 5 个 release 产物（tenth / tenthpm / tenth-debug / tenth-prof / tenth-lsp）
#      （--skip-build 跳过，直接用既有产物）
#   2. 收集到 dist/tenth-<ver>-<target>/：bin/ + std/ + 文档
#   3. 打包 tar.gz + 生成 SHA-256 checksum（dist/SHA256SUMS.txt）
#
# M5.3 跨平台产物（Linux/macOS/Windows-CI 侧，Windows 用 package.ps1）。

set -euo pipefail

VERSION=""
SKIP_BUILD=0
OUT_DIR="dist"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--version) VERSION="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---- 版本号与 target 名 ----
if [[ -z "$VERSION" ]]; then
  VERSION="$(grep -m1 '^version' tenth/Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
fi

case "$(uname -s)" in
  Linux)  TARGET="linux-x86_64" ;;
  Darwin) TARGET="macos-x86_64" ;;
  MINGW*|MSYS*|CYGWIN*) TARGET="windows-x86_64" ;;
  *) TARGET="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)" ;;
esac

STAGING="$OUT_DIR/tenth-$VERSION-$TARGET"
echo "[package] Tenth $VERSION / $TARGET -> $STAGING"

# ---- 1. 构建 ----
if [[ "$SKIP_BUILD" == "0" ]]; then
  for m in \
    "tenth/Cargo.toml" \
    "tenth/tools/tenthpm/Cargo.toml" \
    "tenth/tools/debugger/Cargo.toml" \
    "tenth/tools/profiler/Cargo.toml" \
    "tenth/tools/lsp/Cargo.toml"
  do
    echo "[package] building $(basename "$(dirname "$m")") (release)..."
    cargo build --release -j 2 --manifest-path "$m"
  done
else
  echo "[package] --skip-build：复用既有 release 产物"
fi

# ---- 2. 收集产物 ----
EXT=""; [[ "$TARGET" == windows-* ]] && EXT=".exe"
BIN_DIR="$STAGING/bin"
mkdir -p "$BIN_DIR"

copy_artifact() {
  local name="$1" src="$2"
  if [[ ! -f "$src" ]]; then echo "[package] 缺少产物: $src"; exit 1; fi
  cp "$src" "$BIN_DIR/$name"
  echo "[package]   + bin/$name ($(du -h "$BIN_DIR/$name" | cut -f1))"
}

copy_artifact "tenth$EXT"        "tenth/target/release/tenth$EXT"
copy_artifact "tenthpm$EXT"      "tenth/tools/tenthpm/target/release/tenthpm$EXT"
copy_artifact "tenth-debug$EXT"  "tenth/tools/debugger/target/release/tenth-debug$EXT"
copy_artifact "tenth-prof$EXT"   "tenth/tools/profiler/target/release/tenth-prof$EXT"
copy_artifact "tenth-lsp$EXT"    "tenth/tools/lsp/target/release/tenth-lsp$EXT"

if [[ -d "tenth/std" ]]; then
  cp -r "tenth/std" "$STAGING/std"
  echo "[package]   + std/ (标准库)"
fi

mkdir -p "$STAGING/docs"
for d in "README.md" "RELEASE_NOTES.md" "docs/语言参考手册.md" "docs/API冻结清单.md" "docs/语言规范.md"; do
  if [[ -f "$d" ]]; then cp "$d" "$STAGING/docs/"; fi
done
echo "[package]   + docs/ (手册/规范/冻结清单)"

# ---- 3. 打包 tar.gz + checksum ----
mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/tenth-$VERSION-$TARGET.tar.gz"
rm -f "$TARBALL"
# 打包 staging 目录内容（不含 staging 外壳层）
tar -czf "$TARBALL" -C "$OUT_DIR" "tenth-$VERSION-$TARGET"
HASH="$(sha256sum "$TARBALL" | cut -d' ' -f1)"
echo "$HASH  $(basename "$TARBALL")" >> "$OUT_DIR/SHA256SUMS.txt"
echo "[package] tarball: $TARBALL"
echo "[package] sha256: $HASH"
echo "[package] 完成。"

# sync-public.ps1 — 生成对外发布快照（干净仓库素材）
#
# 用途：从开发仓库（含大量内部文档）整理出"对外发布树"——
#       只含对外源码/文档/CI，排除内部工作文件（.trae/.agents/能力梳理/
#       MEMO/AUDIT/CODE_WIKI/设计调研文档等），且不带 .git 历史。
#       生成物供"新仓库单次初始提交"使用，保证对外提交记录干净专业。
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts/release/sync-public.ps1 -OutDir "<发布仓库绝对路径>"
#       （发布仓库必须在开发仓库之外、独立 .git，避免 git 嵌套）

param(
    [Parameter(Mandatory = $true)]
    [string]$OutDir
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$out  = $OutDir

Write-Host "[sync] 开发仓库: $root"
Write-Host "[sync] 输出目录: $out"

if (Test-Path $out) {
    $existing = (Get-ChildItem $out -Force | Measure-Object).Count
    if ($existing -gt 0) { Write-Host "[sync] 目标目录非空（$existing 项），将清空重建..." }
    # 只清空目录内容，不删目录本体（避免被文件管理器等占用导致锁定失败）
    Get-ChildItem $out -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
}
if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out -Force | Out-Null }

# ---------- 对外清单（白名单） ----------
# 目录（整体复制，内部自动排除 target/ 等）
$dirs = @(
    "tenth",          # 主编译器 + 工具 + 标准库 + 测试
    "tenthc",         # 自举编译器（Tenth 源码）
    "Tenth实例",      # 可运行示例
    ".github",        # CI（release.yml 等）
    "scripts\release" # 发布脚本
)
# 根文件
$files = @(
    "README.md",
    "RELEASE_NOTES.md",
    "LICENSE",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "DEPS.md",
    ".gitignore"
)
# 对外文档（docs/ 仅 4 个用户可读文档，其余为内部设计/调研）
$docFiles = @(
    "docs\语言参考手册.md",
    "docs\语言规范.md",
    "docs\API冻结清单.md",
    "docs\教程.md"
)

# ---------- 复制 ----------
foreach ($d in $dirs) {
    $src = Join-Path $root $d
    if (-not (Test-Path $src)) { Write-Warning "[sync] 缺少目录: $d"; continue }
    Write-Host "[sync] 复制目录: $d"
    # robocopy：/E 含子目录（含空目录） /XD 排除目录 /XF 排除文件
    robocopy $src (Join-Path $out $d) /E /NFL /NDL /NJH /NJS /NC /NS `
        /XD target dist .git node_modules `
        /XF *.pdb *.wasm *.zip *.tar.gz tenthc_full.wasm smoke_m43_tmp.csv test_output.txt sync-public.ps1 | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "[sync] robocopy 失败: $d (code $LASTEXITCODE)" }
}
foreach ($f in $files) {
    $src = Join-Path $root $f
    if (-not (Test-Path $src)) { Write-Warning "[sync] 缺少文件: $f"; continue }
    Copy-Item $src (Join-Path $out $f) -Force
    Write-Host "[sync] 复制文件: $f"
}
foreach ($f in $docFiles) {
    $src = Join-Path $root $f
    if (-not (Test-Path $src)) { Write-Warning "[sync] 缺少文档: $f"; continue }
    $dst = Join-Path $out $f
    New-Item -ItemType Directory -Path (Split-Path $dst) -Force | Out-Null
    Copy-Item $src $dst -Force
    Write-Host "[sync] 复制文档: $f"
}

# ---------- 对外 .gitignore ----------
$gitignore = @"
# 构建产物
target/
dist/
*.pdb
*.wasm
*.zip
*.tar.gz

# 生成物
tenthc_full.wasm
smoke_m43_tmp.csv
"@
Set-Content -Path (Join-Path $out ".gitignore") -Value $gitignore -Encoding UTF8

# ---------- 校验：确保无内部文件泄漏 ----------
Write-Host ""
Write-Host "[sync] === 校验（内部文件不得出现） ==="
$leakPatterns = @(".trae", ".agents", "能力梳理", "MEMO.md", "AUDIT.md", "CODE_WIKI.md",
                  "基本功核查", "现状调研报告", "security_review", "用户反馈",
                  "程序张量探索", "shape-check-roadmap", "虚拟管理体系", "编译到机器码方案",
                  "superpowers", "AGENTS.md", "async-concurrency-design", "gpu-feasibility",
                  "stdlib-可用性盘点", "理论分析点", "程序代数架构", "语言核心补全方案",
                  "f32-f64-parity-roadmap", "io-primitives-design", "论文")
$leaks = @()
Get-ChildItem $out -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($out.Length + 1)
    foreach ($p in $leakPatterns) {
        if ($rel -match [regex]::Escape($p)) { $leaks += $rel; break }
    }
}
if ($leaks.Count -gt 0) {
    Write-Warning "[sync] ⚠️ 发现内部文件泄漏："
    $leaks | ForEach-Object { Write-Warning "    $_" }
} else {
    Write-Host "[sync] ✅ 无内部文件泄漏，快照干净"
}

# ---------- 汇总 ----------
$n = (Get-ChildItem $out -Recurse -File).Count
$size = (Get-ChildItem $out -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ""
Write-Host ("[sync] 完成：{0} 个文件，{1:N1} MB" -f $n, ($size/1MB))
Write-Host "[sync] 输出目录：$out"
Write-Host ""
Write-Host '[sync] 下一步（GitHub 建新仓库后，在此目录执行）：'
Write-Host '    git init'
Write-Host '    git add .'
Write-Host '    git commit -m "feat: Tenth v1.0.0 - initial release"'
Write-Host '    git branch -M main'
Write-Host '    git remote add origin <new-repo-url>.git'
Write-Host '    git push -u origin main'

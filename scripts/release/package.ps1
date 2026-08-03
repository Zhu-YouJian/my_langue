# package.ps1 — Tenth Windows release 产物打包脚本
#
# 用法（仓库根目录执行）：
#   powershell -ExecutionPolicy Bypass -File scripts/release/package.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/release/package.ps1 -SkipBuild
#   powershell -ExecutionPolicy Bypass -File scripts/release/package.ps1 -Version 1.0.0
#
# 行为：
#   1. 构建 5 个 release 产物（tenth / tenthpm / tenth-debug / tenth-prof / tenth-lsp）
#      （-SkipBuild 跳过，直接用既有产物）
#   2. 收集到 dist/tenth-<ver>-windows-x86_64/：bin/ + std/ + 文档
#   3. 打包 zip + 生成 SHA-256 checksum（dist/SHA256SUMS.txt）
#
# M5.3 跨平台产物（Windows 侧）。

param(
    [string]$Version = "",
    [switch]$SkipBuild,
    [string]$OutDir = "dist"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

# ---- 版本号：默认从 tenth/Cargo.toml 读取 ----
if (-not $Version) {
    $line = Get-Content "tenth/Cargo.toml" | Where-Object { $_ -match '^version\s*=\s*"([^"]+)"' } | Select-Object -First 1
    if ($line -match '^version\s*=\s*"([^"]+)"') { $Version = $Matches[1] } else { $Version = "0.0.0" }
}
$Target = "windows-x86_64"
$Staging = Join-Path $OutDir "tenth-$Version-$Target"
Write-Host "[package] Tenth $Version / $Target -> $Staging"

# ---- 1. 构建 ----
if (-not $SkipBuild) {
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    $builds = @(
        @{ Name = "tenth";      Manifest = "tenth/Cargo.toml" },
        @{ Name = "tenthpm";    Manifest = "tenth/tools/tenthpm/Cargo.toml" },
        @{ Name = "tenth-debug";Manifest = "tenth/tools/debugger/Cargo.toml" },
        @{ Name = "tenth-prof"; Manifest = "tenth/tools/profiler/Cargo.toml" },
        @{ Name = "tenth-lsp";  Manifest = "tenth/tools/lsp/Cargo.toml" }
    )
    foreach ($b in $builds) {
        Write-Host "[package] building $($b.Name) (release)..."
        cargo build --release -j 2 --manifest-path $b.Manifest
        if ($LASTEXITCODE -ne 0) { throw "构建失败: $($b.Name)" }
    }
} else {
    Write-Host "[package] -SkipBuild：复用既有 release 产物"
}

# ---- 2. 收集产物 ----
$binDir = Join-Path $Staging "bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

$artifacts = @(
    @{ Name = "tenth.exe";       Src = "tenth/target/release/tenth.exe" },
    @{ Name = "tenthpm.exe";     Src = "tenth/tools/tenthpm/target/release/tenthpm.exe" },
    @{ Name = "tenth-debug.exe"; Src = "tenth/tools/debugger/target/release/tenth-debug.exe" },
    @{ Name = "tenth-prof.exe";  Src = "tenth/tools/profiler/target/release/tenth-prof.exe" },
    @{ Name = "tenth-lsp.exe";   Src = "tenth/tools/lsp/target/release/tenth-lsp.exe" }
)

foreach ($a in $artifacts) {
    if (-not (Test-Path $a.Src)) { throw "缺少产物: $($a.Src)（先构建或去掉 -SkipBuild）" }
    Copy-Item $a.Src (Join-Path $binDir $a.Name) -Force
    Write-Host "[package]   + bin/$($a.Name) ($([math]::Round((Get-Item (Join-Path $binDir $a.Name)).Length/1MB,1)) MB)"
}

# std/ 标准库（运行时必需：use std::... 解析）
if (Test-Path "tenth/std") {
    Copy-Item "tenth/std" (Join-Path $Staging "std") -Recurse -Force
    Write-Host "[package]   + std/ (标准库)"
}

# 文档
$docs = @(
    @{ Name = "README.md";               Src = "README.md" },
    @{ Name = "语言参考手册.md";          Src = "docs/语言参考手册.md" },
    @{ Name = "API冻结清单.md";           Src = "docs/API冻结清单.md" },
    @{ Name = "语言规范.md";              Src = "docs/语言规范.md" }
)
New-Item -ItemType Directory -Force -Path (Join-Path $Staging "docs") | Out-Null
foreach ($d in $docs) {
    if (Test-Path $d.Src) {
        Copy-Item $d.Src (Join-Path (Join-Path $Staging "docs") $d.Name) -Force
    }
}
Write-Host "[package]   + docs/ (手册/规范/冻结清单)"

# ---- 3. 打包 zip + checksum ----
$zip = Join-Path $OutDir "tenth-$Version-$Target.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $Staging -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
$sums = Join-Path $OutDir "SHA256SUMS.txt"
Add-Content -Path $sums -Value "$hash  $((Get-Item $zip).Name)"
Write-Host "[package] zip: $zip"
Write-Host "[package] sha256: $hash"
Write-Host "[package] 完成。"

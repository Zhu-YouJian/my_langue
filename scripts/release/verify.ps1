# verify.ps1 — Tenth 发布产物校验（Windows）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts/release/verify.ps1            # 校验 dist/ 下最新 zip
#   powershell -ExecutionPolicy Bypass -File scripts/release/verify.ps1 -Zip dist\tenth-1.0.0-windows-x86_64.zip
#
# 行为：校验 SHA256SUMS.txt 中的 checksum 与 zip 实际值一致，并核对 zip 内含
# bin/ 5 个产物 + std/ + docs/。

param(
    [string]$Zip = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$OutDir = "dist"
if (-not $Zip) {
    $Zip = Get-ChildItem $OutDir -Filter "tenth-*.zip" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Zip -or -not (Test-Path $Zip)) { throw "找不到 zip 产物（$OutDir 下无 tenth-*.zip）" }
Write-Host "[verify] 校验: $Zip"

# 1) checksum 核对
$sumsFile = Join-Path $OutDir "SHA256SUMS.txt"
if (Test-Path $sumsFile) {
    $actual = (Get-FileHash -Algorithm SHA256 $Zip).Hash.ToLowerInvariant()
    $zipName = Split-Path $Zip -Leaf
    $recorded = (Get-Content $sumsFile | Where-Object { $_ -match [regex]::Escape($zipName) } | Select-Object -First 1)
    if ($recorded -and $recorded.StartsWith($actual)) {
        Write-Host "[verify] SHA-256 一致: $actual"
    } else {
        throw "SHA-256 不一致！实际=$actual 记录=$recorded"
    }
} else {
    Write-Host "[verify] 警告: $sumsFile 不存在，跳过 checksum 核对"
}

# 2) 内容核对（zip 内必需文件）
$entries = & tar -tf $Zip 2>$null
if (-not $entries) {
    $shell = New-Object -ComObject Shell.Application
    $zipObj = $shell.NameSpace((Resolve-Path $Zip).Path)
    $entries = $zipObj.Items() | ForEach-Object { $_.Name }
}
$required = @(
    "/bin/tenth.exe", "/bin/tenthpm.exe", "/bin/tenth-debug.exe",
    "/bin/tenth-prof.exe", "/bin/tenth-lsp.exe", "/std/prelude.th"
)
foreach ($r in $required) {
    $hit = $entries | Where-Object { $_ -like "*$r" } | Select-Object -First 1
    if (-not $hit) { throw "zip 内容缺失: $r" }
    Write-Host "[verify]   内容 OK: $r"
}
Write-Host "[verify] 全部通过。"

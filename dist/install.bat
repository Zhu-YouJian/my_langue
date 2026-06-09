@echo off
REM Tenth 环境安装 — 将 dist 目录加入用户 PATH
set "TENTH_DIR=%~dp0"
echo [Tenth] Installing from %TENTH_DIR%

REM 检查是否已在 PATH 中
echo %PATH% | findstr /C:"%TENTH_DIR%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [Tenth] Already in PATH
) else (
    setx PATH "%PATH%;%TENTH_DIR%"
    echo [Tenth] Added to user PATH — restart terminal to take effect
)

echo.
echo Usage: tenth run file.th   — 解释执行
echo        tenth build file.th  — 编译到 WASM
echo        tenth wasm file.th   — wasmi 执行
echo        tenth                — 启动 REPL
echo        tenth --max-memory N — 限制内存 (MB)

@echo off
set "TENTH_DIR=%~dp0"
echo [Tenth] Installing from %TENTH_DIR%

echo %PATH% | findstr /C:"%TENTH_DIR%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [Tenth] Already in PATH
) else (
    setx PATH "%PATH%;%TENTH_DIR%"
    echo [Tenth] Added to user PATH - restart terminal to take effect
)

echo.
echo Usage:
echo   tenth run file.th     - run .th file
echo   tenth build file.th   - compile to WASM
echo   tenth wasm file.th    - wasmi execute
echo   tenth                 - start REPL
echo   tenth --max-memory N  - limit memory (MB)

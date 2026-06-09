@echo off
REM Tenth Language Tool — 便捷包装
REM 用法: tenth run file.th | tenth build file.th | tenth wasm file.th
REM 不加参数启动 REPL

set TENTH_HOME=%~dp0
"%TENTH_HOME%tenth.exe" %*

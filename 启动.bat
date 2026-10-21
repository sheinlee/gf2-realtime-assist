@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
rem 用 ok-gf2 自带的 Python。找的顺序和 main.py 的 find_ok_gf2 一致：
rem 环境变量 OK_GF2_DIR（ok-gf2 的 working 目录） / 本文件夹放在 ok-gf2 的文件夹里（解压版） / 安装版的默认位置
set "PY="
if defined OK_GF2_DIR if exist "%OK_GF2_DIR%\..\python\pythonw.exe" set "PY=%OK_GF2_DIR%\..\python\pythonw.exe"
if not defined PY if exist "%~dp0..\data\apps\ok-gf2\python\pythonw.exe" set "PY=%~dp0..\data\apps\ok-gf2\python\pythonw.exe"
if not defined PY if exist "%LOCALAPPDATA%\ok-gf2\data\apps\ok-gf2\python\pythonw.exe" set "PY=%LOCALAPPDATA%\ok-gf2\data\apps\ok-gf2\python\pythonw.exe"
if not defined PY (
    echo [错误] 找不到 ok-gf2 自带的 Python。
    echo.
    echo 本程序复用 ok-gf2 的运行环境，ok-gf2 要先装好、正常打开过一次。
    echo   - 安装包版 ok-gf2：默认装在 %LOCALAPPDATA%\ok-gf2，能直接找到
    echo   - 解压版 ok-gf2：把本程序的文件夹放进 ok-gf2 的文件夹，和 ok-gf2.exe 放在同一层
    echo   - 也可以设环境变量 OK_GF2_DIR，指向 ok-gf2 的 data\apps\ok-gf2\working 目录
    pause
    exit /b 1
)
start "" "%PY%" "%~dp0main.py"

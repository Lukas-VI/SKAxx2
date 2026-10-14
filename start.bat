@echo off
REM ==========================================================================
REM SKAxx2 一键启动（Windows cmd）
REM --------------------------------------------------------------------------
REM 用法：双击本文件，或在 cmd 中执行 start.bat
REM   1) 首次运行自动创建 .venv 并安装 requirements.txt
REM   2) 之后每次启动复用已有 .venv，直接进入主程序
REM ==========================================================================
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo 未找到 python，请先安装 Python 3.10+ 并勾选 Add to PATH
  pause
  exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
  echo [1/2] 创建虚拟环境 .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo 创建虚拟环境失败
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
echo [2/2] 检查并安装依赖 ...
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo 启动 SKAxx2 —— 管理后台 http://127.0.0.1:5098 （Ctrl+C 退出）
python main.py

pause

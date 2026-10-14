#!/usr/bin/env bash
# ============================================================================
# SKAxx2 一键启动（Linux / macOS）
# ----------------------------------------------------------------------------
# 用法：  chmod +x start.sh && ./start.sh
#   1) 首次运行自动创建 .venv 并安装 requirements.txt
#   2) 之后每次启动复用已有 .venv，直接进入主程序
# 环境变量：PYTHON=python3.12 ./start.sh   # 指定解释器
# ============================================================================
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "未找到 $PY，请先安装 Python 3.10+"; exit 1; }

if [ ! -d .venv ]; then
  echo "[1/2] 创建虚拟环境 .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

echo "[2/2] 检查并安装依赖 ..."
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo "启动 SKAxx2 —— 管理后台 http://127.0.0.1:5098 （Ctrl+C 退出）"
exec python main.py

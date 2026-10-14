# ============================================================================
# SKAxx2 一键启动（Windows PowerShell）
# ----------------------------------------------------------------------------
# 用法：  powershell -ExecutionPolicy Bypass -File start.ps1
#   若报「禁止运行脚本」，说明执行策略受限，用上面带 Bypass 的写法即可。
#   逻辑与 start.bat 相同：首次建 .venv 装依赖，之后复用并直接启动。
# ============================================================================
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host '未找到 python，请先安装 Python 3.10+ 并勾选 Add to PATH'
    exit 1
}

if (-not (Test-Path '.venv\Scripts\Activate.ps1')) {
    Write-Host '[1/2] 创建虚拟环境 .venv ...'
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1

Write-Host '[2/2] 检查并安装依赖 ...'
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

Write-Host '启动 SKAxx2 —— 管理后台 http://127.0.0.1:5098 （Ctrl+C 退出）'
python main.py

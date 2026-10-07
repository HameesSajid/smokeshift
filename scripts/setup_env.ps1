# Creates the Python environment SmokeShift uses for GPU training and the
# foundation models, in a .venv folder inside the project. Nothing is installed
# into your normal Python.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# Everything this script creates is listed in docs\INSTALLED.md, with how to remove it.

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$venv = if ($env:SMOKESHIFT_VENV) { $env:SMOKESHIFT_VENV } else { Join-Path $root ".venv" }
$py = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment at $venv"
    python -m venv $venv
}
& $py -m pip install --upgrade pip --no-cache-dir --quiet

# PyTorch built for CUDA 12.4 (works with NVIDIA GTX 16xx / RTX cards and current drivers).
# --no-cache-dir stops pip keeping a second 2.5 GB copy of the download.
# On a machine without an NVIDIA GPU, replace the next line with:  & $py -m pip install torch
Write-Host "Installing PyTorch (CUDA build, about 2.5 GB download)"
& $py -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing the scientific stack and the foundation-model packages"
& $py -m pip install --no-cache-dir numpy pandas scipy scikit-learn matplotlib xgboost pytest chronos-forecasting "timesfm[torch]"

Write-Host "`nInstalled:"
& $py -c "import torch, xgboost, sklearn, pandas; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'); print('xgboost', xgboost.__version__, '| pandas', pandas.__version__, '| scikit-learn', sklearn.__version__)"
& $py -m pip freeze | Out-File -Encoding utf8 (Join-Path $root "docs\environment_freeze.txt")
Write-Host "`nRun the project with:  .venv\Scripts\python.exe run.py <stage>"

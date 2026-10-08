# Runs the GPU stages in the project's .venv environment, then rescoring and reporting.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_gpu_queue.ps1
#
# About four hours on a GTX 1660 Ti. Safe to stop and rerun: finished models are
# stored, and each stage overwrites only its own outputs.

$root = Split-Path $PSScriptRoot -Parent
$venv = if ($env:SMOKESHIFT_VENV) { $env:SMOKESHIFT_VENV } else { Join-Path $root ".venv" }
$py = Join-Path $venv "Scripts\python.exe"
$env:PYTHONWARNINGS = "ignore"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$score = "ensemble", "evaluate", "uncertainty", "alerts", "figures", "report", "dashboard"
Set-Location $root

# 1. Transformer on the GPU, then the three foundation models with the shared 168-hour context.
$env:SMOKESHIFT_FM = "Chronos-2,TimesFM,Chronos-Bolt"
& $py run.py neural foundation @score

# 2. The long-context variants, to see whether the 168-hour budget holds a foundation model back.
$env:SMOKESHIFT_FM = "Chronos-2 (long context),Chronos-Bolt (long context),TimesFM (long context)"
& $py run.py foundation @score

# 3. Five training seeds per transformer, to measure run-to-run spread (about 30 minutes).
& $py run.py seeds report

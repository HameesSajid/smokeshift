# What is installed and stored, where, and how to remove it

This is the complete list of everything the project puts on a computer. It is updated whenever something is
installed, downloaded or moved. Sizes were measured on 2026-10-07 after a full run of all 14 methods.

**Everything is inside the project folder.** Nothing is installed into your normal Python, and no system setting
is changed. Deleting the project folder removes all of it.

## Summary

Paths are relative to the project folder, `C:\Users\hamee\vscode\smokeshift`.

| # | What | Where | Size | In git? | Safe to delete? |
|---|---|---|---|---|---|
| 1 | Code, documents, results, cleaned data panel | the folder itself | 30 MB | yes | It is the project |
| 2 | Python environment | `.venv\` | 5.0 GB | no | Yes. `scripts\setup_env.ps1` rebuilds it |
| 3 | Raw EPA downloads and map outline | `.cache\raw\` | 0.27 GB | no | Yes. `python run.py download` fetches them again |
| 4 | Stored forecasts of every model | `.cache\forecasts\` | 7.9 GB | no | Yes, but rescoring then needs the models rerun |
| 5 | Trained model files | `.cache\models\` | 0.23 GB | no | Yes. Retraining recreates them |
| 6 | Pretrained weights | `.cache\huggingface\` | 2.1 GB | no | Yes. Downloaded again on next use |

Total on disk: about 15.5 GB, of which 30 MB is in the repository.

The companion project from the first phase of this work is a separate folder, `C:\Users\hamee\vscode\air_sensor_trust`
(8 MB). It has no environment or cache of its own.

On GitHub: https://github.com/HameesSajid/smokeshift holds item 1 only.

## Details

### 2. Python environment (`.venv`)

Created by `scripts\setup_env.ps1` with `python -m venv`, from the Python already on the computer. It contains,
among others:

| Package | Version | Why |
|---|---|---|
| torch | 2.6.0+cu124 | PyTorch built for CUDA 12.4, so the transformer and foundation models run on an NVIDIA GPU |
| xgboost | 3.2.0 | gradient-boosted trees (GPU capable) |
| chronos-forecasting | 2.3.2 | Amazon Chronos-Bolt and Chronos-2 |
| timesfm | 3.0.2 | Google TimesFM 2.5 |
| transformers, accelerate, huggingface_hub | 5.19.0, 1.15.0, 1.33.0 | required by the two packages above |
| numpy, pandas, scipy, scikit-learn, matplotlib | 2.2.6, 2.3.3, 1.15.3, 1.7.2, 3.10.9 | the scientific stack |
| pytest | 9.1.1 | running the tests |

The exact list of all 62 packages is in `docs/environment_freeze.txt`. The installs use `--no-cache-dir`, so pip
keeps no second copy of the downloads.

The normal Python on this computer (`C:\Users\hamee\AppData\Local\Programs\Python\Python310`) was not modified.

### 3. Raw downloads (`.cache\raw`)

| File | Source | Size |
|---|---|---|
| `epa\hourly_88101_2021.zip` | https://aqs.epa.gov/aqsweb/airdata/hourly_88101_2021.zip | 65.7 MB |
| `epa\hourly_88101_2022.zip` | same site | 72.3 MB |
| `epa\hourly_88101_2023.zip` | same site | 73.4 MB |
| `epa\hourly_88101_2024.zip` | same site | 63.3 MB |
| `us-atlas\states-10m.json` | https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json | 0.1 MB |

Checksums of the EPA files are in `data/processed/raw_manifest.json`.

### 4 and 5. Forecasts and trained models

`.cache\forecasts\test\<model name>\` holds `point.npy`, `quantiles.npy` and `info.json` for each of the 14
methods: about 0.75 GB for each of the eleven that give quantiles and 0.07 GB for each of the three naive ones.
`.cache\models\` holds the three XGBoost models (`.json`) and the two transformer weight files (`.pt`).

### 6. Pretrained weights (`.cache\huggingface`)

Downloaded from the Hugging Face Hub on first use. The project sets `HF_HOME` to this folder so that nothing lands
in a shared cache elsewhere on the computer.

| Model | Repository | Size |
|---|---|---|
| Chronos-Bolt (base) | `amazon/chronos-bolt-base` | 0.76 GB |
| Chronos-2 | `amazon/chronos-2` | 0.45 GB |
| TimesFM 2.5 (200M) | `google/timesfm-2.5-200m-pytorch` | 0.86 GB |

## Things that were already on this computer and were not created by this project

| What | Where | Note |
|---|---|---|
| Git 2.56 and its credential manager | `C:\Program Files\Git` | Used to push to GitHub with the sign-in already stored there |
| pip download cache | `C:\Users\hamee\AppData\Local\pip\Cache` (3.6 GB) | From earlier installs. `pip cache purge` clears it |
| PyTorch hub cache | `C:\Users\hamee\.cache\torch` | Created in February 2025 |
| The original course files | `...\OneDrive - UAB...\FDS\project` | Never modified. The two project folders were created beside them and later moved out |

## How to remove things

Run these in PowerShell from the project folder. Each line is independent.

```powershell
# The Python environment (5.0 GB)
Remove-Item -Recurse -Force .venv

# Everything regenerable: downloads, forecasts, trained models, pretrained weights (10.5 GB)
Remove-Item -Recurse -Force .cache

# Only the pretrained weights (2.1 GB), keeping results
Remove-Item -Recurse -Force .cache\huggingface

# Only the stored forecasts (7.9 GB), keeping everything else
Remove-Item -Recurse -Force .cache\forecasts
```

After removing `.cache`, the report, figures, tables and dashboard still open, because they are ordinary files in
the project. Only re-running models or re-scoring needs it.

To remove everything, delete the folder `C:\Users\hamee\vscode\smokeshift`. To remove the copy on GitHub, use the
repository's Settings page.

To keep the cache on another drive, set the environment variable `SMOKESHIFT_CACHE` to a folder there before
running. To put the environment somewhere else, set `SMOKESHIFT_VENV` before running `scripts\setup_env.ps1`.

## Using the GPU

The project uses the GPU automatically when it can:

| Stage | Uses the GPU when | Otherwise |
|---|---|---|
| `tune`, `gbm`, `alerts` | XGBoost reports CUDA support | CPU, with fewer boosting rounds |
| `neural` | PyTorch was installed with CUDA (it is, in `.venv`) | CPU, with a smaller model |
| `foundation` | PyTorch was installed with CUDA | CPU, impractically slow |

So to run on the GPU, use the environment's Python:

```powershell
.venv\Scripts\python run.py neural foundation
# or all GPU stages followed by rescoring:
powershell -ExecutionPolicy Bypass -File scripts\run_gpu_queue.ps1
```

To confirm the GPU is being used: `.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
should print `True` and the name of the card, and `nvidia-smi` shows the load while a stage runs.

Measured on the GTX 1660 Ti (6 GB): training both transformers takes 6 minutes on the GPU. The smaller CPU version
took about 28 minutes. XGBoost trains about ten times faster on the GPU than on the CPU.

## Log

| Date | Change |
|---|---|
| 2026-10-07 | Downloaded four EPA files (275 MB) and the map outline (0.1 MB) |
| 2026-10-07 | Created a cache folder for raw data, forecasts and models |
| 2026-10-07 | Created a Python environment with CUDA PyTorch, Chronos and TimesFM (5.0 GB) |
| 2026-10-07 | Downloaded pretrained weights for Chronos-Bolt, Chronos-2 and TimesFM (2.1 GB) |
| 2026-10-07 | A test run briefly put a second copy of the TimesFM weights in `C:\Users\hamee\.cache\huggingface` (0.86 GB); removed the same hour |
| 2026-10-07 | Created the private GitHub repository HameesSajid/smokeshift |
| 2026-10-07 | Moved both project folders from the OneDrive course folder to `C:\Users\hamee\vscode` |
| 2026-10-07 | Moved the cache from `C:\Users\hamee\.cache\smokeshift` to `.cache` inside the project |
| 2026-10-07 | Rebuilt the environment as `.venv` inside the project and removed the old one at `C:\Users\hamee\.venvs\smokeshift` |

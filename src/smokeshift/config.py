"""Paths and the experimental design, in one place."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

# Raw downloads, stored forecasts and model weights are large (about 11 GB in
# all). They live in a .cache folder inside the project, which git ignores, so
# that deleting the project folder removes everything. Set SMOKESHIFT_CACHE to
# keep them somewhere else, for example on another drive.
CACHE = Path(os.environ.get("SMOKESHIFT_CACHE", ROOT / ".cache"))
RAW = CACHE / "raw"
FORECASTS = CACHE / "forecasts"
MODELS = CACHE / "models"
# Pretrained weights fetched from the Hugging Face Hub go here too, so that
# deleting CACHE removes everything this project ever downloaded.
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))

SEED = 7

# ------------------------------------------------------------------ data
YEARS = [2021, 2022, 2023, 2024]
EPA_URL = "https://aqs.epa.gov/aqsweb/airdata/hourly_88101_{year}.zip"
MIN_COMPLETENESS = 0.85      # share of hours a monitor must report, in every year
VALID_RANGE = (-10.0, 1000.0)  # µg/m³; readings outside are instrument faults

# -------------------------------------------------------------- the split
# Models may learn from 2021-2022 only. 2023, the record smoke season in the
# eastern US, is the test year; 2024 checks the return to ordinary conditions.
TRAIN_END = "2022-07-01"     # fit on everything before this
VALID_END = "2023-01-01"     # early stopping, tuning and conformal warm-up
TEST_YEARS = [2023, 2024]

# ------------------------------------------------------------ forecasting
CONTEXT = 168                # hours of history every model may look at
HORIZON = 48                 # forecast 1 to 48 hours ahead
ISSUE_HOUR_UTC = 12          # one forecast per monitor per day, issued at 12:00 UTC
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # deciles, as the foundation models emit
INTERVAL = (0.1, 0.9)        # the 80% prediction interval every model is judged on
HORIZON_BUCKETS = {"1-6 h": (1, 6), "7-24 h": (7, 24), "25-48 h": (25, 48)}

# ---------------------------------------------------------------- regimes
# US EPA AQI breakpoint (2024 revision): above this, 24-hour PM2.5 is
# "Unhealthy for Sensitive Groups" or worse.
USG = 35.4

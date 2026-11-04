"""Paths and constants shared by every stage of the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "AirQuality.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
PRED = OUT / "predictions"

SEED = 42

# Hourly NO2 limit value in ug/m3 (EU Directive 2008/50/EC; also the WHO 2021
# 1-hour guideline). An hour above this is the event the alert system predicts.
NO2_LIMIT = 200.0

# Forecast horizons in hours, and the horizon used for the detailed figures.
HORIZONS = [1, 3, 6, 12, 24]
FOCUS_H = 6

# Target miscoverage for prediction intervals (0.10 -> 90% intervals).
ALPHA = 0.10

# A missed exceedance is treated as this many times worse than a false alarm,
# which puts the alert threshold at 1 / (1 + COST_RATIO) for calibrated
# probabilities.
COST_RATIO = 5.0

# Low-cost sensor channels, weather channels and reference analysers.
SENSORS = ["s1_co", "s2_nmhc", "s3_nox", "s4_no2", "s5_o3"]
WEATHER = ["temp", "rh", "ah"]
REFERENCE = ["co_ref", "nox_ref", "no2_ref", "benzene_ref", "nmhc_ref"]

LABELS = {
    "co_ref": "CO (mg/m³)",
    "nox_ref": "NOx (ppb)",
    "no2_ref": "NO2 (µg/m³)",
    "benzene_ref": "Benzene (µg/m³)",
    "nmhc_ref": "NMHC (µg/m³)",
    "s1_co": "Sensor 1 (CO)",
    "s2_nmhc": "Sensor 2 (NMHC)",
    "s3_nox": "Sensor 3 (NOx)",
    "s4_no2": "Sensor 4 (NO2)",
    "s5_o3": "Sensor 5 (O3)",
    "temp": "Temperature (°C)",
    "rh": "Relative humidity (%)",
    "ah": "Absolute humidity",
}

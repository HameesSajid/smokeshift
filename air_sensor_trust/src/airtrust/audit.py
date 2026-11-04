"""Audit of the original notebook's evaluation.

Re-runs the original preprocessing line for line on the same file, so the
effect of each step is measured rather than asserted:

1. The target (a "hazard score") was a weighted sum of three columns that were
   left in the feature matrix, so the models only had to learn an identity.
2. An IQR "outlier" filter removed the highest readings, which are the
   pollution episodes the project is supposed to be about.
3. The -200 missing marker was only replaced in numeric columns. Five columns
   were still text at that point, so their markers survived as real values.
4. Everything else that was missing was forward-filled, including the empty
   rows at the end of the file.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from . import config

WEIGHTS = {"CO(GT)": 0.7, "NO2(GT)": 0.9, "C6H6(GT)": 1.0}
SENSOR_COLS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
TEXT_COLS = ["CO(GT)", "C6H6(GT)", "T", "RH", "AH"]


def _original_preprocessing(path):
    df = pd.read_csv(path, delimiter=";")
    info = {"rows_in_file": int(len(df)), "empty_trailing_rows": int(df["Date"].isna().sum())}
    df = df.replace(-200, np.nan).drop(columns=["Unnamed: 15", "Unnamed: 16"]).ffill()
    for col in TEXT_COLS:
        df[col] = df[col].str.replace(",", ".").astype(float)
    info["missing_markers_kept_as_values"] = {c: int((df[c] == -200).sum()) for c in TEXT_COLS}
    fences, dropped = {}, {}
    for col in ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        before = len(df)
        df = df[~((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr))]
        fences[col], dropped[col] = float(q3 + 1.5 * iqr), int(before - len(df))
    info.update(upper_fences=fences, rows_dropped_by_filter=dropped, rows_left=int(len(df)))
    return df, info


def _first_fold_r2(X, y):
    train, test = next(TimeSeriesSplit(n_splits=5).split(X))
    model = LinearRegression().fit(X.iloc[train], y.iloc[train])
    return float(r2_score(y.iloc[test], model.predict(X.iloc[test])))


def run(df):
    """`df` is the clean table from data.load(); row i of both files is the same hour."""
    old, info = _original_preprocessing(config.DATA)
    hazard = sum(old[c] * w for c, w in WEIGHTS.items()) / sum(WEIGHTS.values())

    kept = np.zeros(len(df), dtype=bool)
    kept[old.index[old.index < len(df)]] = True
    exceed = (df["no2_ref"] > config.NO2_LIMIT).to_numpy()
    co_top = (df["co_ref"] > df["co_ref"].quantile(0.95)).to_numpy()

    info.update({
        "r2_original_feature_set": _first_fold_r2(old[list(WEIGHTS) + ["NOx(GT)"] + SENSOR_COLS], hazard),
        "r2_three_components_only": _first_fold_r2(old[list(WEIGHTS)], hazard),
        "r2_sensors_and_weather_only": _first_fold_r2(old[SENSOR_COLS], hazard),
        "real_hours": int(len(df)),
        "real_hours_removed_pct": float((~kept).mean() * 100),
        "no2_exceedance_hours": int(exceed.sum()),
        "no2_exceedance_hours_removed": int((exceed & ~kept).sum()),
        "no2_exceedance_removed_pct": float((exceed & ~kept).sum() / exceed.sum() * 100),
        "co_top5pct_hours_removed_pct": float((co_top & ~kept).sum() / co_top.sum() * 100),
        "max_no2_left": float(old["NO2(GT)"].max()),
        "nmhc_missing_pct_before_ffill": float(df["nmhc_ref"].isna().mean() * 100),
        "no2_values_invented_by_ffill": int(df["no2_ref"].isna().sum()),
    })
    return info

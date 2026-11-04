"""Loading and auditing the raw data.

Nothing is imputed here. The file marks missing readings with -200; those
become NaN and stay NaN, so every later stage has to decide explicitly how to
handle them instead of inheriting a silent forward-fill.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config

RENAME = {
    "CO(GT)": "co_ref",
    "PT08.S1(CO)": "s1_co",
    "NMHC(GT)": "nmhc_ref",
    "C6H6(GT)": "benzene_ref",
    "PT08.S2(NMHC)": "s2_nmhc",
    "NOx(GT)": "nox_ref",
    "PT08.S3(NOx)": "s3_nox",
    "NO2(GT)": "no2_ref",
    "PT08.S4(NO2)": "s4_no2",
    "PT08.S5(O3)": "s5_o3",
    "T": "temp",
    "RH": "rh",
    "AH": "ah",
}


def load(path=None):
    """Return the hourly table indexed by timestamp, with -200 as NaN."""
    raw = pd.read_csv(path or config.DATA, sep=";", decimal=",")
    raw = raw.dropna(how="all", axis=1).dropna(how="all")
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H.%M.%S")
    df = raw.drop(columns=["Date", "Time"]).set_index(ts).rename(columns=RENAME)
    df = df.replace(-200, np.nan).astype(float)
    # A regular hourly index makes every shift() below mean "hours", not "rows".
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df.index.name = "time"
    return df


def _longest_gap(s):
    na = s.isna()
    if not na.any():
        return 0
    runs = na.groupby((na != na.shift()).cumsum()).agg(["first", "size"])
    return int(runs.loc[runs["first"], "size"].max())


def missingness_table(df):
    out = pd.DataFrame(
        {
            "missing_pct": (df.isna().mean() * 100).round(1),
            "longest_gap_hours": [_longest_gap(df[c]) for c in df.columns],
        }
    )
    out.index = [config.LABELS[c] for c in df.columns]
    return out


def missingness_by_month(df):
    m = df.isna().resample("MS").mean()
    m.index = m.index.strftime("%Y-%m")
    return m


def missingness_vs_hour(df, col="no2_ref"):
    """Chi-square test of whether a reading being missing depends on the hour.

    If it does, the data are not missing completely at random and dropping or
    filling rows changes the hourly mix of what is left.
    """
    table = pd.crosstab(df.index.hour, df[col].isna())
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    rate = df[col].isna().groupby(df.index.hour).mean()
    return {
        "column": col,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(np.sqrt(chi2 / n)),
        "worst_hour": int(rate.idxmax()),
        "worst_hour_missing_pct": float(rate.max() * 100),
        "best_hour_missing_pct": float(rate.min() * 100),
    }

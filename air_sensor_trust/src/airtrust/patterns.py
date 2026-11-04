"""Unsupervised analysis: contextual anomalies and daily pollution regimes."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import config, plotting as P

Z_THRESHOLD = 3.5  # Iglewicz & Hoaglin's cut-off for the modified z-score


def contextual_anomalies(df, col="no2_ref", days=28):
    """Flag hours that are unusual *for that hour and type of day*.

    The expected value is the median of the same clock hour on the previous
    `days` days of the same type (weekday or weekend), so it uses only the
    past. The scale is a trailing robust spread (MAD) of the residuals. A
    reading of 180 at 4 am can be an anomaly while 250 in a winter rush hour
    is not, which a fixed IQR cut-off cannot express.
    """
    y = df[col]
    wide = y.groupby([y.index.normalize(), y.index.hour]).first().unstack()
    weekend = wide.index.dayofweek >= 5
    expected = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for mask, n in [(~weekend, int(days * 5 / 7)), (weekend, int(days * 2 / 7))]:
        expected.loc[mask] = wide[mask].rolling(n, min_periods=max(3, n // 2)).median().shift(1).to_numpy()
    exp = expected.stack(future_stack=True)
    exp.index = exp.index.get_level_values(0) + pd.to_timedelta(exp.index.get_level_values(1), unit="h")
    exp = exp.reindex(df.index)
    resid = y - exp
    med = resid.rolling(days * 24, min_periods=7 * 24).median().shift(1)
    mad = (resid - med).abs().rolling(days * 24, min_periods=7 * 24).median().shift(1)
    z = 0.6745 * (resid - med) / mad
    return pd.DataFrame({"value": y, "expected": exp, "z": z, "anomaly": z.abs() > Z_THRESHOLD})


def compare_with_iqr(df, a, col="no2_ref"):
    y = df[col]
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr_flag = (y < q1 - 1.5 * (q3 - q1)) | (y > q3 + 1.5 * (q3 - q1))
    scored = a["z"].notna()
    ctx = a["anomaly"] & scored
    return {
        "hours_scored": int(scored.sum()),
        "contextual_anomalies": int(ctx.sum()),
        "contextual_high": int((ctx & (a["z"] > 0)).sum()),
        "contextual_low": int((ctx & (a["z"] < 0)).sum()),
        "iqr_outliers": int((iqr_flag & scored).sum()),
        "flagged_by_both": int((ctx & iqr_flag).sum()),
        "contextual_missed_by_iqr_pct": float((ctx & ~iqr_flag).sum() / max(ctx.sum(), 1) * 100),
        "iqr_outliers_that_are_normal_in_context_pct": float(
            (iqr_flag & scored & ~ctx).sum() / max((iqr_flag & scored).sum(), 1) * 100),
        "iqr_upper_fence": float(q3 + 1.5 * (q3 - q1)),
    }


def top_anomaly_days(a, n=8):
    d = a[a["anomaly"]].copy()
    d["day"] = d.index.normalize()
    d["hour"] = d.index.hour
    d["abs_z"] = d["z"].abs()
    # Describe each day by its single most unusual hour.
    peak = d.sort_values("abs_z").groupby("day").tail(1).set_index("day")
    g = pd.DataFrame({"anomalous_hours": d.groupby("day").size(), "peak_hour": peak["hour"], "peak_z": peak["z"],
                      "peak_value": peak["value"], "typical_at_peak": peak["expected"]})
    g = g.sort_values(["anomalous_hours", "peak_z"], ascending=False, key=abs).head(n).reset_index()
    g["weekday"] = g["day"].dt.day_name()
    g["day"] = g["day"].dt.strftime("%Y-%m-%d")
    return g


def daily_regimes(df, col="no2_ref"):
    """Cluster days by the *shape* of their 24-hour NO2 curve."""
    y = df[col]
    wide = y.groupby([y.index.normalize(), y.index.hour]).first().unstack()
    wide = wide[wide.notna().sum(axis=1) >= 22].interpolate(axis=1, limit=2, limit_direction="both").dropna()
    shape = wide.div(wide.mean(axis=1), axis=0)
    sil = {}
    for k in range(2, 7):
        labels = KMeans(k, n_init=10, random_state=config.SEED).fit_predict(shape)
        sil[k] = float(silhouette_score(shape, labels))
    k = max(sil, key=sil.get)
    km = KMeans(k, n_init=10, random_state=config.SEED).fit(shape)
    labels = pd.Series(km.labels_, index=shape.index)
    weekend = pd.Series(shape.index.dayofweek >= 5, index=shape.index)
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(labels, weekend))
    summary = pd.DataFrame({
        "days": labels.value_counts().sort_index(),
        "weekend_share_pct": weekend.groupby(labels).mean() * 100,
        "mean_no2": wide.mean(axis=1).groupby(labels).mean(),
        "peak_hour": pd.DataFrame(km.cluster_centers_, columns=shape.columns).idxmax(axis=1),
    })
    return {"k": k, "silhouette_by_k": sil, "summary": summary, "centers": km.cluster_centers_,
            "chi2_weekend": float(chi2), "p_weekend": float(p), "n_days": int(len(shape)),
            "cramers_v": float(np.sqrt(chi2 / len(shape) / max(min(k, 2) - 1, 1)))}


def fig_regimes(reg):
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    colors = [P.BLUE, P.ORANGE, P.AQUA, P.YELLOW, P.RED, P.MUTED]
    s = reg["summary"]
    for i, center in enumerate(reg["centers"]):
        label = f"Regime {i + 1}: {int(s.loc[i, 'days'])} days, {s.loc[i, 'weekend_share_pct']:.0f}% weekend"
        ax.plot(range(24), center, color=colors[i], label=label)
        P.end_label(ax, 23, center[-1], f"Regime {i + 1}")
    ax.axhline(1, color=P.AXIS, linewidth=1)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 26)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("NO2 relative to that day's mean")
    ax.legend(loc="upper left")
    ax.set_title("Daily NO2 regimes found by k-means on the shape of each day")
    return P.save(fig, "12_daily_regimes")


def run(df):
    a = contextual_anomalies(df)
    cmp = compare_with_iqr(df, a)
    top = top_anomaly_days(a)
    reg = daily_regimes(df)
    a.round(3).to_csv(config.PRED / "no2_contextual_anomalies.csv")
    top.to_csv(config.TAB / "anomaly_top_days.csv", index=False)
    reg["summary"].to_csv(config.TAB / "daily_regimes.csv", index_label="regime")
    summary = reg["summary"].reset_index(names="regime")
    summary["regime"] += 1
    return {
        "anomalies": cmp,
        "top_days": top.to_dict("records"),
        "regimes": {"k": reg["k"], "silhouette_by_k": reg["silhouette_by_k"], "n_days": reg["n_days"],
                    "chi2_weekend": reg["chi2_weekend"], "p_weekend": reg["p_weekend"],
                    "cramers_v": reg["cramers_v"], "summary": summary.to_dict("records")},
        "figures": {"regimes": fig_regimes(reg)},
    }

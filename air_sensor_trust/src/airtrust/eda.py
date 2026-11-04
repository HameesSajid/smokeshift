"""Exploratory analysis with statistical tests instead of eyeballed plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config, data, plotting as P


def _holm(pvals):
    """Holm step-down adjustment for a family of p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def weekday_weekend_tests(df, cols=("no2_ref", "co_ref", "nox_ref"), n_boot=5000):
    """Compare weekday and weekend daily means.

    Daily means are used rather than hours because consecutive hours are
    strongly autocorrelated and would overstate the sample size.
    """
    rng = np.random.default_rng(config.SEED)
    rows = []
    for c in cols:
        daily = df[c].resample("D").agg(["mean", "count"])
        daily = daily.loc[daily["count"] >= 18, "mean"]
        wk = daily[daily.index.dayofweek < 5].to_numpy()
        we = daily[daily.index.dayofweek >= 5].to_numpy()
        _, p = stats.mannwhitneyu(wk, we, alternative="two-sided")
        diff = wk.mean() - we.mean()
        boot = [rng.choice(wk, len(wk)).mean() - rng.choice(we, len(we)).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pooled = np.sqrt(((len(wk) - 1) * wk.var(ddof=1) + (len(we) - 1) * we.var(ddof=1)) / (len(wk) + len(we) - 2))
        rows.append(
            {
                "pollutant": config.LABELS[c],
                "weekday_mean": wk.mean(),
                "weekend_mean": we.mean(),
                "difference": diff,
                "ci95_low": lo,
                "ci95_high": hi,
                "cohens_d": diff / pooled,
                "n_weekdays": len(wk),
                "n_weekend_days": len(we),
                "p_mannwhitney": p,
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = _holm(out["p_mannwhitney"])
    return out


def acf(series, max_lag=168):
    return pd.Series([series.autocorr(l) for l in range(1, max_lag + 1)], index=range(1, max_lag + 1))


def sensor_pca(df):
    X = df[config.SENSORS].dropna()
    pca = PCA().fit(StandardScaler().fit_transform(X))
    return {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "pc1_loadings": dict(zip(config.SENSORS, [float(v) for v in pca.components_[0]])),
    }


def distribution_stats(df, cols=("no2_ref", "co_ref", "nox_ref")):
    rows = []
    for c in cols:
        x = df[c].dropna()
        rows.append(
            {
                "pollutant": config.LABELS[c],
                "mean": x.mean(),
                "median": x.median(),
                "p95": x.quantile(0.95),
                "max": x.max(),
                "skew_raw": stats.skew(x),
                "skew_log": stats.skew(np.log(x[x > 0])),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures


def fig_missingness(df):
    m = data.missingness_by_month(df) * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(m.T.values, aspect="auto", cmap=P.SEQ, vmin=0, vmax=100)
    ax.set_xticks(range(len(m.index)), m.index, rotation=45, ha="right")
    ax.set_yticks(range(len(m.columns)), [config.LABELS[c] for c in m.columns])
    ax.grid(False)
    for i in range(m.shape[1]):
        for j in range(m.shape[0]):
            v = m.values[j, i]
            if v >= 10:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 55 else P.INK)
    fig.colorbar(im, ax=ax, label="Missing readings (%)", pad=0.02)
    ax.set_title("Missing readings by month: the reference analysers fail far more often than the sensors")
    return P.save(fig, "01_missingness")


def fig_diurnal(df):
    """Hourly profile, weekday vs weekend, with a bootstrap band on the mean."""
    rng = np.random.default_rng(config.SEED)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, col in zip(axes, ["no2_ref", "co_ref"]):
        for label, mask, color in [("Weekdays", df.index.dayofweek < 5, P.BLUE),
                                   ("Weekends", df.index.dayofweek >= 5, P.ORANGE)]:
            sub = df.loc[mask, col].dropna()
            # Resample whole days so the band respects within-day correlation.
            by_day = sub.groupby([sub.index.date, sub.index.hour]).mean().unstack()
            mean = by_day.mean()
            boots = np.array([by_day.iloc[rng.integers(0, len(by_day), len(by_day))].mean().to_numpy()
                              for _ in range(400)])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            ax.fill_between(mean.index, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(mean.index, mean.values, color=color, label=label)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(0, 23)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel(config.LABELS[col])
        ax.set_title(config.LABELS[col].split(" (")[0])
    axes[0].legend(loc="upper left")
    fig.suptitle("Two rush-hour peaks on weekdays, flatter weekends (band = 95% bootstrap interval)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return P.save(fig, "02_diurnal_profiles")


def fig_acf(df):
    a = acf(df["no2_ref"])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(a.index, a.values, color=P.BLUE)
    ax.axhline(0, color=P.AXIS, linewidth=1)
    for lag in (24, 48, 72, 96, 120, 144, 168):
        ax.axvline(lag, color=P.GRID, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate(f"24 h: {a[24]:.2f}", (24, a[24]), xytext=(6, 10), textcoords="offset points", fontsize=9, color=P.INK2)
    ax.annotate(f"168 h (one week): {a[168]:.2f}", (168, a[168]), xytext=(-120, 14), textcoords="offset points",
                fontsize=9, color=P.INK2)
    ax.set_xticks(range(0, 169, 24))
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("NO2 remembers yesterday and last week: autocorrelation peaks every 24 hours")
    return P.save(fig, "03_no2_autocorrelation")


def fig_correlation(df):
    cols = ["co_ref", "nox_ref", "no2_ref", "benzene_ref"] + config.SENSORS + config.WEATHER
    c = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(c.values, cmap=P.DIV, vmin=-1, vmax=1)
    names = [config.LABELS[k].split(" (")[0] if k.endswith("_ref") or k in config.WEATHER else config.LABELS[k]
             for k in cols]
    ax.set_xticks(range(len(cols)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), names)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "-."), ha="center", va="center",
                    fontsize=7.5, color="white" if abs(v) > 0.7 else P.INK)
    fig.colorbar(im, ax=ax, label="Spearman correlation", pad=0.02)
    ax.set_title("Sensor 4 barely tracks the NO2 it is sold to measure")
    return P.save(fig, "04_correlation")


def run(df):
    tests = weekday_weekend_tests(df)
    dist = distribution_stats(df)
    tests.to_csv(config.TAB / "weekday_vs_weekend_tests.csv", index=False)
    dist.to_csv(config.TAB / "distribution_stats.csv", index=False)
    miss = data.missingness_table(df)
    miss.to_csv(config.TAB / "missingness.csv")
    a = acf(df["no2_ref"])
    return {
        "n_hours": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "missingness": miss.reset_index(names="variable").to_dict("records"),
        "missing_vs_hour": data.missingness_vs_hour(df),
        "weekday_weekend": tests.to_dict("records"),
        "distribution": dist.to_dict("records"),
        "acf_no2": {"lag1": float(a[1]), "lag24": float(a[24]), "lag168": float(a[168])},
        "sensor_pca": sensor_pca(df),
        "spearman_s4_vs_no2": float(df[["s4_no2", "no2_ref"]].corr(method="spearman").iloc[0, 1]),
        "spearman_s2_vs_benzene": float(df[["s2_nmhc", "benzene_ref"]].corr(method="spearman").iloc[0, 1]),
        "figures": {
            "missingness": fig_missingness(df),
            "diurnal": fig_diurnal(df),
            "acf": fig_acf(df),
            "correlation": fig_correlation(df),
        },
    }

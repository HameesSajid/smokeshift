"""What the data say before any model is fitted.

Each function answers one question with a statistic and, where it helps, a
figure:

    shift_by_region      how different was 2023, and where?
    paired_shift_test    is the change real at the level of individual monitors?
    distribution_distance   how far did the whole distribution move?
    weekday_effect_fdr   a multiple-testing example across 538 monitors
    correlogram          how far does air quality stay correlated?
    eof                  what are the dominant spatial patterns?
    plume_lags           does smoke arrive at upwind monitors first?
"""
import json
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from . import config, plotting as P
from .bench import Bench
from .spatial import distances_and_bearings

REGION_ORDER = ["Northeast", "Midwest", "South", "West"]
# Monitors along the path of the late-June 2023 plume, west to east.
CASE_SITES = {"Minneapolis": "27-053-0962", "Chicago": "17-031-4007", "Detroit": "26-163-0001",
              "Cleveland": "39-035-0073", "Pittsburgh": "42-003-0008", "Washington DC": "11-001-0043",
              "Philadelphia": "42-101-0057", "New York": "36-081-0124"}


def daily_means(bench, min_hours=18):
    """Daily mean per monitor ([monitors, days]); missing where fewer than `min_hours` readings exist."""
    v = bench.values.reshape(bench.n_sites, -1, 24)
    n = np.isfinite(v).sum(axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(v, axis=2)
    days = bench.hours[::24]
    return np.where(n >= min_hours, m, np.nan), days


def holm(p):
    """Holm step-down adjustment of a family of p-values (controls the chance of any false positive)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(p):
    """Adjusted p-values controlling the false discovery rate."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def shift_by_region(bench):
    """Mean, median, upper percentiles and share of hours over the threshold, by region and year."""
    year = bench.hours.year.to_numpy()
    region = bench.stations["region"].to_numpy()
    rows = []
    for r in REGION_ORDER:
        for y in config.YEARS:
            x = bench.values[region == r][:, year == y]
            x = x[np.isfinite(x)]
            rows.append({"region": r, "year": y, "monitors": int((region == r).sum()), "mean": x.mean(),
                         "median": np.median(x), "p99": np.percentile(x, 99), "p999": np.percentile(x, 99.9),
                         "pct_hours_over_usg": (x > config.USG).mean() * 100})
    return pd.DataFrame(rows)


def paired_shift_test(bench):
    """Per monitor: hours above the threshold in 2023 against each other year.

    The same monitors are measured in both years, so the test is paired
    (Wilcoxon signed-rank), which removes between-monitor differences.
    """
    st = bench.stations
    rows = []
    for r in REGION_ORDER:
        s = st[st.region == r]
        for other in (2021, 2022, 2024):
            a, b = s["hours_over_usg_2023"].to_numpy(float), s[f"hours_over_usg_{other}"].to_numpy(float)
            w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            rows.append({"region": r, "comparison": f"2023 vs {other}", "monitors": len(s),
                         "median_2023": np.median(a), "median_other": np.median(b),
                         "median_difference": np.median(a - b), "share_monitors_higher_in_2023": (a > b).mean(),
                         "p_value": w.pvalue})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out


def distribution_distance(bench, n=400_000):
    """How far the hourly distribution moved from the training period, by region."""
    rng = np.random.default_rng(config.SEED)
    lo, hi = bench.bounds("train")
    year = bench.hours.year.to_numpy()
    region = bench.stations["region"].to_numpy()
    rows = []
    for r in REGION_ORDER:
        ref = bench.values[region == r][:, lo:hi]
        ref = rng.choice(ref[np.isfinite(ref)], n)
        for y in config.TEST_YEARS:
            cur = bench.values[region == r][:, year == y]
            cur = rng.choice(cur[np.isfinite(cur)], n)
            rows.append({"region": r, "test_year": y, "ks_statistic": stats.ks_2samp(ref, cur).statistic,
                         "wasserstein": stats.wasserstein_distance(ref, cur),
                         "p999_ratio": np.percentile(cur, 99.9) / np.percentile(ref, 99.9)})
    return pd.DataFrame(rows)


def weekday_effect_fdr(bench):
    """Weekday vs weekend at every monitor: 538 tests, so control false discoveries."""
    daily, days = daily_means(bench)
    lo = bench.idx(config.TRAIN_END) // 24
    daily, dow = daily[:, :lo], days.dayofweek.to_numpy()[:lo]
    p, diff = [], []
    for row in daily:
        a, b = row[(dow < 5) & np.isfinite(row)], row[(dow >= 5) & np.isfinite(row)]
        p.append(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        diff.append(a.mean() - b.mean())
    p, diff = np.array(p), np.array(diff)
    bh = benjamini_hochberg(p)
    return {"monitors": int(len(p)), "significant_uncorrected": int((p < 0.05).sum()),
            "expected_false_positives_uncorrected": float(0.05 * len(p)),
            "significant_bonferroni": int((p < 0.05 / len(p)).sum()), "significant_fdr_5pct": int((bh < 0.05).sum()),
            "share_fdr_with_higher_weekdays": float((diff[bh < 0.05] > 0).mean()) if (bh < 0.05).any() else None,
            "median_weekday_minus_weekend": float(np.median(diff))}


def _anomalies(daily):
    """Log daily means, standardised per monitor; missing days become 0 (the mean)."""
    z = np.log1p(np.clip(daily, 0, None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z = (z - np.nanmean(z, axis=1, keepdims=True)) / np.nanstd(z, axis=1, keepdims=True)
    return z


def correlogram(bench, bins=np.arange(0, 2001, 100)):
    """Correlation of daily anomalies between monitor pairs, by distance."""
    daily, days = daily_means(bench)
    dist, _ = distances_and_bearings(bench.stations["lat"].to_numpy(), bench.stations["lon"].to_numpy())
    iu = np.triu_indices(bench.n_sites, 1)
    periods = {"Training period (2021 to mid 2022)": days < config.TRAIN_END,
               "Smoke season (May to July 2023)": (days >= "2023-05-01") & (days < "2023-08-01")}
    out = {"distance_km": ((bins[:-1] + bins[1:]) / 2).tolist()}
    for name, m in periods.items():
        c = np.ma.corrcoef(np.ma.masked_invalid(_anomalies(daily[:, m]))).filled(np.nan)[iu]
        which = np.digitize(dist[iu], bins) - 1
        out[name] = [float(np.nanmean(c[which == k])) for k in range(len(bins) - 1)]
    return out


def eof(bench, n_modes=10):
    """Principal components of the monitor-by-day anomaly matrix (EOF analysis)."""
    daily, days = daily_means(bench)
    z = np.nan_to_num(_anomalies(daily)).T                        # days x monitors
    pca = PCA(n_modes, random_state=config.SEED).fit(z)
    pcs = pca.transform(z)
    year = days.year.to_numpy()
    # Share of the 50 most extreme days of each leading mode that fall in each year.
    extreme = {f"mode{k + 1}": {int(y): float((year[np.argsort(-np.abs(pcs[:, k]))[:50]] == y).mean())
                                for y in config.YEARS} for k in range(3)}
    return {"explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "loadings": pca.components_[:3],
            "pcs": pcs[:, :3], "days": days, "extreme_days_by_year": extreme}


def plume_lags(bench, start="2023-06-25", end="2023-07-02", max_lag=96):
    """Hours by which each downwind city trails the first one, by cross-correlation."""
    site_row = {s: i for i, s in enumerate(bench.stations["site"])}
    a, b = bench.idx(start), bench.idx(end)
    names = list(CASE_SITES)
    dist, _ = distances_and_bearings(bench.stations["lat"].to_numpy(), bench.stations["lon"].to_numpy())
    ref = site_row[CASE_SITES[names[0]]]
    x = pd.Series(bench.values[ref, a:b]).interpolate(limit_direction="both").to_numpy()
    rows = []
    for name in names[1:]:
        j = site_row[CASE_SITES[name]]
        y = pd.Series(bench.values[j, a - max_lag:b + max_lag]).interpolate(limit_direction="both").to_numpy()
        corr = [np.corrcoef(x, y[max_lag + lag:max_lag + lag + len(x)])[0, 1] for lag in range(-max_lag, max_lag + 1)]
        k = int(np.argmax(corr))
        rows.append({"from": names[0], "to": name, "distance_km": float(dist[ref, j]), "lag_hours": k - max_lag,
                     "correlation_at_lag": float(corr[k]), "correlation_at_zero": float(corr[max_lag]),
                     "peak": float(np.nanmax(bench.values[j, a:b]))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures


def fig_map(bench):
    """Map of monitors coloured by extra unhealthy hours in 2023."""
    st = bench.stations
    extra = st["hours_over_usg_2023"] - (st["hours_over_usg_2021"] + st["hours_over_usg_2022"]) / 2
    fig, ax = plt.subplots(figsize=(10, 5.6))
    order = np.argsort(np.abs(extra.to_numpy()))
    sc = ax.scatter(st["lon"].to_numpy()[order], st["lat"].to_numpy()[order], c=extra.to_numpy()[order], cmap=P.DIV,
                    vmin=-300, vmax=300, s=26, edgecolor=P.SURFACE, linewidth=0.6)
    ax.set_aspect(1.28)
    ax.grid(False)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01, extend="both", label="Extra hours above 35.4 µg/m³")
    ax.set_title(f"Hours of unhealthy air in 2023 compared with the 2021-2022 average, at {len(st)} monitors")
    return P.save(fig, "01_map_shift")


def fig_shift(shift):
    """Bars of the share of hours over the threshold, by region and year."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharey=True)
    for ax, r in zip(axes, REGION_ORDER):
        s = shift[shift.region == r]
        colors = [P.ORANGE if y == 2023 else P.MUTED for y in s.year]
        bars = ax.bar(s.year.astype(str), s.pct_hours_over_usg, color=colors, width=0.62)
        for bar, v in zip(bars, s.pct_hours_over_usg):
            ax.annotate(f"{v:.1f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=P.INK2)
        ax.set_title(f"{r} ({int(s.monitors.iloc[0])} monitors)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Hours above 35.4 µg/m³ (%)")
    fig.suptitle("The test year is unlike the training years in the Northeast and Midwest",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return P.save(fig, "02_shift_by_region")


def fig_case(bench):
    """Hourly series in eight cities during the late-June 2023 plume."""
    site_row = {s: i for i, s in enumerate(bench.stations["site"])}
    a, b = bench.idx("2023-06-25"), bench.idx("2023-07-02")
    t = bench.hours[a:b]
    fig, axes = plt.subplots(len(CASE_SITES), 1, figsize=(9.5, 8.4), sharex=True, sharey=True)
    for ax, (name, site) in zip(axes, CASE_SITES.items()):
        y = bench.values[site_row[site], a:b]
        ax.fill_between(t, 0, np.nan_to_num(y), color=P.ORANGE, alpha=0.25, linewidth=0)
        ax.plot(t, y, color=P.ORANGE, linewidth=1.4)
        ax.axhline(config.USG, color=P.INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(name, (0.008, 0.78), xycoords="axes fraction", fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, 320)
        ax.set_yticks([0, 150, 300])
    axes[len(axes) // 2].set_ylabel("PM2.5 (µg/m³)")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].set_title("One plume, eight cities: 25 June to 1 July 2023, ordered west to east (dashed line = 35.4)")
    fig.subplots_adjust(hspace=0.12)
    return P.save(fig, "03_plume_case_study")


def fig_correlogram(cg):
    """Correlation against distance, for the training period and for the smoke season."""
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    for name, color in zip(list(cg)[1:], [P.MUTED, P.ORANGE]):
        ax.plot(cg["distance_km"], cg[name], color=color, marker="o", markersize=4, label=name)
    ax.axhline(0, color=P.AXIS, linewidth=1)
    ax.set_xlabel("Distance between two monitors (km)")
    ax.set_ylabel("Correlation of daily anomalies")
    ax.legend(loc="upper right")
    ax.set_title("Nearby monitors move together, and more so during the smoke season")
    return P.save(fig, "04_correlogram")


def fig_eof(bench, e):
    """Loadings of the leading principal component on a map, and its strength over time."""
    st = bench.stations
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    load = e["loadings"][0] * np.sign(np.median(e["loadings"][0]))
    lim = float(np.abs(load).max())
    sc = axes[0].scatter(st["lon"], st["lat"], c=load, cmap=P.DIV, vmin=-lim, vmax=lim, s=20, edgecolor=P.SURFACE,
                         linewidth=0.5)
    axes[0].set_aspect(1.28), axes[0].grid(False), axes[0].set_xticks([]), axes[0].set_yticks([])
    for s in axes[0].spines.values():
        s.set_visible(False)
    fig.colorbar(sc, ax=axes[0], shrink=0.8, pad=0.01, label="Loading")
    axes[0].set_title(f"Leading spatial pattern ({e['explained_variance_ratio'][0]:.0%} of variance)")
    pc = e["pcs"][:, 0] * np.sign(np.median(e["loadings"][0]))
    axes[1].plot(e["days"], pc, color=P.BLUE, linewidth=0.9)
    axes[1].axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"), color=P.ORANGE, alpha=0.08, linewidth=0)
    axes[1].set_ylabel("Strength of the pattern, by day")
    axes[1].set_title("When it was active (2023 shaded)")
    fig.tight_layout()
    return P.save(fig, "05_leading_pattern")


def run(bench=None, log=print):
    """Compute every statistic, draw the five figures and write the eda tables."""
    P.setup()
    bench = bench or Bench.load()
    config.TAB.mkdir(parents=True, exist_ok=True)
    shift, paired, dd = shift_by_region(bench), paired_shift_test(bench), distribution_distance(bench)
    fdr, cg, e, lags = weekday_effect_fdr(bench), correlogram(bench), eof(bench), plume_lags(bench)
    for name, t in [("shift_by_region", shift), ("paired_shift_test", paired), ("distribution_distance", dd),
                    ("plume_lags", lags)]:
        t.to_csv(config.TAB / f"eda_{name}.csv", index=False)
    summary = json.loads((config.PROCESSED / "panel_summary.json").read_text(encoding="utf-8"))
    out = {
        "panel": summary, "weekday_fdr": fdr, "correlogram": cg,
        "eof": {"explained_variance_ratio": e["explained_variance_ratio"],
                "extreme_days_by_year": e["extreme_days_by_year"]},
        "figures": {"map": fig_map(bench), "shift": fig_shift(shift), "case": fig_case(bench),
                    "correlogram": fig_correlogram(cg), "eof": fig_eof(bench, e)},
    }
    (config.TAB / "eda.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    ne = shift[(shift.region == "Northeast")].set_index("year")["pct_hours_over_usg"]
    log(f"  Northeast hours over the threshold: {ne[2022]:.2f}% in 2022, {ne[2023]:.2f}% in 2023, {ne[2024]:.2f}% in 2024")
    log(f"  weekday effect: {fdr['significant_uncorrected']} monitors at p<0.05, "
        f"{fdr['significant_fdr_5pct']} after FDR control, {fdr['significant_bonferroni']} after Bonferroni")
    log("  plume lags from Minneapolis: " + ", ".join(f"{r.to} {r.lag_hours:+d} h" for r in lags.itertuples()))
    return out

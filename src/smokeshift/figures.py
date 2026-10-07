"""Result figures, drawn from the tables that the evaluation stages wrote."""
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, conformal, plotting as P, store
from .bench import Bench, H

NAIVE = ["Persistence", "Seasonal naive", "Damped persistence"]
# Models shown in line charts; the rest appear in tables only, to keep charts readable.
HEADLINE = ["Damped persistence", "XGBoost + neighbours", "Transformer + neighbours", "Chronos-2", "TimesFM",
            "Ensemble", "Ensemble + foundation"]


def _order(models):
    """Sort model names into the fixed display order used in every chart."""
    known = list(P.MODEL_STYLE)
    return sorted(models, key=lambda m: known.index(m) if m in known else len(known))


def fig_leaderboard():
    """Relative MAE with day-block bootstrap intervals, in three forecast-time strata."""
    b = pd.read_csv(config.TAB / "relative_mae_bootstrap.csv")
    slices = ["All hours", "Calm at issue", "Smoke at issue"]
    models = _order(b.model.unique())[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.42 * len(models) + 1.6), sharey=True)
    for ax, s in zip(axes, slices):
        t = b[b.slice == s].set_index("model").reindex(models)
        for i, (m, r) in enumerate(t.iterrows()):
            color = P.style(m)[0]
            ax.plot([r.ci_low, r.ci_high], [i, i], color=color, linewidth=2.2, solid_capstyle="butt")
            ax.scatter([r.relative_mae], [i], color=color, s=44, zorder=3, edgecolor=P.SURFACE, linewidth=1)
            ax.annotate(f"{r.relative_mae:.2f}", (max(r.ci_high, r.relative_mae), i), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=P.INK2)
        ax.axvline(1, color=P.INK2, linewidth=1)
        ax.set_yticks(range(len(models)), models)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("MAE relative to damped persistence")
        ax.set_title(s)
        lo, hi = t.ci_low.min(), t.ci_high.max()
        ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.16 * (hi - lo))
    fig.suptitle("Lower is better. Bars are 95% intervals from resampling whole days.", x=0.01, y=0.985, ha="left",
                 fontsize=10, color=P.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return P.save(fig, "06_leaderboard")


def fig_horizon():
    """MAE against hours ahead, for all hours and for forecasts issued in smoke."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    for ax, (fname, title) in zip(axes, [("mae_by_horizon_all_hours", "All hours"),
                                         ("mae_by_horizon_smoke_at_issue", "Smoke at issue")]):
        t = pd.read_csv(config.TAB / f"{fname}.csv", index_col=0)
        for m in [m for m in HEADLINE if m in t.columns]:
            color, ls = P.style(m)
            ax.plot(t.index, t[m], color=color, linestyle=ls, label=m, linewidth=1.8)
        ax.set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
        ax.set_xlabel("Hours ahead")
        ax.set_ylabel("MAE (µg/m³)")
        ax.set_ylim(bottom=0)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    return P.save(fig, "07_error_by_horizon")


def fig_dilemma():
    """Bias by what was later observed: every model under-predicts the worst hours."""
    t = pd.read_csv(config.TAB / "bias_by_outcome.csv")
    models = [m for m in HEADLINE + ["Persistence"] if m in t.model.unique()]
    levels = list(dict.fromkeys(t.level))
    fig, ax = plt.subplots(figsize=(8.6, 4))
    for m in models:
        color, ls = P.style(m)
        s = t[t.model == m].set_index("level").reindex(levels)
        ax.plot(range(len(levels)), s.bias, color=color, linestyle=ls, marker="o", markersize=4, label=m, linewidth=1.8)
    ax.axhline(0, color=P.INK2, linewidth=1)
    ax.set_xticks(range(len(levels)), levels)
    ax.set_xlabel("PM2.5 that was then observed (µg/m³)")
    ax.set_ylabel("Mean forecast minus observed (µg/m³)")
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("Looking back from bad hours, every forecast was too low")
    return P.save(fig, "08_bias_by_outcome")


def fig_spatial_ablation():
    """Own history against own history plus neighbours, for each model family."""
    s = pd.read_csv(config.TAB / "scores_by_regime.csv")
    a = pd.read_csv(config.TAB / "alerts.csv")
    pairs = [(m, f"{m} + neighbours") for m in ("XGBoost", "Transformer")
             if f"{m} + neighbours" in s.model.unique() and m in s.model.unique()]
    panels = [("MAE, all hours (µg/m³)", lambda m: s[(s.model == m) & (s.slice == "All hours")].mae.iloc[0]),
              ("Quantile loss (WQL), all hours", lambda m: s[(s.model == m) & (s.slice == "All hours")].wql.iloc[0]),
              ("MAE, smoke at issue (µg/m³)", lambda m: s[(s.model == m) & (s.slice == "Smoke at issue")].mae.iloc[0]),
              ("Onset warning skill (PR-AUC, day 1)",
               lambda m: a[(a.method == m) & a.day.str.startswith("Day 1") & a.slice.str.startswith("Onsets")].pr_auc.iloc[0])]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
    for ax, (title, get) in zip(axes, panels):
        x = 0
        ticks = []
        for own, nb in pairs:
            color = P.style(own)[0]
            vals = [get(own), get(nb)]
            bars = ax.bar([x, x + 1], vals, color=[color, color], width=0.8, alpha=1)
            bars[0].set_alpha(0.45)
            for bx, v in zip([x, x + 1], vals):
                ax.annotate(f"{v:.3g}", (bx, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5,
                            color=P.INK2)
            ticks += [(x, "own"), (x + 1, "+ nbrs")]
            ax.annotate(own, (x + 0.5, 0), xytext=(0, -30), textcoords="offset points", ha="center", fontsize=9,
                        annotation_clip=False)
            x += 2.6
        ax.set_xticks([t for t, _ in ticks], [lab for _, lab in ticks], fontsize=8.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10)
    fig.suptitle("What knowing the neighbours adds (pale = own history only, solid = with neighbours)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return P.save(fig, "09_spatial_ablation")


def fig_coverage(model):
    """Monthly coverage of raw, static-conformal and adaptive-conformal 80% intervals for one model."""
    t = pd.read_csv(config.TAB / "intervals_monthly.csv")
    t = t[t.model == model]
    months = sorted(t.month.unique())
    fig, ax = plt.subplots(figsize=(10, 3.9))
    for method, color, ls in [("Raw", P.MUTED, "--"), ("Static conformal", P.ORANGE, "-"),
                              ("Adaptive conformal", P.BLUE, "-")]:
        s = t[t.method == method].set_index("month").reindex(months)
        ax.plot(range(len(months)), s.coverage * 100, color=color, linestyle=ls, marker="o", markersize=4, label=method)
    ax.axhline(80, color=P.INK, linewidth=1)
    ax.annotate("target 80%", (0, 80), xytext=(-2, -12), textcoords="offset points", fontsize=9, color=P.INK2)
    ax.set_xticks(range(len(months))[::2], months[::2], rotation=45, ha="right")
    ax.set_ylim(60, 90)
    ax.set_ylabel("Hours inside the 80% interval (%)")
    ax.legend(loc="lower right", ncol=3)
    ax.set_title(f"Interval coverage by month, {model}")
    return P.save(fig, "10_interval_coverage")


def fig_alerts():
    """PR-AUC of every alert method, for all days and for onsets."""
    a = pd.read_csv(config.TAB / "alerts.csv")
    a = a[a.day.str.startswith("Day 1")]
    order = a[a.slice == "All days"].sort_values("pr_auc").method.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.36 * len(order) + 1.4), sharey=True)
    for ax, s in zip(axes, ["All days", "Onsets (clean at issue)"]):
        t = a[a.slice == s].set_index("method").reindex(order)
        colors = [P.style(m.replace("Classifier", "XGBoost").replace("Last 24 h mean (persistence)", "Persistence"))[0]
                  for m in order]
        ax.barh(range(len(order)), t.pr_auc, color=colors, height=0.62)
        for i, v in enumerate(t.pr_auc):
            ax.annotate(f"{v:.2f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5,
                        color=P.INK2)
        ax.axvline(t.base_rate.iloc[0], color=P.INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_yticks(range(len(order)), order)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("PR-AUC (dashed line = chance)")
        ax.set_title(f"{s}: {int(t.events.iloc[0]):,} events in {int(t.n.iloc[0]):,} monitor-days")
    fig.tight_layout()
    return P.save(fig, "11_alerts")


def fig_shap(model="XGBoost + neighbours", top=12):
    """Top features by mean absolute SHAP value, for clean and for elevated target hours."""
    info = json.loads((config.TAB / "gbm_training.json").read_text(encoding="utf-8"))[model]["shap"]
    df = pd.DataFrame({"feature": info["features"], "clean": info["clean"], "elevated": info["elevated"]})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, col, color, n in zip(axes, ["clean", "elevated"], [P.BLUE, P.ORANGE], [info["n_clean"], info["n_elevated"]]):
        t = df.sort_values(col).tail(top)
        is_nb = t.feature.str.contains("nb_|ring_|quad_|share_elevated")
        for bar, nb in zip(ax.barh(range(top), t[col], color=color, height=0.62), is_nb):
            bar.set_alpha(1.0 if nb else 0.45)
        ax.set_yticks(range(top), t.feature)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Mean |SHAP| on the median forecast (µg/m³)")
        ax.set_title(f"Target hour was {col} ({n:,} rows)")
    fig.suptitle(f"What {model} relies on (solid bars are neighbour features)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return P.save(fig, "12_shap")


def fig_case_forecast(bench, site="42-101-0057", city="Philadelphia", start="2023-06-04", days=7):
    """Day-ahead forecasts through the 6-8 June 2023 episode, for the headline models."""
    row = list(bench.stations["site"]).index(site)
    t_idx = bench.issue_times("test")
    d0 = int(np.searchsorted(t_idx, bench.idx(start)))
    models = [m for m in ["Damped persistence", "XGBoost + neighbours", "Transformer + neighbours", "Chronos-2",
                          "TimesFM"] if m in store.available()]
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 1.75 * len(models) + 0.8), sharex=True, sharey=True)
    a, b = t_idx[d0], t_idx[d0 + days - 1] + H
    obs_t, obs = bench.hours[a:b + 1], bench.values[row, a:b + 1]
    for ax, m in zip(np.atleast_1d(axes), models):
        f = store.load(m)
        lo, hi = conformal.raw_interval(f)
        ax.plot(obs_t, obs, color=P.INK, linewidth=1.3, label="Observed")
        color = P.style(m)[0]
        for k in range(days):
            d = d0 + k
            tt = bench.hours[t_idx[d] + 1:t_idx[d] + 1 + 24]
            if f["quantiles"] is not None:
                ax.fill_between(tt, lo[row, d, :24], hi[row, d, :24], color=color, alpha=0.2, linewidth=0)
            ax.plot(tt, f["point"][row, d, :24], color=color, linewidth=1.6, label="Forecast, next 24 h" if k == 0 else None)
            ax.axvline(bench.hours[t_idx[d]], color=P.GRID, linewidth=1)
        ax.axhline(config.USG, color=P.INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(m, (0.008, 0.82), xycoords="axes fraction", fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, max(60.0, float(np.nanmax(obs)) * 1.1))
    np.atleast_1d(axes)[0].legend(loc="upper right", ncol=2)
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    np.atleast_1d(axes)[len(models) // 2].set_ylabel("PM2.5 (µg/m³)")
    np.atleast_1d(axes)[0].set_title(f"{city}, June 2023: each morning's forecast for the next 24 hours "
                                     "(band = raw 80% interval)")
    fig.subplots_adjust(hspace=0.1)
    return P.save(fig, "13_case_forecasts")


def run(bench=None, log=print):
    """Draw every result chart and record the file names in outputs/tables/figures.json."""
    P.setup()
    bench = bench or Bench.load()
    s = pd.read_csv(config.TAB / "scores_by_regime.csv")
    best = s[s.slice == "All hours"].sort_values("mae").model.iloc[0]
    out = {"leaderboard": fig_leaderboard(), "horizon": fig_horizon(), "dilemma": fig_dilemma(),
           "spatial": fig_spatial_ablation(), "coverage": fig_coverage(best), "alerts": fig_alerts(),
           "shap": fig_shap(), "case": fig_case_forecast(bench)}
    (config.TAB / "figures.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("  wrote " + ", ".join(out.values()))
    return out

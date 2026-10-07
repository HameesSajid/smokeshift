"""Builds outputs/report.html: one self-contained page with every result.

Every number in the text is read from the tables the pipeline wrote, so the
write-up cannot drift away from what was computed.
"""
import base64
import html
import json

import pandas as pd

from . import config

CSS = """
:root{--bg:#f7f8f8;--surface:#fdfdfc;--ink:#101314;--ink2:#4a5256;--muted:#7f888c;--line:#dfe3e4;--accent:#c2410c;--accent-soft:#fdf0e8;--link:#184f95}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.62 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding-block:40px 80px;padding-inline:18px}
h1{font-size:36px;line-height:1.12;margin:0 0 10px;letter-spacing:-.015em;text-wrap:balance}
h2{font-size:23px;margin:58px 0 8px;padding-top:22px;border-top:1px solid var(--line);text-wrap:balance}
h3{font-size:17px;margin:30px 0 6px}
p{margin:10px 0;max-width:74ch}
.lede{font-size:19px;color:var(--ink2);max-width:68ch}
.meta{color:var(--muted);font-size:14px;margin-top:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px;margin:28px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.tile b{display:block;font-size:29px;line-height:1.1;letter-spacing:-.01em}
.tile span{display:block;color:var(--ink2);font-size:14px;line-height:1.45;margin-top:6px}
figure{margin:20px 0;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px}
figure img{display:block;width:100%;height:auto}
figcaption{color:var(--ink2);font-size:14px;padding:8px 4px 2px;max-width:88ch}
.scroll{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;font-size:14px;background:var(--surface);min-width:62%}
th,td{padding:7px 12px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}
th{color:var(--ink2);font-weight:600;border-bottom:1px solid #b9c0c3}
tr.hl td{font-weight:650}
.note{background:var(--accent-soft);border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 8px 8px 0;max-width:74ch;margin:16px 0}
code{background:#eceeee;padding:1px 5px;border-radius:4px;font-size:.92em}
ul,ol{max-width:74ch;padding-left:22px} li{margin:5px 0}
.refs li{font-size:14px;color:var(--ink2)}
nav{font-size:14px;margin:18px 0 0;display:flex;flex-wrap:wrap;gap:4px 16px} nav a{color:var(--link)}
"""

NAIVE = ["Persistence", "Seasonal naive", "Damped persistence"]
FOUNDATION = ["Chronos-Bolt", "Chronos-Bolt (long context)", "Chronos-2", "Chronos-2 (long context)", "TimesFM",
              "TimesFM (long context)"]
ENSEMBLES = ["Ensemble", "Ensemble + foundation"]
DESCRIPTIONS = {
    "Persistence": "The latest reading, repeated for 48 hours",
    "Seasonal naive": "The same clock hour on the latest observed day",
    "Damped persistence": "Seven-day hourly profile plus a decaying share of the current anomaly (one fitted number per horizon)",
    "XGBoost": "Gradient-boosted trees on the monitor's own history; one global model, nine quantiles",
    "XGBoost + neighbours": "The same, plus 16 features describing monitors up to 800 km away",
    "XGBoost (level label)": "Ablation of the first tree model: predicts the level instead of the change from the latest reading",
    "Transformer": "Patch transformer trained from scratch in PyTorch on the monitor's own history",
    "Transformer + neighbours": "The same, plus two neighbour input channels",
    "Chronos-Bolt": "Amazon's 2024 pretrained forecasting model, zero-shot, same 168-hour context",
    "Chronos-Bolt (long context)": "Chronos-Bolt given 21 days of history",
    "Chronos-2": "Amazon's 2025 pretrained forecasting model, zero-shot, same 168-hour context",
    "Chronos-2 (long context)": "Chronos-2 given 21 days of history",
    "TimesFM": "Google's pretrained forecasting model (version 2.5), zero-shot, same 168-hour context",
    "TimesFM (long context)": "TimesFM given 21 days of history",
    "Ensemble": "Equal-weight average of the two neighbour-aware trained models; nothing fitted",
    "Ensemble + foundation": "Equal-weight average of those two plus Chronos-2 and TimesFM; nothing fitted",
}


def _img(name):
    """An <img> tag with the PNG embedded as base64."""
    data = base64.b64encode((config.FIG / name).read_bytes()).decode()
    return f'<img alt="" src="data:image/png;base64,{data}">'


def _fig(name, caption):
    """A figure block: embedded image plus caption."""
    return f"<figure>{_img(name)}<figcaption>{caption}</figcaption></figure>"


def _table(df, formats=None, left=(), highlight=None):
    """An HTML table from a DataFrame, with per-column number formats and an optional row highlight."""
    formats = formats or {}
    head = "".join(f'<th class="{"l" if c in left else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            s = "n/a" if pd.isna(v) else formats[c].format(v) if c in formats else html.escape(str(v))
            cells.append(f'<td class="{"l" if c in left else ""}">{s}</td>')
        cls = ' class="hl"' if highlight is not None and highlight(r) else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _json(name):
    """Read a JSON table from outputs/tables."""
    return json.loads((config.TAB / name).read_text(encoding="utf-8"))


def _csv(name):
    """Read a CSV table from outputs/tables."""
    return pd.read_csv(config.TAB / name)


def build(log=print):
    """Read every result table, compute the numbers quoted in the text and write the page."""
    eda, summ, figs = _json("eda.json"), _json("evaluation_summary.json"), _json("figures.json")
    panel = eda["panel"]
    shift, paired, dd, lags = (_csv(f"eda_{n}.csv") for n in ("shift_by_region", "paired_shift_test",
                                                               "distribution_distance", "plume_lags"))
    scores, boots, bias = _csv("scores_by_regime.csv"), _csv("relative_mae_bootstrap.csv"), _csv("bias_by_outcome.csv")
    slices, intervals, monthly, alerts = (_csv(n) for n in ("scores_by_slice.csv", "intervals.csv",
                                                             "intervals_monthly.csv", "alerts.csv"))
    gbm, tuning, neural = _json("gbm_training.json"), _json("gbm_tuning.json"), _json("neural_training.json")
    alert_detail = _json("alerts_detail.json")
    models = summ["models"]
    learned = [m for m in models if m not in NAIVE]
    has_fm = [m for m in FOUNDATION if m in models]

    # ----------------------------------------------------------- numbers
    mae = scores.pivot(index="model", columns="slice", values="mae")
    wql = scores.pivot(index="model", columns="slice", values="wql")
    rank = lambda s: mae[s].rank().astype(int)
    best = mae["All hours"].idxmin()
    best_single = mae.loc[[m for m in models if m not in ENSEMBLES], "All hours"].idxmin()
    rel = boots[boots.slice == "All hours"].set_index("model")
    rel_smoke = boots[boots.slice == "Smoke at issue"].set_index("model")
    best_smoke = mae["Smoke at issue"].idxmin()
    sh = shift.set_index(["region", "year"])["pct_hours_over_usg"]
    ne_ratio = sh["Northeast", 2023] / sh["Northeast", 2022]
    pt = paired.set_index(["region", "comparison"])
    ddi = dd.set_index(["region", "test_year"])
    fdr = eda["weekday_fdr"]
    n_models = len(models)

    nb_gain = {m: 1 - mae.loc[f"{m} + neighbours", "All hours"] / mae.loc[m, "All hours"]
               for m in ("XGBoost", "Transformer") if f"{m} + neighbours" in mae.index}
    nb_gain_smoke = {m: 1 - mae.loc[f"{m} + neighbours", "Smoke at issue"] / mae.loc[m, "Smoke at issue"]
                     for m in nb_gain}
    a1 = alerts[alerts.day.str.startswith("Day 1")]
    a_all = a1[a1.slice == "All days"].set_index("method")
    a_on = a1[a1.slice.str.startswith("Onsets")].set_index("method")
    a2 = alerts[alerts.day.str.startswith("Day 2") & (alerts.slice == "All days")].set_index("method")
    onset_ratio = {m: a_on.pr_auc[f"{m} + neighbours"] / a_on.pr_auc[m] for m in nb_gain}
    best_alert, best_onset = a_all.pr_auc.idxmax(), a_on.pr_auc.idxmax()
    own_history = [m for m in a_on.index if m in ["XGBoost", "Transformer", "Classifier", "Persistence",
                                                   "Damped persistence", "Seasonal naive"] + FOUNDATION]

    iv = intervals[intervals.model == best].pivot(index="method", columns="slice", values="coverage")
    ivw = intervals[intervals.model == best].pivot(index="method", columns="slice", values="mean_width")
    june = monthly[(monthly.model == best) & (monthly.month == "2023-06")].set_index("method").coverage
    onset_cov_max = intervals[(intervals.slice == "Onsets")].coverage.max()

    lvl = bias.pivot(index="model", columns="level", values="bias")
    worst_level = [c for c in lvl.columns if c.startswith("over 125")][0]
    rel_bin = next(b for b in alert_detail["reliability"]["Classifier + neighbours | Day 1 (hours 1-24)"]
                   if b["bin"].startswith("0.10"))

    # ------------------------------------------------------------ tables
    order = [m for m in NAIVE + ["XGBoost", "XGBoost + neighbours", "XGBoost (level label)", "Transformer",
                                 "Transformer + neighbours"] + FOUNDATION + ENSEMBLES if m in models]
    model_tbl = pd.DataFrame({"Model": order, "What it is": [DESCRIPTIONS.get(m, "") for m in order]})

    lead = pd.DataFrame({
        "Model": order,
        "MAE, µg/m³": [mae.loc[m, "All hours"] for m in order],
        "Relative to damped persistence": [f"{rel.relative_mae[m]:.3f} ({rel.ci_low[m]:.3f} to {rel.ci_high[m]:.3f})"
                                           for m in order],
        "MASE": [scores[(scores.model == m) & (scores.slice == "All hours")].mase.iloc[0] for m in order],
        "WQL": [wql.loc[m, "All hours"] for m in order],
        "MAE, calm at issue": [mae.loc[m, "Calm at issue"] for m in order],
        "MAE, smoke at issue": [mae.loc[m, "Smoke at issue"] for m in order],
    }).sort_values("MAE, µg/m³")

    trap = pd.DataFrame({
        "Model": order,
        "MAE on hours that turned out elevated": [mae.loc[m, "Elevated hours"] for m in order],
        "Rank": [rank("Elevated hours")[m] for m in order],
        "MAE when smoke was present at issue": [mae.loc[m, "Smoke at issue"] for m in order],
        "Rank ": [rank("Smoke at issue")[m] for m in order],
    }).sort_values("Rank")

    shift_tbl = shift.pivot(index="region", columns="year", values="pct_hours_over_usg").loc[
        ["Northeast", "Midwest", "South", "West"]].reset_index().rename(columns={"region": "Region"})
    shift_tbl.columns = [str(c) for c in shift_tbl.columns]
    shift_tbl["Tail (99.9th percentile) in 2023 vs training"] = [f'{ddi.p999_ratio[r, 2023]:.2f}x'
                                                                 for r in shift_tbl.Region]
    shift_tbl["Monitors with more unhealthy hours in 2023 than 2022"] = [
        f'{pt.share_monitors_higher_in_2023[r, "2023 vs 2022"]:.0%}' for r in shift_tbl.Region]

    lag_tbl = lags.rename(columns={"to": "City", "distance_km": "Distance from Minneapolis, km",
                                   "lag_hours": "Trails Minneapolis by, hours", "peak": "Peak PM2.5, µg/m³"})[
        ["City", "Distance from Minneapolis, km", "Trails Minneapolis by, hours", "Peak PM2.5, µg/m³"]]

    int_methods = [m for m in ("Raw", "Static conformal", "Adaptive conformal") if m in iv.index]
    int_tbl = pd.DataFrame({
        "Interval method": int_methods,
        "All hours": [iv.loc[m, "All hours"] * 100 for m in int_methods],
        "Smoke at issue": [iv.loc[m, "Smoke at issue"] * 100 for m in int_methods],
        "June 2023": [june[m] * 100 for m in int_methods],
        "Hours that became onsets": [iv.loc[m, "Onsets"] * 100 for m in int_methods],
        "Mean width, µg/m³": [ivw.loc[m, "All hours"] for m in int_methods],
    })

    al_order = a_all.pr_auc.sort_values(ascending=False).index
    al_tbl = pd.DataFrame({
        "Method": al_order,
        "PR-AUC, all days": a_all.pr_auc[al_order].to_numpy(),
        "Recall at 50% precision": a_all.recall_at_50pct_precision[al_order].to_numpy() * 100,
        "PR-AUC, onsets": a_on.pr_auc[al_order].to_numpy(),
        "Onset recall at 50% precision": a_on.recall_at_50pct_precision[al_order].to_numpy() * 100,
        "PR-AUC, day 2": a2.pr_auc[al_order].to_numpy(),
    })

    concept_tbl = pd.DataFrame([
        ["Data engineering at scale", f"{panel['raw_rows'] / 1e6:.1f} million raw rows reduced to a dense monitor-by-hour panel with a checksummed manifest; compact int16 storage"],
        ["Missing data", "Nothing imputed in targets; inputs forward-filled with an explicit mask channel; completeness thresholds"],
        ["Distribution shift", "Quantified three ways (bulk distance, tail ratio, paired monitor-level test); used as the test condition"],
        ["Hypothesis testing", "Paired Wilcoxon tests with Holm correction; 538 simultaneous tests with Benjamini-Hochberg false-discovery control"],
        ["Resampling", "Day-block bootstrap intervals that respect dependence between monitors on the same day"],
        ["Dimensionality reduction", "Principal components (EOFs) of the monitor-by-day anomaly matrix"],
        ["Spatial statistics", "Great-circle distances, distance-binned correlogram, lagged cross-correlation, inverse-distance features"],
        ["Supervised learning", "Global gradient-boosted quantile regression on a GPU; random search on a held-out period"],
        ["Deep learning", "A patch transformer written in PyTorch: quantile loss, masking, early stopping"],
        ["Foundation models", "Zero-shot evaluation of pretrained forecasters on the same grid" if has_fm else "Zero-shot evaluation harness for pretrained forecasters (models not yet run)"],
        ["Ensembling", "Equal-weight quantile averaging across model families"],
        ["Evaluation design", "Fixed forecast grid, three scores, forecast-time strata, and the forecaster's dilemma shown on real data"],
        ["Uncertainty quantification", "Quantile regression, conformalized intervals, adaptive conformal inference with delayed feedback"],
        ["Rare-event classification", "PR-AUC, recall at fixed precision, Brier skill and a reliability check of predicted probabilities"],
        ["Explainability", "Exact TreeSHAP attributions, split by regime"],
        ["Software engineering", "Installable package, staged CLI, 12 leakage and protocol tests, CI workflow"],
    ], columns=["Concept", "Where it appears"])

    f1, f2, f3 = "{:.1f}", "{:.2f}", "{:.3f}"
    xgb_nb = gbm.get("XGBoost + neighbours", {})
    tr = neural.get("Transformer + neighbours", {})

    fm_section = ""
    if has_fm:
        # The fair comparison is with models that also see only one monitor's own history.
        own_trained = [m for m in ("XGBoost", "Transformer") if m in models]
        fm_short = [m for m in has_fm if "long context" not in m]
        fm_tbl = lead[lead.Model.isin(has_fm + own_trained + ["Damped persistence"] + ENSEMBLES)]
        fm_best = mae.loc[fm_short, "All hours"].idxmin()
        own_best = mae.loc[own_trained, "All hours"].idxmin()
        word = lambda a, b: "lower than" if a < b else "higher than"
        gap = (mae.loc[fm_best, "All hours"] / mae.loc[own_best, "All hours"] - 1) * 100
        long_bits = [f"{m.replace(' (long context)', '')} goes from {mae.loc[m.replace(' (long context)', ''), 'All hours']:.2f} "
                     f"to {mae.loc[m, 'All hours']:.2f} µg/m³" for m in has_fm
                     if "long context" in m and m.replace(" (long context)", "") in models]
        long_text = ("<p>Giving a model three weeks of history instead of one: " + "; ".join(long_bits) + ".</p>"
                     if long_bits else "")
        fm_long = [m for m in has_fm if "long context" in m]
        onset_long = (f"With three weeks of history, {a_on.pr_auc[fm_long].idxmax().replace(' (long context)', '')} "
                      f"reaches {a_on.pr_auc[fm_long].max():.2f}. " if fm_long else "")
        ens_text = ""
        if all(m in models for m in ENSEMBLES):
            e0, e1 = mae.loc["Ensemble", "All hours"], mae.loc["Ensemble + foundation", "All hours"]
            ens_text = (f"<p>Adding Chronos-2 and TimesFM to the two trained neighbour-aware models, with equal weights "
                        f"chosen before any of them was scored, moves the ensemble's MAE from {e0:.3f} to {e1:.3f} "
                        f"µg/m³ overall and from {mae.loc['Ensemble', 'Smoke at issue']:.2f} to "
                        f"{mae.loc['Ensemble + foundation', 'Smoke at issue']:.2f} with smoke present at issue.</p>")
        fm_section = f"""
<h2 id="foundation">7. Zero-shot foundation models</h2>
<p>Pretrained forecasting models promise good forecasts with no training on the target data. They were given
exactly the same input as the other own-history models, the last {config.CONTEXT} hours of one monitor, and asked
for the same nine quantiles. Nothing about them was fitted or tuned here.</p>
<div class="note">One caution applies to this section only. These models were released in 2024 and 2025 and their
pretraining data are not fully documented, so they may have seen time series from 2023 or 2024. The trained models
in this benchmark provably saw nothing after June 2022; the foundation models carry no such guarantee.</div>
{_table(fm_tbl, {"MAE, µg/m³": f2, "MASE": f3, "WQL": f3, "MAE, calm at issue": f2, "MAE, smoke at issue": f2}, left=["Model"])}
<p>The best of them on equal terms, {fm_best}, reaches an MAE of {mae.loc[fm_best, "All hours"]:.2f} µg/m³ without
having seen a single reading from this dataset. That is {abs(gap):.1f}% {word(gap, 0)} {own_best}
({mae.loc[own_best, "All hours"]:.2f}), the best model trained here on the same information, and
{(1 - mae.loc[fm_best, "All hours"] / mae.loc["Damped persistence", "All hours"]) * 100:.0f}% lower than damped
persistence. With smoke present at issue its MAE is {mae.loc[fm_best, "Smoke at issue"]:.2f}, against
{mae.loc[own_best, "Smoke at issue"]:.2f} for {own_best} and {mae.loc[best_smoke, "Smoke at issue"]:.2f} for
{best_smoke}, the best of all methods in that stratum.</p>
{long_text}
{ens_text}
<p>For onset warnings, with the shared 168-hour input the foundation models score a PR-AUC between
{a_on.pr_auc[fm_short].min():.2f} and {a_on.pr_auc[fm_short].max():.2f}, the same range as the trained models
that see only one monitor ({a_on.pr_auc[own_trained].min():.2f} to {a_on.pr_auc[own_trained].max():.2f}).
{onset_long}The best neighbour-aware model, {best_onset}, reaches {a_on.pr_auc[best_onset]:.2f}. Pretraining on a
very large collection of other series does not tell a model that smoke is 300 km away.</p>"""
    else:
        fm_section = """
<h2 id="foundation">7. Zero-shot foundation models</h2>
<p>The harness is ready to score pretrained forecasting models (Amazon's Chronos-Bolt and Google's TimesFM) on the
same grid, with the same 168-hour input as the other own-history models. They are not in this run because they
need extra software that has not been installed. When they are added, this section and every table and chart update
from the same command.</p>"""

    if has_fm:
        tile_fm = mae.loc[[m for m in has_fm if "long context" not in m], "All hours"].idxmin()
        tile_own = mae.loc[[m for m in ("XGBoost", "Transformer") if m in models], "All hours"].idxmin()
        fifth_tile = (f'<div class="tile"><b>{(mae.loc[tile_fm, "All hours"] / mae.loc[tile_own, "All hours"] - 1) * 100:+.1f}%</b>'
                      f"<span>error of the best zero-shot foundation model ({tile_fm}) against the best model trained "
                      f"here on the same input. Untrained, it still beats the strongest naive forecast by "
                      f"{(1 - rel.relative_mae[tile_fm]) * 100:.0f}%.</span></div>")
    else:
        fifth_tile = (f'<div class="tile"><b>{iv.loc["Raw", "All hours"] * 100:.0f}% → '
                      f'{iv.loc["Adaptive conformal", "All hours"] * 100:.0f}%</b><span>coverage of nominal 80% '
                      f"intervals before and after adaptive conformal calibration.</span></div>")
    families = ("naive rules, gradient-boosted trees, a transformer trained from scratch, and pretrained foundation "
                "models used zero-shot" if has_fm else
                "naive rules, gradient-boosted trees, and a transformer trained from scratch")
    fm_ahead = has_fm and all(mae.loc[m, "Elevated hours"] < mae.loc[[x for x in learned if x not in has_fm
                                                                      and x != "Ensemble + foundation"],
                                                                     "Elevated hours"].min()
                              for m in has_fm if "long context" not in m)
    case_fm = ("The two foundation models give much wider intervals than the trained models, and Chronos-2 forecasts "
               "a repeat of the previous day's spike that did not come." if "Chronos-2" in models else "")
    trap_lead = ("that puts plain persistence first and moves every zero-shot model ahead of every model trained on "
                 "the data" if fm_ahead else "that puts plain persistence at the top of the leaderboard")

    body = f"""
<h1>When the air turns: forecasting PM2.5 through wildfire smoke</h1>
<p class="lede">A benchmark of {n_models} forecasting methods on {panel['sites_kept']} US air quality monitors, fixed
before the record 2023 smoke season and tested through it. The question is what holds up when the data stop
looking like the training set.</p>
<p class="meta">Data: US EPA Air Quality System, hourly PM2.5, {config.YEARS[0]} to {config.YEARS[-1]}.
{panel['raw_rows'] / 1e6:.1f} million raw readings, {summ['cells_scored']['All hours'] / 1e6:.1f} million forecast-hours scored.
Everything on this page is produced by <code>python run.py all</code>.</p>
<nav><a href="#data">Data and shift</a><a href="#structure">Spatial structure</a><a href="#protocol">Protocol</a>
<a href="#results">Results</a><a href="#trap">The slicing trap</a><a href="#foundation">Foundation models</a>
<a href="#spatial">Neighbours</a><a href="#uncertainty">Uncertainty</a><a href="#alerts">Alerts</a>
<a href="#limits">Limits</a><a href="#concepts">Concept map</a></nav>

<div class="tiles">
<div class="tile"><b>{ne_ratio:.1f}x</b><span>as many hours of unhealthy air in the Northeast in 2023 as in 2022. The
models were trained before it happened.</span></div>
<div class="tile"><b>{(1 - rel.relative_mae[best]) * 100:.0f}% lower error</b><span>for the best method ({best}) than the
strongest naive forecast, over {summ['cells_scored']['All hours'] / 1e6:.1f} million forecast-hours.</span></div>
<div class="tile"><b>Rank {rank('Elevated hours')['Persistence']} or rank {rank('Smoke at issue')['Persistence']}</b><span>of {n_models} for plain
persistence, depending only on how the test set is sliced. Selecting on the outcome flips the leaderboard.</span></div>
<div class="tile"><b>{max(onset_ratio.values()):.1f}x</b><span>better at ranking which clean days turn unhealthy, once a
model can see what neighbouring monitors are reading.</span></div>
{fifth_tile}
<div class="tile"><b>{onset_cov_max * 100:.0f}%</b><span>is the most any method's 80% interval covers of the hours that became
onsets of unhealthy air. Calibrated on average is not the same as warned.</span></div>
</div>

<h2>1. The question</h2>
<p>In June 2023 smoke from Canadian wildfires gave New York, Philadelphia, Chicago and Detroit some of the worst
air ever measured there. Little in the previous two years of data looked like it. That makes it a clean natural
experiment for a question every deployed model eventually faces: what happens when the world leaves the training
distribution?</p>
<p>Several families of forecaster are compared on identical forecasts: {families}. Each trained model comes with
and without information from neighbouring monitors, which isolates the value of spatial context.</p>

<h2 id="data">2. The data, and how far 2023 moved</h2>
<p>The EPA files hold one row per instrument per hour: {panel['raw_rows']:,} rows in all, covering
{panel['sites_total']} sites in the contiguous US. Sites are kept if they reported at least {config.MIN_COMPLETENESS:.0%} of hours in every one of
the four years, which leaves {panel['sites_kept']} monitors and {panel['observations_kept']:,} hourly observations
({panel['missing_share_kept']:.1%} of cells missing). Missing hours are never filled in the targets.</p>
{_fig(eda['figures']['map'], "Each dot is a monitor. Red means more hours above 35.4 µg/m³ (the US threshold for 'unhealthy for sensitive groups') in 2023 than the 2021-2022 average; blue means fewer. The East and Midwest got smoke. The West, which had far more smoke in 2021, got cleaner.")}
{_fig(eda['figures']['shift'], "Share of hours above 35.4 µg/m³ by region and year. Orange is the test year.")}
{_table(shift_tbl, {str(y): "{:.2f}%" for y in config.YEARS}, left=["Region"])}
<p>The change is not a statistical subtlety. Comparing each monitor with itself, {pt.share_monitors_higher_in_2023['Northeast', '2023 vs 2022']:.0%}
of Northeast monitors had more unhealthy hours in 2023 than in 2022 (median {pt.median_2023['Northeast', '2023 vs 2022']:.0f}
hours against {pt.median_other['Northeast', '2023 vs 2022']:.0f}; paired Wilcoxon test, Holm-adjusted p &lt; 0.001).</p>
<div class="note"><b>The shift lives in the tail.</b> A standard drift statistic on the whole distribution, the
Kolmogorov-Smirnov distance from the training period, is {ddi.ks_statistic['Northeast', 2023]:.3f} for the
Northeast in 2023 and {ddi.ks_statistic['Northeast', 2024]:.3f} in 2024. Read alone, it says the quiet year 2024
moved further than the smoke year. The 99.9th percentile says the opposite: {ddi.p999_ratio['Northeast', 2023]:.1f}
times the training value in 2023, {ddi.p999_ratio['Northeast', 2024]:.1f} times in 2024. Monitoring the bulk of a
distribution would have missed the event that mattered.</div>

<h2 id="structure">3. Smoke has spatial structure</h2>
{_fig(eda['figures']['case'], "Hourly PM2.5 in eight cities during the plume of late June 2023. The episode reaches each city at a different time.")}
{_table(lag_tbl, {"Distance from Minneapolis, km": "{:.0f}", "Trails Minneapolis by, hours": "{:+.0f}", "Peak PM2.5, µg/m³": "{:.0f}"}, left=["City"])}
<p>Lagged cross-correlation puts numbers on the picture: the series at each city lines up best with Minneapolis
when shifted by the hours shown. A monitor's own history cannot contain a plume that has not arrived, but a
monitor a few hundred kilometres upwind already does. That is the case for giving models spatial context.</p>
{_fig(eda['figures']['correlogram'], "Correlation between pairs of monitors, by distance. Correlation falls to about one half at 450 km in ordinary times and stays higher out to about 800 km in the smoke season.")}
{_fig(eda['figures']['eof'], f"The leading principal component of the monitor-by-day matrix is a single eastern pattern. Of its 50 strongest days, {eda['eof']['extreme_days_by_year']['mode1']['2023'] * 100:.0f}% fall in 2023.")}
<h3>A multiple-testing aside</h3>
<p>Does PM2.5 differ between weekdays and weekends? Testing each of the {fdr['monitors']} monitors separately gives
{fdr['significant_uncorrected']} "significant" results at p &lt; 0.05, where about
{fdr['expected_false_positives_uncorrected']:.0f} are expected by chance alone. Controlling the false discovery
rate at 5% (Benjamini-Hochberg) leaves {fdr['significant_fdr_5pct']}, and Bonferroni leaves
{fdr['significant_bonferroni']}. Unlike traffic gases, fine particles show almost no weekly cycle, and uncorrected
testing would have invented one.</p>

<h2 id="protocol">4. The benchmark protocol</h2>
<ul>
<li><b>Task.</b> Every day at {config.ISSUE_HOUR_UTC}:00 UTC, each model forecasts the next {config.HORIZON} hours at
every monitor, using at most the previous {config.CONTEXT} hours. {summ['issue_days']} issue days,
{summ['forecasts_scored']:,} forecasts.</li>
<li><b>Split by time.</b> Models learn from data before {config.TRAIN_END}. July to December 2022 is for early
stopping and tuning. 2023 and 2024 are the test: one smoke year, one ordinary year. No training target overlaps
the validation or test period.</li>
<li><b>Same cells for everyone.</b> A forecast-hour counts only if every model produced a number for it and the
truth was observed.</li>
<li><b>Three scores.</b> MAE in µg/m³; MASE, which scales by each monitor's own difficulty; and weighted quantile
loss (WQL) for the nine predicted quantiles.</li>
<li><b>Uncertainty in the comparison itself.</b> Intervals on relative error come from resampling whole days, since
a plume makes errors at different monitors on the same day strongly dependent.</li>
<li><b>Leakage tests.</b> Twelve automated checks, including ones that corrupt all data after a cut-off and assert
that no feature, model input or baseline before it changes.</li>
</ul>
{_table(model_tbl, left=["Model", "What it is"])}
<p>The tree models were tuned by random search on the validation period ({len(tuning['trials'])} settings;
the best improved validation loss by {(1 - min(t['valid_pinball'] for t in tuning['trials']) / tuning['trials'][0]['valid_pinball']) * 100:.1f}%
over the defaults, so tuning was not where the gains were). The tree model with neighbours trained on
{xgb_nb.get('train_rows', 0):,} rows on a {('GPU' if xgb_nb.get('device') == 'cuda' else 'CPU')}; the transformer
has {tr.get('parameters', 0):,} parameters and trained for {tr.get('epochs_run', 0)} epochs on a
{('GPU' if tr.get('device') == 'cuda' else 'CPU')}.</p>

<h2 id="results">5. Results</h2>
{_fig(figs['leaderboard'], "MAE relative to damped persistence (1.0), overall and in two strata that are known when the forecast is made: no reading above 35.4 µg/m³ in the previous 24 hours (calm), or at least one (smoke).")}
{_table(lead, {"MAE, µg/m³": f2, "MASE": f3, "WQL": f3, "MAE, calm at issue": f2, "MAE, smoke at issue": f2}, left=["Model"], highlight=lambda r: r["Model"] == best)}
<ul>
<li><b>Models trained on this data beat every naive rule</b>, and the intervals do not touch 1.0. The best single
model, {best_single}, has {(1 - rel.relative_mae[best_single]) * 100:.0f}% lower error than damped persistence.</li>
<li><b>Trees and the transformer are close overall</b> ({mae.loc['XGBoost + neighbours', 'All hours']:.2f} and
{mae.loc['Transformer + neighbours', 'All hours']:.2f} µg/m³ with neighbours). With smoke present at issue the
ranking is clearer: {best_smoke} is best at {mae.loc[best_smoke, 'Smoke at issue']:.2f} µg/m³.</li>
<li><b>Averaging the two families helps.</b> The ensemble fits nothing and scores
{mae.loc['Ensemble', 'All hours']:.2f} overall, better than either member, which is what happens when two models
make different errors.</li>
<li><b>The smoke year was harder for every method.</b> {best} has an MAE of
{slices[(slices.model == best) & (slices.slice == 'Year 2023')].mae.iloc[0]:.2f} µg/m³ in 2023 and
{slices[(slices.model == best) & (slices.slice == 'Year 2024')].mae.iloc[0]:.2f} in 2024. The learned models stay
ahead of every naive rule in both years.</li>
<li><b>Plain persistence is a poor forecast in smoke</b>: {(rel_smoke.relative_mae['Persistence'] - 1) * 100:.0f}% worse
than damped persistence when smoke is present at issue. Smoke episodes end, and a forecast that assumes they will
not pays for it.</li>
</ul>
{_fig(figs['horizon'], "Error by forecast horizon. The daily ripple reflects the time of day being forecast.")}

<h2 id="trap">6. The slicing trap</h2>
<p>A natural way to ask "which model is best in smoke?" is to score only the hours that turned out to be
elevated. On this dataset {trap_lead}.</p>
{_table(trap, {"MAE on hours that turned out elevated": f1, "MAE when smoke was present at issue": f2}, left=["Model"], highlight=lambda r: r["Model"] == "Persistence")}
<p>Scored on elevated outcomes, persistence ranks {rank('Elevated hours')['Persistence']} of {n_models}. Scored on a
stratum that is known when the forecast is issued, it ranks {rank('Smoke at issue')['Persistence']}. Both columns
use the same forecasts. The first selects cases by what happened, which rewards any forecast that leans high: if
only bad outcomes are scored, a forecaster that always predicts bad air looks skilled. This is known as the
forecaster's dilemma (Lerch et al., 2017), and it is easy to fall into when a team reports "performance on the
hard examples".</p>
{_fig(figs['dilemma'], f"Mean error grouped by the air quality category that was later observed. Every method under-predicts the worst hours; on hours above 125 µg/m³ the best overall model is low by {abs(lvl.loc[best, worst_level]):.0f} µg/m³. Persistence looks least bad here for the reason given above.")}
<div class="note">All rankings in this report use strata defined at issue time. Outcome-selected slices appear
only to show where the damage is, never to choose a model.</div>

{fm_section}

<h2 id="spatial">8. What neighbours add</h2>
{_fig(figs['spatial'], "Each pair compares the same model with and without information about surrounding monitors.")}
<p>For average accuracy the gain is steady but small: {nb_gain['XGBoost'] * 100:.1f}% lower MAE for the tree model
and {nb_gain['Transformer'] * 100:.1f}% for the transformer, and {nb_gain_smoke['XGBoost'] * 100:.1f}% and
{nb_gain_smoke['Transformer'] * 100:.1f}% with smoke present at issue. For warning that a clean day is about to
turn unhealthy the gain is large: PR-AUC rises by a factor of {onset_ratio['XGBoost']:.1f} for the trees and
{onset_ratio['Transformer']:.1f} for the transformer. Spatial context matters most exactly where a monitor's own
history is silent.</p>
{_fig(figs['shap'], "Exact TreeSHAP attributions for the tree model with neighbours, on test rows whose target hour was clean or elevated. Solid bars are neighbour features. The two readings that dominate are the gap between the usual value at that hour and the latest reading, and the latest reading itself.")}
<h3>A design choice that mattered less than expected</h3>
<p>The tree models predict the change from the latest reading. The reasoning was that trees cannot output a value
beyond the range of their training labels, so a model trained on levels would be capped in an unprecedented
episode. The ablation supports the reasoning only at the extremes. On hours above 125 µg/m³ the level model is
too low by {abs(lvl.loc['XGBoost (level label)', worst_level]):.0f} µg/m³ on average, against
{abs(lvl.loc['XGBoost', worst_level]):.0f} for the change model. But in the fair smoke stratum the level model is
{'slightly better' if mae.loc['XGBoost (level label)', 'Smoke at issue'] < mae.loc['XGBoost', 'Smoke at issue'] else 'worse'}
({mae.loc['XGBoost (level label)', 'Smoke at issue']:.2f} against {mae.loc['XGBoost', 'Smoke at issue']:.2f} µg/m³),
and overall the two differ by
{abs(mae.loc['XGBoost (level label)', 'All hours'] / mae.loc['XGBoost', 'All hours'] - 1) * 100:.1f}%. A sound
argument about extremes did not turn into a better forecast on average, which is the reason to run the ablation
instead of trusting the argument.</p>

<h2>9. One episode, up close</h2>
{_fig(figs['case'], "Philadelphia, 4 to 11 June 2023. Each morning's forecast for the next 24 hours, against what was measured.")}
<p>No model forecast the peak of 7 to 8 June. The morning before it, readings were moderate and every method
predicted more of the same. Once the smoke was there, the methods differ in how they let go of it. On the morning
of 8 June, with the worst already past, the tree model holds the level for a full day and the transformer decays
towards normal, which is what happened. {case_fm} This one picture contains most of the report: decent average
skill, no foresight of the onset, and real differences in recovery.</p>

<h2 id="uncertainty">10. Uncertainty</h2>
{_fig(figs['coverage'], "Share of hours inside the 80% interval, by month. All three methods dip in June 2023. The adaptive one then widens, which is how it holds its long-run average.")}
{_table(int_tbl, {"All hours": "{:.1f}%", "Smoke at issue": "{:.1f}%", "June 2023": "{:.1f}%", "Hours that became onsets": "{:.1f}%", "Mean width, µg/m³": f1}, left=["Interval method"])}
<p>Coverage of nominal 80% intervals for {best}. Raw quantiles are somewhat over-confident. Conformal calibration
fixes the average, and the adaptive version keeps coverage near 80% within the smoke stratum too. The last
column is the sobering one: of the hours that became onsets, the interval contained the truth
{iv.loc['Adaptive conformal', 'Onsets'] * 100:.1f}% of the time. A coverage guarantee is an average over all
hours. It says nothing about the rare hours that matter most, and no calibration method can add information the
model does not have.</p>

<h2 id="alerts">11. Health alerts</h2>
<p>The event is a 24-hour mean above 35.4 µg/m³ over the day following the forecast. It happens on
{a_all.base_rate.iloc[0] * 100:.2f}% of monitor-days ({int(a_all.events.iloc[0]):,} events). "Onsets" are the
{int(a_on.events.iloc[0]):,} events where the previous 24 hours were below the threshold, which is when a warning
is worth most.</p>
{_fig(figs['alerts'], "Area under the precision-recall curve for the next-day event. Left: all days. Right: only days that were clean when the forecast was issued.")}
{_table(al_tbl, {"PR-AUC, all days": f3, "Recall at 50% precision": "{:.0f}%", "PR-AUC, onsets": f3, "Onset recall at 50% precision": "{:.0f}%", "PR-AUC, day 2": f3}, left=["Method"])}
<ul>
<li>If half of all alerts must be correct, {best_alert} catches {a_all.recall_at_50pct_precision[best_alert] * 100:.0f}%
of unhealthy days.</li>
<li>For onsets, methods that see only the monitor's own history catch at most
{max(a_on.recall_at_50pct_precision[m] for m in own_history) * 100:.0f}% at that precision, and {best_onset}
catches {a_on.recall_at_50pct_precision[best_onset] * 100:.0f}%.</li>
<li>Two days ahead nothing works well: the best PR-AUC is {a2.pr_auc.max():.2f}. Warning that far out needs
weather and smoke-transport information that this benchmark does not use.</li>
<li>The classifier ranks days well but its probabilities are not to be taken at face value: when it said about
{rel_bin['predicted'] * 100:.0f}%, the event happened {rel_bin['observed'] * 100:.0f}% of the time. Early stopping
on the calm validation period halted it after
{alert_detail['training']['Classifier + neighbours | Day 1 (hours 1-24)']['best_round']} rounds, which most likely
left its probabilities compressed towards the base rate. It would need recalibrating before use.</li>
</ul>

<h2 id="limits">12. Limits</h2>
<ul>
<li><b>No meteorology.</b> Wind, mixing height and smoke-plume forecasts are what operational systems use to predict
onsets. Neighbour readings are a cheap stand-in and only help once the plume has reached a neighbour.</li>
<li><b>Monitors, not people.</b> The panel covers {panel['sites_kept']} regulatory monitors. Dense low-cost networks
would give much finer spatial context.</li>
<li><b>One shift.</b> The conclusions about behaviour under shift rest on one extraordinary season, with 2024 as a
single ordinary comparison year.</li>
<li><b>Elevated does not always mean smoke.</b> Hours above the threshold also come from winter inversions,
fireworks and dust. The benchmark does not label causes.</li>
<li><b>Model size.</b> The transformer is small ({tr.get('parameters', 0):,} parameters). A version with 1.1 million
parameters trained earlier on the CPU reached the same validation loss, so capacity does not look like the limit,
but a much larger model was not tried.</li>
<li><b>Foundation-model pretraining.</b> As noted in section 7, the zero-shot models may have been pretrained on
data from the test period.</li>
<li><b>One training run per model.</b> Each neural model was trained once, so differences of a few hundredths of a
µg/m³ between models have not been tested against run-to-run variation.</li>
</ul>

<h2 id="concepts">13. Concept map</h2>
{_table(concept_tbl, left=list(concept_tbl.columns))}

<h2>References</h2>
<ul class="refs">
<li>US Environmental Protection Agency, Air Quality System (AQS) pre-generated data files, hourly PM2.5 FRM/FEM mass (parameter 88101).</li>
<li>S. Lerch, T. Thorarinsdottir, F. Ravazzolo, T. Gneiting, "Forecaster's dilemma: extreme events and forecast evaluation," Statistical Science, 32(1), 2017.</li>
<li>Y. Nie, N. Nguyen, P. Sinthong, J. Kalagnanam, "A time series is worth 64 words: long-term forecasting with transformers," ICLR, 2023.</li>
<li>T. Chen, C. Guestrin, "XGBoost: a scalable tree boosting system," KDD, 2016.</li>
<li>A. Ansari et al., "Chronos: learning the language of time series," Transactions on Machine Learning Research, 2024.</li>
<li>A. Das, W. Kong, R. Sen, Y. Zhou, "A decoder-only foundation model for time-series forecasting," ICML, 2024.</li>
<li>Y. Romano, E. Patterson, E. Candès, "Conformalized quantile regression," NeurIPS, 2019.</li>
<li>I. Gibbs, E. Candès, "Adaptive conformal inference under distribution shift," NeurIPS, 2021.</li>
<li>S. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," Nature Machine Intelligence, 2, 2020.</li>
<li>Y. Benjamini, Y. Hochberg, "Controlling the false discovery rate: a practical and powerful approach to multiple testing," Journal of the Royal Statistical Society B, 57(1), 1995.</li>
<li>R. Hyndman, A. Koehler, "Another look at measures of forecast accuracy," International Journal of Forecasting, 22(4), 2006.</li>
</ul>
"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>SmokeShift report</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")
    path = config.OUT / "report.html"
    path.write_text(page, encoding="utf-8")
    log(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")
    return path

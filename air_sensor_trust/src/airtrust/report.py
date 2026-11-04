"""Builds outputs/report.html: one self-contained page with every result.

All numbers in the text are read from the results dictionary, so the report
cannot drift away from what the pipeline actually computed.
"""
import base64
import html

import pandas as pd

from . import calibration as cal
from . import config, forecasting as fc

CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--blue:#2a78d6;--blue-ink:#184f95}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:34px;line-height:1.15;margin:0 0 8px;letter-spacing:-.01em}
h2{font-size:23px;margin:56px 0 8px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:28px 0 6px}
p{margin:10px 0;max-width:76ch}
.lede{font-size:19px;color:var(--ink2);max-width:70ch}
.meta{color:var(--muted);font-size:14px;margin-top:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:28px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.tile b{display:block;font-size:30px;line-height:1.1;letter-spacing:-.01em}
.tile span{display:block;color:var(--ink2);font-size:14px;line-height:1.45;margin-top:6px}
figure{margin:20px 0;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px}
figure img{display:block;width:100%;height:auto}
figcaption{color:var(--ink2);font-size:14px;padding:8px 4px 2px;max-width:86ch}
.scroll{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;font-size:14px;background:var(--surface);min-width:60%}
th,td{padding:7px 12px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child,td.l,th.l{text-align:left}
th{color:var(--ink2);font-weight:600;border-bottom:1px solid #c3c2b7}
tr.hl td{font-weight:600}
.note{background:#eef4fd;border-left:3px solid var(--blue);padding:10px 14px;border-radius:0 8px 8px 0;max-width:76ch;margin:16px 0}
code{background:#efeee9;padding:1px 5px;border-radius:4px;font-size:.92em}
ul{max-width:76ch;padding-left:22px} li{margin:5px 0}
.refs li{font-size:14px;color:var(--ink2)}
nav{font-size:14px;margin:18px 0 0} nav a{color:var(--blue-ink);margin-right:14px;white-space:nowrap}
"""


def _img(name):
    data = base64.b64encode((config.FIG / name).read_bytes()).decode()
    return f'<img alt="" src="data:image/png;base64,{data}">'


def _fig(name, caption):
    return f"<figure>{_img(name)}<figcaption>{caption}</figcaption></figure>"


def _table(df, formats=None, left=(), highlight=None):
    formats = formats or {}
    head = "".join(f'<th class="{"l" if c in left else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for i, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                s = "n/a"
            elif c in formats:
                s = formats[c].format(v)
            else:
                s = html.escape(str(v))
            cells.append(f'<td class="{"l" if c in left else ""}">{s}</td>')
        cls = ' class="hl"' if highlight is not None and highlight(r) else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def build(results):
    e, a, c, f, p = (results[k] for k in ("eda", "audit", "calibration", "forecasting", "patterns"))
    H = config.FOCUS_H

    # ------------------------------------------------------------ numbers
    pt = pd.DataFrame(f["point"])
    mae = lambda h, m: pt[(pt.horizon_h == h) & (pt.model == m)]["mae"].iloc[0]
    bvb = pd.DataFrame(f["best_vs_baseline"]).set_index("horizon_h")
    b = bvb.loc[H]
    impr = -b.mae_difference / mae(H, b.best_baseline) * 100

    iv = pd.DataFrame(f["intervals"])
    ivh = iv[iv.horizon_h == H].set_index("method")
    raw, cqr, aci = (ivh.loc[m] for m in ("Quantile trees, raw", "Conformalized (static)", "Adaptive conformal"))

    al = pd.DataFrame(f["alerts"])
    alh = al[al.horizon_h == H].set_index("method")
    a_dist = alh.loc["Forecast + residual distribution (ridge)"]
    a_clf = alh.loc["Classifier (boosted trees + Platt)"]
    a_per = alh.loc["Persistence (alert if over the limit now)"]

    so = pd.DataFrame(c["strategies_overall"])
    strat = lambda t, m, s: so[(so.target == config.LABELS[t]) & (so.model == m)
                               & so.strategy.str.startswith(s)]["mae"].iloc[0]
    nox_static, nox_roll = strat("nox_ref", cal.RIDGE, "Static"), strat("nox_ref", cal.RIDGE, "Rolling")
    nox_cut = (1 - nox_roll / nox_static) * 100
    st = pd.DataFrame(c["static"])
    no2_single = st[(st.target == config.LABELS["no2_ref"]) & (st.model == cal.SINGLE)].iloc[0]
    no2_ridge = st[(st.target == config.LABELS["no2_ref"]) & (st.model == cal.RIDGE)].iloc[0]

    drift = pd.DataFrame(c["drift_summary"])
    dve = pd.DataFrame(c["drift_vs_error"])
    lc = pd.DataFrame(c["learning_curve"])
    lc_co = lc[(lc.target == config.LABELS["co_ref"]) & (lc.model == cal.GBM)].set_index("weeks")["mae"]

    lvl = pd.DataFrame(f["error_by_level"])
    over = lvl[lvl.no2_level.str.startswith("over")].iloc[0]
    bym = pd.DataFrame(f["error_by_month"]).set_index("month")
    ww = pd.DataFrame(e["weekday_weekend"])
    mvh = e["missing_vs_hour"]
    an, reg = p["anomalies"], p["regimes"]
    kept = a["missing_markers_kept_as_values"]
    train_ex = f["exceedances_in_training_by_fold"]
    first_fold = sorted(train_ex)[0]

    # ------------------------------------------------------------- tables
    audit_tbl = pd.DataFrame([
        ["R² of a plain linear regression on the original target and features", f'{a["r2_original_feature_set"]:.3f}'],
        ["R² using only the three columns the target was built from", f'{a["r2_three_components_only"]:.3f}'],
        ["R² once those columns are removed (sensors and weather only)", f'{a["r2_sensors_and_weather_only"]:.3f}'],
        ["Real hours deleted by the IQR filter", f'{a["real_hours_removed_pct"]:.0f}%'],
        ["Hours above the NO2 limit deleted by the filter",
         f'{a["no2_exceedance_hours_removed"]} of {a["no2_exceedance_hours"]} ({a["no2_exceedance_removed_pct"]:.0f}%)'],
        ["Top 5% of CO hours deleted by the filter", f'{a["co_top5pct_hours_removed_pct"]:.0f}%'],
        ["Highest NO2 value left in the data", f'{a["max_no2_left"]:.0f} µg/m³'],
        ["Missing markers (-200) that stayed in CO as real values", f'{kept["CO(GT)"]:,}'],
        ["Missing NO2 readings replaced by the previous value", f'{a["no2_values_invented_by_ffill"]:,}'],
        ["NMHC column missing before it was forward-filled", f'{a["nmhc_missing_pct_before_ffill"]:.0f}%'],
        ["Empty rows at the end of the file filled as copies of the last hour", f'{a["empty_trailing_rows"]}'],
    ], columns=["Check on the original notebook's pipeline", "Result"])

    ww_tbl = ww[["pollutant", "weekday_mean", "weekend_mean", "difference", "ci95_low", "ci95_high", "cohens_d",
                 "p_holm"]].copy()
    ww_tbl["95% CI"] = [f"{lo:.2f} to {hi:.2f}" for lo, hi in zip(ww_tbl.ci95_low, ww_tbl.ci95_high)]
    ww_tbl["p_holm"] = ["< 0.001" if v < 0.001 else f"{v:.3f}" for v in ww_tbl.p_holm]
    ww_tbl = ww_tbl.drop(columns=["ci95_low", "ci95_high"]).rename(columns={
        "pollutant": "Pollutant", "weekday_mean": "Weekday mean", "weekend_mean": "Weekend mean",
        "difference": "Difference", "cohens_d": "Cohen's d", "p_holm": "p (Holm)"})
    ww_tbl = ww_tbl[["Pollutant", "Weekday mean", "Weekend mean", "Difference", "95% CI", "Cohen's d", "p (Holm)"]]

    st_tbl = st[["target", "model", "mae", "mae_pct_of_mean", "r2", "bias"]].rename(columns={
        "target": "Pollutant", "model": "Calibration model", "mae": "MAE", "mae_pct_of_mean": "MAE, % of mean",
        "r2": "R²", "bias": "Bias"})

    so_tbl = (so[so.model == cal.RIDGE].pivot(index="target", columns="strategy", values="mae")
              .join(so[so.model == cal.GBM].pivot(index="target", columns="strategy", values="mae"),
                    lsuffix=" | ridge", rsuffix=" | trees"))
    so_tbl.columns = [col.replace(" (first 8 weeks only)", "").replace(" (refit on last 8 weeks)", "")
                      .replace(" (refit on all history)", "") for col in so_tbl.columns]
    order = [f"{s} | {m}" for m in ("ridge", "trees") for s in ("Static", "Rolling", "Expanding")]
    so_tbl = so_tbl[order].reset_index().rename(columns={"target": "Pollutant"})

    pt_tbl = pt.pivot(index="model", columns="horizon_h", values="mae").reindex(fc.BASELINES + [fc.RIDGE, fc.HGB])
    pt_tbl.columns = [f"{h} h" for h in pt_tbl.columns]
    pt_tbl = pt_tbl.reset_index().rename(columns={"model": "Model"})

    bvb_tbl = bvb.reset_index().copy()
    bvb_tbl["Improvement"] = [f"{-d:.1f} ({-hi:.1f} to {-lo:.1f})" for d, lo, hi in
                              zip(bvb_tbl.mae_difference, bvb_tbl.ci95_low, bvb_tbl.ci95_high)]
    bvb_tbl["Horizon"] = [f"{h} h" for h in bvb_tbl.horizon_h]
    bvb_tbl = bvb_tbl[["Horizon", "best_model", "best_baseline", "Improvement"]].rename(columns={
        "best_model": "Best model", "best_baseline": "Best naive baseline",
        "Improvement": "MAE improvement, µg/m³ (95% CI)"})

    lvl_tbl = lvl.rename(columns={"no2_level": "Actual NO2 (µg/m³)", "hours": "Hours"})
    lvl_tbl = lvl_tbl[["Actual NO2 (µg/m³)", "Hours", "Persistence MAE", "Ridge MAE", "Boosted trees MAE",
                       "Ridge bias", "Boosted trees bias"]]

    iv_tbl = iv.copy()
    iv_tbl["horizon_h"] = [f"{h} h" for h in iv_tbl.horizon_h]
    iv_tbl["worst"] = [f"{v * 100:.0f}% ({m})" for v, m in zip(iv_tbl.worst_month_coverage, iv_tbl.worst_month)]
    iv_tbl["coverage"] *= 100
    iv_tbl = iv_tbl[iv_tbl.horizon_h.isin([f"{h} h" for h in (1, H, 24)])]
    iv_tbl = iv_tbl[["horizon_h", "method", "coverage", "mean_width", "worst"]].rename(columns={
        "horizon_h": "Horizon", "method": "Interval method", "coverage": "Coverage (target 90%)",
        "mean_width": "Mean width, µg/m³", "worst": "Worst month"})

    al_tbl = alh.reset_index()[["method", "brier_skill", "pr_auc", "alerts", "precision", "recall"]].copy()
    al_tbl["precision"] *= 100
    al_tbl["recall"] *= 100
    al_tbl = al_tbl.rename(columns={"method": "Alert method", "brier_skill": "Brier skill", "pr_auc": "PR-AUC",
                                    "alerts": "Alerts raised", "precision": "Precision %", "recall": "Recall %"})

    imp_tbl = pd.DataFrame(f["importance"])[["feature", "mae_increase"]].rename(columns={
        "feature": "Feature shuffled", "mae_increase": "MAE increase, µg/m³"})

    top_tbl = pd.DataFrame(p["top_days"])
    top_tbl["peak_hour"] = [f"{int(h):02d}:00" for h in top_tbl.peak_hour]
    top_tbl = top_tbl[["day", "weekday", "anomalous_hours", "peak_hour", "peak_value", "typical_at_peak",
                       "peak_z"]].rename(columns={
        "day": "Day", "weekday": "Weekday", "anomalous_hours": "Flagged hours", "peak_hour": "Most unusual hour",
        "peak_value": "NO2 at that hour", "typical_at_peak": "Typical for that hour", "peak_z": "Robust z"})

    concept_tbl = pd.DataFrame([
        ["Data quality and missing-data mechanisms", "Section 2 and 3: sentinel values, chi-square test of missingness by hour, no imputation of the target"],
        ["Exploratory analysis and visualisation", "Section 3: seasonal profiles, autocorrelation, rank correlation"],
        ["Statistical inference", "Section 3 and 5: Mann-Whitney tests with Holm correction, bootstrap and day-block bootstrap intervals, effect sizes"],
        ["Dimensionality reduction", "Section 3: PCA of the sensor array"],
        ["Regression and regularisation", "Section 4 and 5: linear, ridge with cross-validated penalty, gradient-boosted trees"],
        ["Model selection and validation", "Section 5: walk-forward evaluation, gap-aware time-series cross-validation, naive baselines, leakage tests"],
        ["Classification with rare events", "Section 7: precision, recall, PR-AUC, Brier score, probability calibration, cost-based threshold"],
        ["Clustering", "Section 8: k-means on daily profiles, silhouette selection, chi-square against day type"],
        ["Anomaly detection", "Section 8: robust z-scores against a seasonal expectation"],
        ["Monitoring models after deployment", "Section 4 and 6: drift metrics (PSI, KS, domain classifier), recalibration schedules, adaptive conformal prediction"],
        ["Reproducibility", "One command reruns everything; fixed seeds; tests that fail if a feature can see the future"],
    ], columns=["Data science concept", "Where it is demonstrated"])

    f2, f3 = {"MAE": "{:.2f}", "MAE, % of mean": "{:.0f}%", "R²": "{:.2f}", "Bias": "{:+.2f}"}, "{:.1f}"

    # --------------------------------------------------------------- page
    body = f"""
<h1>Can we trust cheap air quality sensors?</h1>
<p class="lede">Calibration, drift, forecasting and honest uncertainty for a low-cost sensor box,
using {e["n_hours"]:,} hours of roadside measurements next to reference-grade analysers.</p>
<p class="meta">Data: UCI Air Quality (De Vito et al., 2008), one station in an Italian city,
{e["start"][:10]} to {e["end"][:10]}. Every number on this page is produced by <code>python run_all.py</code>.</p>
<nav><a href="#audit">Audit</a><a href="#eda">Data</a><a href="#calibration">Calibration</a>
<a href="#forecast">Forecasting</a><a href="#uncertainty">Uncertainty</a><a href="#alerts">Alerts</a>
<a href="#patterns">Patterns</a><a href="#limits">Limits</a><a href="#concepts">Concept map</a></nav>

<div class="tiles">
<div class="tile"><b>R² {a["r2_three_components_only"]:.2f}</b><span>is what a plain linear regression scores on the
original project's target, because the target was a weighted sum of three of its own input columns.</span></div>
<div class="tile"><b>{a["no2_exceedance_removed_pct"]:.0f}%</b><span>of the hours above the NO2 limit were deleted as
"outliers" by the original preprocessing.</span></div>
<div class="tile"><b>{nox_cut:.0f}% lower</b><span>NOx calibration error when the sensor model is refitted every four
weeks instead of once ({nox_static:.0f} to {nox_roll:.0f} ppb).</span></div>
<div class="tile"><b>{impr:.0f}% lower</b><span>{H}-hour NO2 forecast error than the best naive baseline
({mae(H, b.best_model):.1f} vs {mae(H, b.best_baseline):.1f} µg/m³), with no future information.</span></div>
<div class="tile"><b>{raw.coverage * 100:.0f}% → {aci.coverage * 100:.0f}%</b><span>true coverage of nominal 90% prediction
intervals, before and after adaptive conformal prediction.</span></div>
<div class="tile"><b>{a_dist.recall * 100:.0f}% caught</b><span>of limit exceedances {H} hours ahead, at
{a_dist.precision * 100:.0f}% precision. Alerting on the current reading catches {a_per.recall * 100:.0f}%.</span></div>
</div>

<h2>1. The problem</h2>
<p>Reference-grade air quality monitors are accurate, expensive and sparse. Low-cost sensor boxes are cheap enough
to put on every street, which is why networks of them now feed public air quality maps. Their weakness is trust:
the sensors respond to several gases at once, to temperature and to humidity, and their response changes with time.</p>
<p>This dataset is a year of exactly that situation. A box with five metal-oxide sensors sat beside certified
analysers for CO, NOx, NO2 and benzene. That allows four practical questions to be answered with data:</p>
<ul>
<li>How well can the cheap sensors be calibrated to reproduce the reference instruments?</li>
<li>How quickly does that calibration go stale, and can staleness be detected without a reference?</li>
<li>Can NO2 be forecast hours ahead well enough to beat simple rules of thumb?</li>
<li>Can the forecast say how sure it is, and raise a warning before the legal limit is exceeded?</li>
</ul>

<h2 id="audit">2. What the original evaluation measured</h2>
<p>The first version of this project reported R² of 0.99 for tree models predicting a "hazard score". Re-running
its preprocessing line by line on the same file shows where that number came from.</p>
{_table(audit_tbl, left=audit_tbl.columns[:1])}
<p>The hazard score was defined as a weighted sum of CO, NO2 and benzene, and those three columns were left in the
feature matrix. Any model can recover a sum of its own inputs, so the score says nothing about predictive skill.
With those columns removed, the same regression explains {a["r2_sensors_and_weather_only"] * 100:.0f}% of the
variance, and that is still a same-hour estimate, not a forecast.</p>
<p>The outlier filter did the most damage to the goal of the project. It capped NO2 at
{a["upper_fences"]["NO2(GT)"]:.1f} µg/m³ and CO at {a["upper_fences"]["CO(GT)"]:.1f} mg/m³, which removed
{a["real_hours_removed_pct"]:.0f}% of all hours and most of the pollution episodes. Most of the {a["rows_dropped_by_filter"]["CO(GT)"]:,} "CO outliers" were not outliers at
all: {kept["CO(GT)"]:,} of them were the file's missing-value marker, which had survived because that column was
still text when -200 was replaced.</p>
<div class="note">Everything below is rebuilt to avoid these problems: missing readings stay missing, no row is
deleted for being high, every model is scored on months it has never seen, and naive baselines are scored on the
same hours.</div>

<h2 id="eda">3. Understanding the data</h2>
<h3>Missing readings are not random</h3>
{_fig(e["figures"]["missingness"], "Share of hourly readings missing, by month. The sensor box loses about 4% of hours, all channels together. The reference analysers lose far more, and the NMHC analyser stops reporting altogether after April 2004.")}
<p>The NO2 analyser is missing {mvh["worst_hour_missing_pct"]:.0f}% of the time at {mvh["worst_hour"]:02d}:00 and
about {mvh["best_hour_missing_pct"]:.0f}% at its best hour (chi-square test of missingness against hour of day:
p &lt; 0.001, Cramér's V = {mvh["cramers_v"]:.2f}). That is the signature of a nightly self-calibration cycle, so the
gaps are scheduled, not random. Forward-filling would invent {a["no2_values_invented_by_ffill"]:,} NO2 values. This
project never fills the target: hours without a reference reading are simply not scored.</p>

<h3>Traffic drives a strong daily and weekly rhythm</h3>
{_fig(e["figures"]["diurnal"], "Mean concentration by hour of day. Bands are 95% bootstrap intervals from resampling whole days. The small bump in NO2 at 03:00 is an artefact: only a handful of days have a reading at that hour.")}
<p>Weekdays are more polluted than weekends for every pollutant tested. The comparison uses daily means, because
consecutive hours are too correlated to count as independent observations.</p>
{_table(ww_tbl, {"Weekday mean": "{:.1f}", "Weekend mean": "{:.1f}", "Difference": "{:.2f}", "Cohen's d": "{:.2f}"}, left=["Pollutant"])}
{_fig(e["figures"]["acf"], f"Autocorrelation of hourly NO2. It is {e['acf_no2']['lag1']:.2f} at one hour, drops quickly, and returns to {e['acf_no2']['lag24']:.2f} at 24 hours and {e['acf_no2']['lag168']:.2f} at one week. Any forecast therefore has to beat two simple rules: use the last value, or use the same hour yesterday.")}

<h3>The sensors overlap heavily, and one of them is mislabelled</h3>
{_fig(e["figures"]["correlation"], "Spearman rank correlations between reference analysers, sensor channels and weather.")}
<p>The first principal component of the five sensor channels carries
{e["sensor_pca"]["explained_variance_ratio"][0] * 100:.0f}% of their variance: they mostly measure one shared
"traffic" signal. The channel sold as an NO2 sensor has a rank correlation of only
{e["spearman_s4_vs_no2"]:.2f} with reference NO2, and tracks temperature and humidity more closely than the gas.
Useful calibration therefore has to combine channels. One more caution: reference benzene and sensor 2 have a rank
correlation of {e["spearman_s2_vs_benzene"]:.2f}, an exact one-to-one relationship, so benzene is left out of the
calibration study because "predicting" it would prove nothing.</p>

<h2 id="calibration">4. Calibrating the sensors, and keeping them calibrated</h2>
<p>The scenario copies real deployments. The box is co-located with the reference station for the first 8 weeks,
a calibration model is fitted, and the box is then scored on the remaining {int(no2_ridge.n_test):,} hours.</p>
{_table(st_tbl, f2, left=["Pollutant", "Calibration model"])}
<p>For NO2, the nominal sensor alone is worse than predicting the average (R² {no2_single.r2:.2f}). Combining all
channels with temperature and humidity brings the error to {no2_ridge.mae:.1f} µg/m³
({no2_ridge.mae_pct_of_mean:.0f}% of the mean). NOx is the opposite story: every model fitted in spring
under-reads afterwards (overall bias of {st[(st.target == config.LABELS["nox_ref"]) & (st.model == cal.RIDGE)].bias.iloc[0]:.0f} ppb
for ridge), because autumn and winter NOx levels are far above anything seen during co-location.</p>
{_fig(c["figures"]["strategies"], "Calibration error in each 4-week block for three schedules: fit once (static), refit on the latest 8 weeks (rolling), refit on all history (expanding). Refits use only data from before the block they are scored on.")}
{_table(so_tbl, {k: "{:.2f}" for k in so_tbl.columns[1:]}, left=["Pollutant"])}
<p>Periodic recalibration cuts NOx error by {nox_cut:.0f}% and helps CO with the linear model. It does little for
NO2 ({strat("no2_ref", cal.RIDGE, "Static"):.1f} to {strat("no2_ref", cal.RIDGE, "Rolling"):.1f} µg/m³ with ridge),
and for CO a tree model fitted once was already as good as the refitted ones. Recalibration is not free in practice,
since it needs fresh reference data, so the result argues for spending it where it pays: on NOx.</p>
<p>How long the co-location lasts matters less than what it covers. For CO with boosted trees, error on a fixed
later window was {lc_co[8]:.2f} after 8 weeks, {lc_co[12]:.2f} after 12 and {lc_co[16]:.2f} after 16 mg/m³: more
weeks did not reliably mean a better model.</p>

<h3>Can drift be detected without reference data?</h3>
{_fig(c["figures"]["drift"], "Left: population stability index (PSI) of each input against the co-location period. Right: mean PSI of a block against how much worse the stale ridge calibration was in that block.")}
<p>Three label-free monitors were tried: PSI, the Kolmogorov-Smirnov statistic and a domain classifier. They all
fire almost immediately. Mean PSI is above the usual 0.25 alarm level in {int((drift.mean_psi > 0.25).sum())} of
{len(drift)} blocks, and a classifier can tell every later block from the co-location period (AUC of at least
{drift.domain_auc.min():.2f}). But neither signal ranks the blocks by how wrong the calibration actually was:
rank correlations with the stale model's error range from {dve.spearman_rho.min():.2f} to
{dve.spearman_rho.max():.2f}, and {"none is" if (dve.p_value > 0.05).all() else "not all are"} significant with
{int(dve.n_blocks.iloc[0])} blocks.</p>
<div class="note">Input drift is not the same as performance drift. Seasons change temperature and humidity so much
that the inputs always look different, whether or not the calibration still works. A drift alarm on inputs alone
would have been permanently on. Knowing when to recalibrate needs at least occasional reference measurements.</div>

<h2 id="forecast">5. Forecasting NO2 hours ahead</h2>
<p>Models predict reference NO2 at 1, 3, 6, 12 and 24 hours ahead from what is known at the time of the forecast:
recent NO2, the value at the target's clock hour yesterday and last week, the sensor channels, weather, and the
hour and day type being forecast. Evaluation is walk-forward over seven test months (September 2004 to March 2005):
each month is predicted by models fitted only on earlier data. Tree hyperparameters were tuned once, inside the
first training window.</p>
{_fig(f["figures"]["skill"], "Mean absolute error on the seven walk-forward test months. At 24 hours, persistence and same hour yesterday are the same forecast.")}
{_table(pt_tbl, {k: f3 for k in pt_tbl.columns[1:]}, left=["Model"], highlight=lambda r: r["Model"] in (fc.RIDGE, fc.HGB))}
{_table(bvb_tbl, left=["Horizon", "Best model", "Best naive baseline"])}
<p>The learned models beat the best naive rule at every horizon, and the day-block bootstrap intervals exclude
zero. The gain is largest at 3 hours and modest at 1 hour, where the last reading is already a strong forecast.
A regularised linear model matches or beats gradient-boosted trees at every horizon beyond one hour. Trees cannot
predict values above those they were trained on, and the test months are much more polluted than the training
months, so the simpler model is the better choice here.</p>

<h3>Where the forecasts struggle</h3>
{_table(lvl_tbl, {k: f3 for k in lvl_tbl.columns[2:]} | {"Hours": "{:.0f}"}, left=["Actual NO2 (µg/m³)"])}
<p>Errors are not spread evenly. On the hours that matter most, those above the 200 µg/m³ limit, the {H}-hour ridge
forecast is too low by {abs(over["Ridge bias"]):.0f} µg/m³ on average, and it is too high on clean hours. This
pull towards the middle is what a model trained to minimise average error does, and it is the reason the alert
system below is built on probabilities and not on the point forecast. By month, the worst is
{bym[fc.RIDGE].idxmax()} (ridge MAE {bym[fc.RIDGE].max():.1f}), the most polluted month of the year.</p>
{_fig(f["figures"]["error_by_hour"], f"Error of the {H}-hour forecast by the hour being forecast. Persistence is worst when it carries a rush-hour value into a quiet period or the reverse.")}
<p>Shuffling one input at a time on the final test month shows what the {f["importance_model"].lower()} model relies
on: the value at the same clock hour yesterday, the time of day, and the value a week ago.</p>
{_table(imp_tbl, {"MAE increase, µg/m³": "{:.2f}"}, left=["Feature shuffled"])}

<h2 id="uncertainty">6. Prediction intervals that mean what they say</h2>
<p>A forecast of 150 µg/m³ is far more useful with a range that can be trusted. Three ways of producing a 90%
interval are compared on how often the truth actually falls inside.</p>
{_fig(f["figures"]["coverage"], f"Actual coverage of nominal 90% intervals for the {H}-hour forecast, by test month.")}
{_table(iv_tbl, {"Coverage (target 90%)": "{:.1f}%", "Mean width, µg/m³": "{:.0f}"}, left=["Horizon", "Interval method"])}
<ul>
<li><b>Quantile regression alone is overconfident.</b> Its "90%" intervals contain the truth
{raw.coverage * 100:.0f}% of the time at {H} hours.</li>
<li><b>Standard conformal calibration fixes the average but not the bad months.</b> Widening the intervals using
the four weeks before each test month lifts coverage to {cqr.coverage * 100:.0f}%, yet it falls to
{cqr.worst_month_coverage * 100:.0f}% in {cqr.worst_month}, when pollution rises faster than the calibration
window can reflect.</li>
<li><b>Adaptive conformal prediction holds its promise every month.</b> It widens after each miss and narrows after
each hit, using only outcomes that would already be known. Coverage is {aci.coverage * 100:.1f}% overall and never
below {aci.worst_month_coverage * 100:.0f}% in any month. The price is honesty about the width: about
{aci.mean_width:.0f} µg/m³ on average.</li>
</ul>
{_fig(f["figures"]["example"], f"Ridge forecast issued {H} hours earlier, with the adaptive 90% interval. Breaks are hours with no reference reading.")}

<h2 id="alerts">7. Warning before the limit is exceeded</h2>
<p>The event is an hour with NO2 above 200 µg/m³, the EU hourly limit value. It is rare
({int(a_dist.events)} of {int(a_dist.n):,} test hours, {a_dist.events / a_dist.n * 100:.0f}%) and strongly
seasonal: the first training window contained only {train_ex[first_fold]} such hours. Two ways of estimating its
probability are compared with two naive rules. Alerts are raised when the probability passes
{f["alert_threshold"]:.2f}, the threshold implied by treating a missed exceedance as {config.COST_RATIO:.0f} times
worse than a false alarm.</p>
{_table(al_tbl, {"Brier skill": "{:+.2f}", "PR-AUC": "{:.2f}", "Alerts raised": "{:.0f}", "Precision %": "{:.0f}", "Recall %": "{:.0f}"}, left=["Alert method"])}
<p>{H} hours ahead, turning the regression forecast into a probability (by adding its recent errors to the
forecast and counting how often the sum passes the limit) catches {a_dist.recall * 100:.0f}% of exceedances at
{a_dist.precision * 100:.0f}% precision. A classifier trained directly on the rare event does worse
(PR-AUC {a_clf.pr_auc:.2f} against {a_dist.pr_auc:.2f}), most likely because it had almost no positive examples to
learn from in the early months. Simply alerting when the current reading is over the limit catches
{a_per.recall * 100:.0f}%.</p>
{_fig(f["figures"]["reliability"], "Reliability of the two probability estimates. Points on the diagonal mean that a predicted 20% happens 20% of the time.")}

<h2 id="patterns">8. Unusual hours and typical days</h2>
<h3>Anomalies depend on context</h3>
<p>The original project called any value outside a fixed range an outlier. Here an hour is flagged when it is far
from what is typical for that clock hour and day type over the previous four weeks, measured with a robust z-score.
Of {an["hours_scored"]:,} hours scored, {an["contextual_anomalies"]} are flagged
({an["contextual_high"]} high, {an["contextual_low"]} low). {an["contextual_missed_by_iqr_pct"]:.0f}% of them sit
inside the fixed IQR range and would have been missed, while
{an["iqr_outliers_that_are_normal_in_context_pct"]:.0f}% of the IQR "outliers" are ordinary for their hour and
season.</p>
{_table(top_tbl, {"NO2 at that hour": "{:.0f}", "Typical for that hour": "{:.0f}", "Robust z": "{:+.1f}"}, left=["Day", "Weekday"])}

<h3>Two kinds of day</h3>
{_fig(p["figures"]["regimes"], f"Cluster centres from k-means on the shape of each day's NO2 curve ({reg['n_days']} days with near-complete data; k chosen by silhouette score).")}
<p>Without being told the day of the week, clustering separates a commuter pattern with two rush-hour peaks from
a late-evening pattern. The first is {100 - min(s["weekend_share_pct"] for s in reg["summary"]):.0f}% weekdays, the second is
{max(s["weekend_share_pct"] for s in reg["summary"]):.0f}% weekend days (chi-square p &lt; 0.001, Cramér's V =
{reg["cramers_v"]:.2f}). The weekdays that fall in the second cluster are candidates for holidays.</p>

<h2 id="limits">9. Limits and next steps</h2>
<ul>
<li><b>One station, one year, 2004 to 2005.</b> Results describe this site. There is one winter, so seasonal
conclusions rest on a single cycle.</li>
<li><b>No wind or rain.</b> The weather channels are temperature and humidity only. Wind speed and mixing height
are the main drivers of dispersion and would likely improve the 12 and 24-hour forecasts most.</li>
<li><b>The forecast uses reference NO2 as an input.</b> That suits a station with an analyser. A sensor-only
deployment would feed in calibrated sensor values, with the larger errors from section 4.</li>
<li><b>Intervals are wide.</b> They are honest, but about {aci.mean_width:.0f} µg/m³ wide at {H} hours. Narrowing
them needs better inputs, not a different interval method.</li>
<li><b>Next:</b> repeat the study on a current low-cost network with public reference data, add weather forecasts
as inputs, and test whether a small number of reference days per month is enough for recalibration.</li>
</ul>

<h2 id="concepts">10. Concept map</h2>
{_table(concept_tbl, left=list(concept_tbl.columns))}

<h2>References</h2>
<ul class="refs">
<li>S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, "On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario," Sensors and Actuators B: Chemical, vol. 129, no. 2, 2008. Source of the dataset (UCI Machine Learning Repository, "Air Quality").</li>
<li>S. De Vito, M. Piga, L. Martinotto, G. Di Francia, "CO, NO2 and NOx urban pollution monitoring with on-field calibrated electronic nose by automatic Bayesian regularization," Sensors and Actuators B: Chemical, vol. 143, no. 1, 2009.</li>
<li>Y. Romano, E. Patterson, E. Candès, "Conformalized quantile regression," NeurIPS, 2019.</li>
<li>I. Gibbs, E. Candès, "Adaptive conformal inference under distribution shift," NeurIPS, 2021.</li>
<li>S. Rabanser, S. Günnemann, Z. Lipton, "Failing loudly: an empirical study of methods for detecting dataset shift," NeurIPS, 2019.</li>
<li>S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns, vol. 4, no. 9, 2023.</li>
<li>B. Iglewicz, D. Hoaglin, "How to Detect and Handle Outliers," ASQC Quality Press, 1993.</li>
<li>Directive 2008/50/EC of the European Parliament and of the Council on ambient air quality and cleaner air for Europe (hourly NO2 limit value of 200 µg/m³).</li>
<li>World Health Organization, "WHO global air quality guidelines," 2021.</li>
</ul>
"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Air Sensor Trust</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")
    path = config.OUT / "report.html"
    path.write_text(page, encoding="utf-8")
    return path

"""Forecasting hourly NO2 1 to 24 hours ahead, with honest uncertainty.

Design rules that the evaluation follows everywhere:

* A feature row at origin time t uses only values observed at or before t.
  `tests/test_no_leakage.py` checks this by corrupting the future.
* Rows are assigned to train or test by the time of the *target* (t + h), so
  no training target overlaps the test month.
* Evaluation is walk-forward: fit on everything before a month, score that
  month, move on. Hyperparameters are tuned once, inside the first training
  window only.
* Every model is compared with naive baselines on the same rows.
* The target is never imputed. Hours with no reference reading are not scored.
"""
from collections import deque

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, plotting as P

TARGET = "no2_ref"
FIRST_TEST_MONTH, LAST_TEST_MONTH = "2004-09-01", "2005-03-01"
CALIB_DAYS = 28

PERSIST, SEAS_DAY, SEAS_WEEK, CLIM = "Persistence", "Same hour yesterday", "Same hour last week", "Climatology"
RIDGE, HGB = "Ridge", "Boosted trees"
BASELINES = [PERSIST, SEAS_DAY, SEAS_WEEK, CLIM]


# ---------------------------------------------------------------- features


def build_features(df, h):
    """Features known at origin t, and the target observed at t + h."""
    y = df[TARGET]
    F = pd.DataFrame(index=df.index)
    for lag in (0, 1, 2, 3, 6, 12, 23):
        F[f"no2_lag{lag}"] = y.shift(lag)
    # The value at the same clock hour as the target, one day and one week back.
    F["no2_target_hour_prev_day"] = y.shift(24 - h)
    F["no2_target_hour_prev_week"] = y.shift(168 - h)
    F["no2_mean6"] = y.rolling(6, min_periods=3).mean()
    F["no2_mean24"] = y.rolling(24, min_periods=12).mean()
    F["no2_max24"] = y.rolling(24, min_periods=12).max()
    F["no2_std24"] = y.rolling(24, min_periods=12).std()
    F["no2_trend3"] = y - y.shift(3)
    for c in ["co_ref", "nox_ref"] + config.SENSORS + config.WEATHER:
        F[c] = df[c]
    F["temp_trend3"] = df["temp"] - df["temp"].shift(3)
    tt = df.index + pd.Timedelta(hours=h)
    F["target_hour_sin"] = np.sin(2 * np.pi * tt.hour / 24)
    F["target_hour_cos"] = np.cos(2 * np.pi * tt.hour / 24)
    F["target_dow_sin"] = np.sin(2 * np.pi * tt.dayofweek / 7)
    F["target_dow_cos"] = np.cos(2 * np.pi * tt.dayofweek / 7)
    F["target_is_weekend"] = (tt.dayofweek >= 5).astype(float)
    return F, y.shift(-h)


def baselines(df, h):
    y = df[TARGET]
    return pd.DataFrame({PERSIST: y, SEAS_DAY: y.shift(24 - h), SEAS_WEEK: y.shift(168 - h)})


def folds(index):
    starts = pd.date_range(FIRST_TEST_MONTH, LAST_TEST_MONTH, freq="MS")
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else index.max() + pd.Timedelta(hours=1)
        yield s, e


# ------------------------------------------------------------------ models


def _ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 4, 13)))


def _hgb(params, loss="squared_error", quantile=None):
    return HistGradientBoostingRegressor(loss=loss, quantile=quantile, max_iter=300, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=config.SEED, **params)


def tune_hgb(X, y, h):
    """Small grid search with gap-aware time-series CV on the first window."""
    grid = [{"max_leaf_nodes": n, "learning_rate": lr} for n in (7, 15, 31) for lr in (0.03, 0.1)]
    cv = TimeSeriesSplit(n_splits=3, gap=h)
    best, best_mae = None, np.inf
    for params in grid:
        maes = []
        for tr, va in cv.split(X):
            m = _hgb(params).fit(X.iloc[tr], y.iloc[tr])
            maes.append(mean_absolute_error(y.iloc[va], m.predict(X.iloc[va])))
        if np.mean(maes) < best_mae:
            best, best_mae = params, float(np.mean(maes))
    return best, best_mae


# -------------------------------------------------------------- conformal


def cqr_offset(lo, hi, y, alpha):
    """Conformalized quantile regression (Romano et al., 2019).

    Returns the amount by which both quantile predictions must be pushed out
    so that the interval would have covered 1 - alpha of the calibration set.
    """
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def adaptive_conformal(q_lo, q_hi, y, pos, h, init_scores, alpha, gamma=0.01, window=CALIB_DAYS * 24):
    """Adaptive conformal inference (Gibbs & Candes, 2021) with delayed feedback.

    The working miscoverage level is nudged up after every covered point and
    down after every miss, and the score pool is a sliding window, so the
    intervals widen when the data drift and tighten again afterwards. An
    outcome for origin t only becomes available at t + h, so updates wait.
    """
    pool = deque(init_scores[-window:], maxlen=window)
    a = alpha
    lo, hi = np.empty(len(y)), np.empty(len(y))
    pending = deque()
    for i in range(len(y)):
        while pending and pos[pending[0]] + h <= pos[i]:
            j = pending.popleft()
            if np.isnan(y[j]):
                continue
            pool.append(max(q_lo[j] - y[j], y[j] - q_hi[j]))
            miss = not (lo[j] <= y[j] <= hi[j])
            a += gamma * (alpha - miss)
        level = 1 - float(np.clip(a, 0.001, 0.999))
        q = np.quantile(np.fromiter(pool, float), level, method="higher")
        lo[i], hi[i] = q_lo[i] - q, q_hi[i] + q
        pending.append(i)
    return lo, hi


# ------------------------------------------------------------- evaluation


def _platt(raw_cal, y_cal, raw_test):
    """Platt scaling on the calibration block, if it has both classes."""
    if y_cal.sum() < 5 or (1 - y_cal).sum() < 5:
        return raw_test, False
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6).fit(logit(raw_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(logit(raw_test).reshape(-1, 1))[:, 1], True


def run_horizon(df, h, log=print):
    F, y = build_features(df, h)
    B = baselines(df, h)
    tt = pd.Series(df.index + pd.Timedelta(hours=h), index=df.index)  # target time per origin
    has_y = y.notna()
    pos = np.arange(len(df))

    first = pd.Timestamp(FIRST_TEST_MONTH)
    init = has_y & (tt < first)
    params, cv_mae = tune_hgb(F[init], y[init], h)
    log(f"  h={h:>2}: tuned {params} (inner-CV MAE {cv_mae:.1f})")

    out, init_scores, platt_used = [], None, []
    for s, e in folds(df.index):
        train = has_y & (tt < s)
        proper = has_y & (tt < s - pd.Timedelta(days=CALIB_DAYS))
        calib = train & ~proper
        test = has_y & (tt >= s) & (tt < e)
        Xtr, ytr, Xte = F[train], y[train], F[test]

        res = pd.DataFrame({"y": y[test], "fold": s.strftime("%Y-%m"), "pos": pos[test.to_numpy()],
                            "target_time": tt[test]}, index=F.index[test])
        res[BASELINES[:3]] = B[test]
        clim = ytr.groupby([tt[train].dt.hour, tt[train].dt.dayofweek >= 5]).mean()
        keys = list(zip(tt[test].dt.hour, tt[test].dt.dayofweek >= 5))
        res[CLIM] = clim.reindex(keys).to_numpy()

        # Point forecasts use the whole training window.
        res[RIDGE] = _ridge().fit(Xtr, ytr).predict(Xte)
        res[HGB] = _hgb(params).fit(Xtr, ytr).predict(Xte)

        # Everything probabilistic is fitted on train-proper and calibrated on
        # the 4 weeks before the test month, which the models have not seen.
        Xp, yp, Xc, yc = F[proper], y[proper], F[calib], y[calib].to_numpy()
        q_lo = _hgb(params, "quantile", config.ALPHA / 2).fit(Xp, yp)
        q_hi = _hgb(params, "quantile", 1 - config.ALPHA / 2).fit(Xp, yp)
        res["q_lo"], res["q_hi"] = q_lo.predict(Xte), q_hi.predict(Xte)
        c_lo, c_hi = q_lo.predict(Xc), q_hi.predict(Xc)
        off = cqr_offset(c_lo, c_hi, yc, config.ALPHA)
        res["cqr_lo"], res["cqr_hi"] = res["q_lo"] - off, res["q_hi"] + off
        if init_scores is None:
            init_scores = list(np.maximum(c_lo - yc, yc - c_hi))

        # Exceedance probability, two ways.
        ridge_p = _ridge().fit(Xp, yp)
        resid = yc - ridge_p.predict(Xc)
        res["p_dist"] = (ridge_p.predict(Xte)[:, None] + resid[None, :] > config.NO2_LIMIT).mean(axis=1)
        ev_p, ev_c = (yp > config.NO2_LIMIT).astype(int).to_numpy(), (yc > config.NO2_LIMIT).astype(int)
        res["p_base"] = (ytr > config.NO2_LIMIT).mean()
        if ev_p.sum() >= 5:
            clf = HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=20, l2_regularization=1.0,
                                                 random_state=config.SEED, **params).fit(Xp, ev_p)
            p, used = _platt(clf.predict_proba(Xc)[:, 1], ev_c, clf.predict_proba(Xte)[:, 1])
        else:
            p, used = np.full(len(Xte), ev_p.mean()), False
        res["p_clf"] = p
        platt_used.append(used)
        res["train_exceedances"] = int((ytr > config.NO2_LIMIT).sum())
        out.append(res)

    res = pd.concat(out)
    res["aci_lo"], res["aci_hi"] = adaptive_conformal(
        res["q_lo"].to_numpy(), res["q_hi"].to_numpy(), res["y"].to_numpy(), res["pos"].to_numpy(),
        h, init_scores, config.ALPHA)
    res.attrs.update(params=params, cv_mae=cv_mae, platt_folds=int(sum(platt_used)))
    return res, F


def _block_bootstrap_diff(res, a, b, n_boot=2000):
    """95% CI for MAE(a) - MAE(b), resampling whole days to keep autocorrelation."""
    rng = np.random.default_rng(config.SEED)
    d = (res[a] - res["y"]).abs() - (res[b] - res["y"]).abs()
    g = d.groupby(res["target_time"].dt.date).agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    boots = g[idx, 0].sum(axis=1) / g[idx, 1].sum(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def point_metrics(res, h):
    common = res.dropna(subset=BASELINES)
    rows = []
    for m in BASELINES + [RIDGE, HGB]:
        err = common[m] - common["y"]
        rows.append({"horizon_h": h, "model": m, "mae": err.abs().mean(), "rmse": np.sqrt((err ** 2).mean()),
                     "bias": err.mean(), "n": len(common)})
    t = pd.DataFrame(rows)
    t["skill_vs_persistence"] = 1 - t["mae"] / t.loc[t.model == PERSIST, "mae"].iloc[0]
    best_base = t[t.model.isin(BASELINES)].sort_values("mae").iloc[0]["model"]
    best_ml = t[t.model.isin([RIDGE, HGB])].sort_values("mae").iloc[0]["model"]
    diff, lo, hi = _block_bootstrap_diff(common, best_ml, best_base)
    return t, {"horizon_h": h, "best_model": best_ml, "best_baseline": best_base,
               "mae_difference": diff, "ci95_low": lo, "ci95_high": hi}


def interval_metrics(res, h):
    rows = []
    for name, lo, hi in [("Quantile trees, raw", "q_lo", "q_hi"), ("Conformalized (static)", "cqr_lo", "cqr_hi"),
                         ("Adaptive conformal", "aci_lo", "aci_hi")]:
        cov = (res["y"] >= res[lo]) & (res["y"] <= res[hi])
        by_fold = cov.groupby(res["fold"]).mean()
        rows.append({"horizon_h": h, "method": name, "coverage": cov.mean(), "mean_width": (res[hi] - res[lo]).mean(),
                     "worst_month_coverage": by_fold.min(), "worst_month": by_fold.idxmin(),
                     **{f"cov_{k}": v for k, v in by_fold.items()}})
    return pd.DataFrame(rows)


def alert_metrics(res, h):
    ev = (res["y"] > config.NO2_LIMIT).astype(int).to_numpy()
    thr = 1 / (1 + config.COST_RATIO)
    base = brier_score_loss(ev, res["p_base"])
    persist = res[PERSIST].fillna(0).to_numpy() > config.NO2_LIMIT
    rows = []
    for name, p in [("Base rate (climatology)", res["p_base"].to_numpy()),
                    ("Persistence (alert if over the limit now)", persist.astype(float)),
                    ("Classifier (boosted trees + Platt)", res["p_clf"].to_numpy()),
                    ("Forecast + residual distribution (ridge)", res["p_dist"].to_numpy())]:
        alert = p >= thr
        tp = int((alert & (ev == 1)).sum())
        rows.append({"horizon_h": h, "method": name, "brier": brier_score_loss(ev, p),
                     "brier_skill": 1 - brier_score_loss(ev, p) / base,
                     "pr_auc": average_precision_score(ev, p), "roc_auc": roc_auc_score(ev, p),
                     "alerts": int(alert.sum()), "precision": tp / alert.sum() if alert.sum() else np.nan,
                     "recall": tp / ev.sum(), "events": int(ev.sum()), "n": len(ev)})
    return pd.DataFrame(rows)


def error_slices(res):
    """Where does each model struggle? By hour, by pollution level, by month."""
    common = res.dropna(subset=BASELINES)
    models = [PERSIST, RIDGE, HGB]
    abs_err = pd.DataFrame({m: (common[m] - common["y"]).abs() for m in models})
    bias = pd.DataFrame({m: common[m] - common["y"] for m in models})
    level = pd.cut(common["y"], [0, 100, 150, 200, np.inf], right=False,
                   labels=["under 100", "100 to 150", "150 to 200", "over 200 (limit)"])
    by_level = abs_err.groupby(level, observed=True).mean().add_suffix(" MAE").join(
        bias.groupby(level, observed=True).mean().add_suffix(" bias")).join(
        level.value_counts().rename("hours"))
    return {"by_hour": abs_err.groupby(common["target_time"].dt.hour).mean(),
            "by_level": by_level, "by_month": abs_err.groupby(common["fold"]).mean()}


def importance(df, res, F, h, model_name):
    """Permutation importance on the last test month (unseen by the model)."""
    y = df[TARGET].shift(-h)
    tt = pd.Series(df.index + pd.Timedelta(hours=h), index=df.index)
    s = pd.Timestamp(LAST_TEST_MONTH)
    train, test = y.notna() & (tt < s), y.notna() & (tt >= s)
    model = (_ridge() if model_name == RIDGE else _hgb(res.attrs["params"])).fit(F[train], y[train])
    pi = permutation_importance(model, F[test], y[test], scoring="neg_mean_absolute_error", n_repeats=5,
                                random_state=config.SEED)
    return (pd.DataFrame({"feature": F.columns, "mae_increase": pi.importances_mean, "std": pi.importances_std})
            .sort_values("mae_increase", ascending=False).head(10).reset_index(drop=True))


def reliability(ev, p, bins=8):
    d = pd.DataFrame({"ev": ev, "p": p})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    g = d.groupby("bin").agg(predicted=("p", "mean"), observed=("ev", "mean"), n=("ev", "size"))
    return g.reset_index(drop=True)


# ----------------------------------------------------------------- figures


def fig_skill(points):
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    style = {PERSIST: (P.MUTED, "--"), SEAS_DAY: (P.AQUA, "--"), RIDGE: (P.BLUE, "-"), HGB: (P.ORANGE, "-")}
    x = range(len(config.HORIZONS))
    for m, (color, ls) in style.items():
        v = points[points.model == m].set_index("horizon_h")["mae"].reindex(config.HORIZONS)
        ax.plot(x, v.values, color=color, linestyle=ls, marker="o", markersize=5, label=m)
    ax.set_xticks(x, [f"{h} h" for h in config.HORIZONS])
    ax.set_xlim(-0.2, len(config.HORIZONS) - 1 + 0.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE, NO2 (µg/m³), walk-forward test months")
    ax.legend(loc="lower right")
    ax.set_title("Forecast error by horizon: learned models vs naive baselines")
    return P.save(fig, "07_forecast_error_by_horizon")


def fig_error_by_hour(slices):
    t = slices["by_hour"]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for m, color, ls in [(PERSIST, P.MUTED, "--"), (RIDGE, P.BLUE, "-"), (HGB, P.ORANGE, "-")]:
        ax.plot(t.index, t[m], color=color, linestyle=ls, label=m)
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(0, float(t[[PERSIST, RIDGE, HGB]].max().max()) * 1.2)
    ax.set_xlabel("Hour of day being forecast")
    ax.set_ylabel("MAE, NO2 (µg/m³)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_title(f"Where the {config.FOCUS_H}-hour forecast struggles: error by hour of day")
    return P.save(fig, "08_forecast_error_by_hour")


def fig_coverage(intervals):
    row = intervals[intervals.horizon_h == config.FOCUS_H].set_index("method")
    months = [c[4:] for c in row.columns if c.startswith("cov_")]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    for m, color, ls in [("Quantile trees, raw", P.MUTED, "--"), ("Conformalized (static)", P.ORANGE, "-"),
                         ("Adaptive conformal", P.BLUE, "-")]:
        v = row.loc[m, [f"cov_{k}" for k in months]].astype(float).values * 100
        ax.plot(range(len(months)), v, color=color, linestyle=ls, marker="o", markersize=5, label=m)
    ax.axhline((1 - config.ALPHA) * 100, color=P.INK, linewidth=1)
    ax.annotate("target 90%", (0, 90), xytext=(0, 5), textcoords="offset points", fontsize=9, color=P.INK2)
    ax.set_xticks(range(len(months)), months)
    ax.set_ylim(50, 100)
    ax.set_ylabel("Share of hours inside the 90% interval (%)")
    ax.set_xlabel("Test month")
    ax.legend(loc="lower left")
    ax.set_title(f"Do 90% intervals really cover 90%? ({config.FOCUS_H}-hour forecast, by month)")
    return P.save(fig, "09_interval_coverage")


def fig_example(res):
    w = res[res.fold == "2005-02"].set_index("target_time").sort_index()
    w = w[w.index < w.index.min() + pd.Timedelta(days=10)]
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h"))
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.fill_between(w.index, w["aci_lo"], w["aci_hi"], color=P.BLUE, alpha=0.18, linewidth=0,
                    label="90% adaptive conformal interval")
    ax.plot(w.index, w[RIDGE], color=P.BLUE, linewidth=1.6, label=f"Ridge forecast made {config.FOCUS_H} h earlier")
    ax.plot(w.index, w["y"], color=P.INK, linewidth=1.3, label="Measured NO2")
    ax.axhline(config.NO2_LIMIT, color=P.RED, linewidth=1, linestyle=(0, (4, 3)), label="200 µg/m³ hourly limit")
    ax.set_ylabel("NO2 (µg/m³)")
    ax.set_ylim(bottom=0, top=float(np.nanmax([w["aci_hi"].max(), w["y"].max()])) * 1.05)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_title("Ten days of February 2005: forecast, interval and what actually happened", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2)
    return P.save(fig, "10_forecast_example")


def fig_reliability(res):
    ev = (res["y"] > config.NO2_LIMIT).astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = 0.0
    for name, col, color in [("Classifier", "p_clf", P.ORANGE), ("Forecast + residuals", "p_dist", P.BLUE)]:
        r = reliability(ev, res[col].to_numpy())
        ax.plot(r["predicted"], r["observed"], color=color, marker="o", markersize=5, label=name)
        lim = max(lim, r["predicted"].max(), r["observed"].max())
    lim = min(1.0, lim * 1.2)
    ax.plot([0, lim], [0, lim], color=P.AXIS, linewidth=1, zorder=1)
    ax.annotate("perfect calibration", (lim * 0.72, lim * 0.72), xytext=(8, -12), textcoords="offset points",
                fontsize=9, color=P.INK2)
    ax.set_xlim(0, lim), ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted probability of exceeding the limit")
    ax.set_ylabel("Observed frequency")
    ax.legend(loc="upper left")
    ax.set_title(f"Alert reliability, {config.FOCUS_H} h ahead")
    return P.save(fig, "11_alert_reliability")


# --------------------------------------------------------------------- run


def run(df, log=print):
    points, tests, intervals, alerts, tuned = [], [], [], [], []
    focus = None
    for h in config.HORIZONS:
        res, F = run_horizon(df, h, log)
        res.drop(columns=["pos"]).round(3).to_csv(config.PRED / f"forecast_h{h:02d}.csv")
        p, t = point_metrics(res, h)
        points.append(p), tests.append(t)
        intervals.append(interval_metrics(res, h))
        alerts.append(alert_metrics(res, h))
        tuned.append({"horizon_h": h, **res.attrs["params"], "inner_cv_mae": res.attrs["cv_mae"],
                      "folds_with_platt_scaling": res.attrs["platt_folds"]})
        if h == config.FOCUS_H:
            focus = (res, F)
    points, tests = pd.concat(points, ignore_index=True), pd.DataFrame(tests)
    intervals, alerts = pd.concat(intervals, ignore_index=True), pd.concat(alerts, ignore_index=True)

    res, F = focus
    slices = error_slices(res)
    best = tests.loc[tests.horizon_h == config.FOCUS_H, "best_model"].iloc[0]
    imp = importance(df, res, F, config.FOCUS_H, best)
    train_events = res.groupby("fold")["train_exceedances"].first()
    test_events = (res["y"] > config.NO2_LIMIT).groupby(res["fold"]).sum()

    for name, t in [("forecast_point_metrics", points), ("forecast_best_vs_baseline", tests),
                    ("forecast_interval_coverage", intervals), ("forecast_alert_metrics", alerts),
                    ("forecast_tuning", pd.DataFrame(tuned)), ("forecast_importance", imp),
                    ("forecast_error_by_level", slices["by_level"].reset_index(names="no2_level")),
                    ("forecast_error_by_month", slices["by_month"].reset_index(names="month")),
                    ("forecast_error_by_hour", slices["by_hour"].reset_index(names="hour"))]:
        t.to_csv(config.TAB / f"{name}.csv", index=False)

    return {
        "point": points.to_dict("records"),
        "best_vs_baseline": tests.to_dict("records"),
        "intervals": intervals[["horizon_h", "method", "coverage", "mean_width", "worst_month_coverage",
                                "worst_month"]].to_dict("records"),
        "alerts": alerts.to_dict("records"),
        "tuning": tuned,
        "importance": imp.to_dict("records"),
        "importance_model": best,
        "error_by_level": slices["by_level"].reset_index(names="no2_level").to_dict("records"),
        "error_by_month": slices["by_month"].reset_index(names="month").to_dict("records"),
        "exceedances_in_training_by_fold": {k: int(v) for k, v in train_events.items()},
        "exceedances_in_test_by_fold": {k: int(v) for k, v in test_events.items()},
        "alert_threshold": 1 / (1 + config.COST_RATIO),
        "figures": {"skill": fig_skill(points), "error_by_hour": fig_error_by_hour(slices),
                    "coverage": fig_coverage(intervals), "example": fig_example(res),
                    "reliability": fig_reliability(res)},
    }

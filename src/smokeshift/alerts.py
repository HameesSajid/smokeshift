"""Health alerts: will the next day be unhealthy for sensitive groups?

The event is a 24-hour mean PM2.5 above 35.4 µg/m³ (the US AQI category
"Unhealthy for Sensitive Groups" or worse), for the day starting at the issue
time (hours 1-24) and for the day after (hours 25-48).

Two questions are asked of every method:

* ranking: does a higher score mean the event is more likely? (PR-AUC)
* onset: can it warn when the air is still clean at issue time? This is the
  case a warning system exists for. Persistence cannot do it by construction.

A classifier trained directly on the event gives probabilities, fitted with
and without the neighbour features. Whether those probabilities can be taken
at face value is checked with a reliability table rather than assumed.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config, gbm, spatial, store
from .bench import Bench

DAYS = {"Day 1 (hours 1-24)": (0, 24), "Day 2 (hours 25-48)": (24, 48)}
MIN_HOURS = 18
PARAMS = {"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist", "max_depth": 6, "eta": 0.05,
          "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20, "seed": config.SEED}


def events(bench, t_idx):
    """{day: (label [S, D] float with NaN where unknown)}, plus the onset mask [S, D]."""
    y = bench.targets(t_idx)
    out = {}
    for name, (a, b) in DAYS.items():
        seg = y[:, :, a:b]
        n = np.isfinite(seg).sum(axis=2)
        mean = np.nansum(seg, axis=2) / np.maximum(n, 1)
        out[name] = np.where(n >= MIN_HOURS, (mean > config.USG).astype("float32"), np.nan)
    last_day = bench.context(t_idx, 24)
    n = np.isfinite(last_day).sum(axis=2)
    calm = (np.nansum(last_day, axis=2) / np.maximum(n, 1)) <= config.USG
    return out, calm


def _design(bench, t_idx, use_spatial, nb):
    """Flatten the horizon-independent feature tensor to one row per (monitor, issue time)."""
    X, names, _ = gbm._base_in_blocks(bench, t_idx, use_spatial, nb)
    return X.reshape(-1, X.shape[2]), names


def fit_classifiers(bench, log=print):
    """Probability of each event, from features known at the issue time."""
    t_train, t_valid, t_test = (bench.issue_times("train", gbm.TRAIN_ISSUE_HOURS), bench.issue_times("valid"),
                                bench.issue_times("test"))
    dev = gbm.device()
    probs, info = {}, {}
    for label, use_spatial in [("Classifier", False), ("Classifier + neighbours", True)]:
        nb = spatial.Neighbours(bench.stations) if use_spatial else None
        Xs = {k: _design(bench, t, use_spatial, nb) for k, t in (("train", t_train), ("valid", t_valid), ("test", t_test))}
        names = Xs["train"][1]
        ys = {k: events(bench, t)[0] for k, t in (("train", t_train), ("valid", t_valid))}
        el = {k: bench.eligible(t).ravel() for k, t in (("train", t_train), ("valid", t_valid), ("test", t_test))}
        for day in DAYS:
            ytr, yva = ys["train"][day].ravel(), ys["valid"][day].ravel()
            mtr, mva = el["train"] & np.isfinite(ytr), el["valid"] & np.isfinite(yva)
            dtr = xgb.QuantileDMatrix(Xs["train"][0][mtr], ytr[mtr], feature_names=names)
            dva = xgb.QuantileDMatrix(Xs["valid"][0][mva], yva[mva], feature_names=names, ref=dtr)
            booster = xgb.train({**PARAMS, "device": dev}, dtr, 800 if dev == "cuda" else 300, evals=[(dva, "valid")],
                                early_stopping_rounds=40, verbose_eval=False)
            p = booster.predict(xgb.DMatrix(Xs["test"][0], feature_names=names),
                                iteration_range=(0, booster.best_iteration + 1))
            probs[(label, day)] = np.where(el["test"], p, np.nan).reshape(bench.n_sites, len(t_test))
            info[f"{label} | {day}"] = {"best_round": int(booster.best_iteration + 1), "train_rows": int(mtr.sum()),
                                        "train_events": int(ytr[mtr].sum()), "valid_events": int(yva[mva].sum())}
            log(f"  {label}, {day}: {int(ytr[mtr].sum()):,} training events in {int(mtr.sum()):,} rows, "
                f"best round {booster.best_iteration + 1}")
    return probs, info


def _recall_at_precision(y, score, precision=0.5):
    """Largest share of events caught while at least `precision` of the alerts raised are correct."""
    order = np.argsort(-score)
    tp = np.cumsum(y[order])
    prec = tp / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(prec >= precision)
    return float(tp[ok[-1]] / y.sum()) if len(ok) else 0.0


def _row(method, day, slice_name, y, score, is_prob):
    """One line of the results table: ranking skill of one method on one event day and slice."""
    row = {"method": method, "day": day, "slice": slice_name, "n": int(len(y)), "events": int(y.sum()),
           "base_rate": float(y.mean()), "pr_auc": float(average_precision_score(y, score)),
           "roc_auc": float(roc_auc_score(y, score)), "recall_at_50pct_precision": _recall_at_precision(y, score)}
    if is_prob:
        row["brier_skill"] = float(1 - brier_score_loss(y, score) / brier_score_loss(y, np.full(len(y), y.mean())))
    return row


def reliability(y, p, edges=(0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001)):
    """Predicted probability against observed frequency, in fixed probability bins holding at least 30 cases."""
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() >= 30:
            rows.append({"bin": f"{a:.2f}-{min(b, 1):.2f}", "predicted": float(p[m].mean()),
                         "observed": float(y[m].mean()), "n": int(m.sum())})
    return rows


def run(bench=None, log=print):
    """Fit the classifiers, score every method on both event days and both slices, and write the alert tables."""
    bench = bench or Bench.load()
    t_test = bench.issue_times("test")
    ev, calm = events(bench, t_test)
    probs, info = fit_classifiers(bench, log)
    forecasts = {name: store.load(name)["point"] for name in store.available()}
    last_day = bench.context(t_test, 24)
    recent_mean = np.nansum(last_day, axis=2) / np.maximum(np.isfinite(last_day).sum(axis=2), 1)

    rows, rel = [], {}
    for day, (a, b) in DAYS.items():
        y = ev[day]
        base = np.isfinite(y) & bench.eligible(t_test)
        for f in forecasts.values():
            base &= np.isfinite(f[:, :, a:b]).all(axis=2)
        scores = {"Last 24 h mean (persistence)": (recent_mean, False)}
        scores |= {name: (f[:, :, a:b].mean(axis=2), False) for name, f in forecasts.items()}
        scores |= {label: (probs[(label, day)], True) for label in ("Classifier", "Classifier + neighbours")}
        for slice_name, m in (("All days", base), ("Onsets (clean at issue)", base & calm)):
            for method, (s, is_prob) in scores.items():
                rows.append(_row(method, day, slice_name, y[m], s[m], is_prob))
        for label in ("Classifier", "Classifier + neighbours"):
            rel[f"{label} | {day}"] = reliability(y[base], probs[(label, day)][base])
    table = pd.DataFrame(rows)
    config.TAB.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TAB / "alerts.csv", index=False)
    (config.TAB / "alerts_detail.json").write_text(json.dumps({"training": info, "reliability": rel}, indent=2),
                                                   encoding="utf-8")
    show = table[table.day.str.startswith("Day 1")].pivot(index="method", columns="slice", values="pr_auc")
    log("  PR-AUC for day 1:\n" + show.round(3).sort_values("All days", ascending=False).to_string())
    return table

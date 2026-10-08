"""How much does the transformer's score move from one training run to the next?

A neural network trained twice with different random seeds gives two slightly
different models. Until that spread is measured, a small gap between two
models on a leaderboard cannot be called a difference. This stage trains each
transformer variant with several seeds and scores every run on the same cells
as the main benchmark, so the leaderboard gaps can be read against it.

The forecasts of these extra runs are scored and then discarded (each would
take 0.75 GB); only the score tables are kept. The leaderboard entry for each
variant remains the single run stored by the `neural` stage.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from . import alerts, config, evaluate, metrics, neural
from .bench import Bench

SEEDS = [7, 11, 23, 37, 53]
SLICES = ["All hours", "Calm at issue", "Smoke at issue"]


def score_run(point, quantiles, y, reg, scale, day1, base, calm, name="run"):
    """Benchmark scores of one set of forecasts: MAE and WQL by slice, and day-1 alert PR-AUC."""
    s = metrics.ModelScores(name, {"point": point, "quantiles": quantiles}, y, scale)
    row = {}
    for k in SLICES:
        r = s.row(k, reg[k])
        row[f"mae | {k}"], row[f"wql | {k}"] = r["mae"], r["wql"]
    alert_score = point[:, :, :24].mean(axis=2)
    row["pr_auc | all days"] = float(average_precision_score(day1[base], alert_score[base]))
    onset = base & calm
    row["pr_auc | onsets"] = float(average_precision_score(day1[onset], alert_score[onset]))
    return row


def summarise(runs):
    """Mean, standard deviation, minimum and maximum of every score, per model."""
    metric_cols = [c for c in runs.columns if " | " in c]
    long = runs.melt(id_vars=["model", "seed"], value_vars=metric_cols, var_name="metric")
    out = long.groupby(["model", "metric"])["value"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    return out.rename(columns={"count": "runs"})


def run(bench=None, log=print, seeds=SEEDS):
    """Train every transformer variant once per seed, score each run and write the two seed tables."""
    bench = bench or Bench.load()
    t_idx = bench.issue_times("test")
    y, elig = evaluate.common_grid(bench, t_idx)
    reg = metrics.regimes(bench, t_idx, y, elig)
    scale = metrics.mase_scale(bench)
    events, calm = alerts.events(bench, t_idx)
    day1 = events["Day 1 (hours 1-24)"]
    base = np.isfinite(day1) & elig

    rows = []
    for name in neural.VARIANTS:
        for seed in seeds:
            model, windows, info = neural.fit(bench, name, log=lambda *_: None, seed=seed, save=False)
            point, quantiles = neural.predict(bench, model, windows, t_idx)
            row = {"model": name, "seed": seed, "epochs": info["epochs_run"],
                   "valid_loss": info["best_valid_pinball_asinh"], "train_seconds": info["train_seconds"],
                   "device": info["device"], **score_run(point, quantiles, y, reg, scale, day1, base, calm)}
            rows.append(row)
            log(f"  {name}, seed {seed}: MAE {row['mae | All hours']:.3f} overall, "
                f"{row['mae | Smoke at issue']:.2f} in smoke, onset PR-AUC {row['pr_auc | onsets']:.3f} "
                f"({info['epochs_run']} epochs, {info['train_seconds']:.0f}s)")
            model = windows = point = quantiles = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    runs = pd.DataFrame(rows)
    summary = summarise(runs)
    config.TAB.mkdir(parents=True, exist_ok=True)
    runs.to_csv(config.TAB / "seed_runs.csv", index=False)
    summary.to_csv(config.TAB / "seed_summary.csv", index=False)
    show = summary[summary.metric.isin(["mae | All hours", "mae | Smoke at issue", "pr_auc | onsets"])]
    log("  spread over seeds:\n" + show.round(4).to_string(index=False))
    return runs

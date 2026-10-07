"""Scoring. Every function takes forecasts on the benchmark grid.

Three accuracy measures are reported because each answers a different question:

* MAE in µg/m³: how far off, in units a person can picture.
* MASE: MAE divided by the in-sample error of the seasonal naive forecast at
  the same monitor, so that clean and dirty sites weigh equally.
* WQL (weighted quantile loss): accuracy of the whole predictive
  distribution, the standard score for probabilistic forecasts.

Uncertainty for differences between models comes from a block bootstrap over
issue days: all monitors on a day are resampled together, because a smoke
plume makes errors on the same day strongly dependent across monitors.
"""
import numpy as np
import pandas as pd

from . import config
from .bench import H

Q = np.array(config.QUANTILES, dtype="float32")


def regimes(bench, t_idx, y, elig):
    """Boolean masks [monitors, days, horizon] for conditional verification.

    Two kinds of slice, and the difference matters:

    * By what was known at issue time ("Calm at issue", "Smoke at issue").
      These are fair strata: a forecaster could have been told which one
      applies before forecasting.
    * By what then happened ("Elevated hours", "Onsets"). These select on the
      outcome. They show where the damage is, but a score computed only on
      extreme outcomes rewards any forecast that leans high (the "forecaster's
      dilemma", Lerch et al. 2017), so they are not used to rank models.
    """
    valid = np.isfinite(y) & elig[:, :, None]
    elevated = valid & (y > config.USG)
    last_day = bench.context(t_idx, 24)
    recent_max = np.where(np.isfinite(last_day), last_day, -np.inf).max(axis=2)
    calm_at_issue = (recent_max <= config.USG)[:, :, None]
    return {
        "All hours": valid,
        "Calm at issue": valid & calm_at_issue,       # no reading above the threshold in the last 24 h
        "Smoke at issue": valid & ~calm_at_issue,
        "Elevated hours": elevated,
        "Onsets": elevated & calm_at_issue,
    }


def horizon_masks():
    """Boolean masks over the 48 horizons for the buckets 1-6 h, 7-24 h and 25-48 h."""
    h = np.arange(1, H + 1)
    return {name: (h >= lo) & (h <= hi) for name, (lo, hi) in config.HORIZON_BUCKETS.items()}


def mase_scale(bench):
    """Per-monitor MAE of 'same hour yesterday' over the training period."""
    lo, hi = bench.bounds("train")
    v = bench.values[:, lo:hi]
    return np.nanmean(np.abs(v[:, 24:] - v[:, :-24]), axis=1)


def pinball(y, q):
    """Pinball loss per level; y [...], q [..., levels]. For small arrays."""
    d = y[..., None] - q
    return np.maximum(Q * d, (Q - 1) * d)


def pinball_mean(y, q):
    """Pinball loss averaged over levels, one level at a time to bound memory."""
    out = np.zeros(y.shape, dtype="float32")
    for k, a in enumerate(Q):
        d = y - q[..., k]
        out += np.maximum(a * d, (a - 1) * d)
    return out / len(Q)


class ModelScores:
    """Everything needed from one model, computed in a single pass over its forecast."""

    def __init__(self, name, forecast, y, scale):
        """Compute error, absolute error and pinball loss for one model, once."""
        self.name = name
        self.err = np.asarray(forecast["point"]) - y
        self.abs = np.abs(self.err)
        self.pb = pinball_mean(y, forecast["quantiles"]) if forecast["quantiles"] is not None else None
        self.y_abs = np.abs(y)
        self.scale = scale[:, None, None]

    def row(self, slice_name, m):
        """MAE, bias, MASE and WQL on the cells selected by mask `m`."""
        n = int(m.sum())
        if n == 0:
            return None
        return {"model": self.name, "slice": slice_name, "n": n, "mae": float(self.abs[m].mean()),
                "bias": float(self.err[m].mean()), "mase": float((self.abs / self.scale)[m].mean()),
                "wql": float(2 * self.pb[m].sum() / self.y_abs[m].sum()) if self.pb is not None else np.nan}

    def by_horizon(self, m):
        """MAE at each of the 48 horizons on the cells selected by `m`."""
        return np.where(m, self.abs, 0).sum(axis=(0, 1)) / np.maximum(m.sum(axis=(0, 1)), 1)

    def daily_sum(self, m):
        """Sum of absolute errors per issue day on the cells selected by `m` (input to the bootstrap)."""
        return np.where(m, self.abs, 0).sum(axis=(0, 2))


def bootstrap_relative(daily, reference, n_boot=1000, seed=config.SEED):
    """MAE of each model relative to `reference`, with a day-block bootstrap CI.

    daily maps model -> per-day sum of absolute errors on a fixed set of cells.
    """
    rng = np.random.default_rng(seed)
    n_days = len(daily[reference])
    draws = rng.integers(0, n_days, (n_boot, n_days))
    ref = daily[reference][draws].sum(axis=1)
    rows = []
    for name, d in daily.items():
        ratio = d[draws].sum(axis=1) / ref
        rows.append({"model": name, "relative_mae": float(d.sum() / daily[reference].sum()),
                     "ci_low": float(np.percentile(ratio, 2.5)), "ci_high": float(np.percentile(ratio, 97.5))})
    return pd.DataFrame(rows).sort_values("relative_mae").reset_index(drop=True)

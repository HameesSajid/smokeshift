"""Tabular features for the tree models.

One row is (monitor, issue time, horizon). Everything in a row is known at
the issue time; `tests/test_no_leakage.py` corrupts the future and checks
that no feature moves.

The label is the *change* from the latest reading, not the level. The
reasoning: trees cannot predict a value outside the range of their training
labels, so a tree trained on levels is capped near the highest level it has
seen, while a tree predicting the change can ride on top of whatever the
current reading is. The "XGBoost (level label)" variant in gbm.py tests that
reasoning rather than assuming it; the report gives the result.
"""
import warnings

import numpy as np

from . import spatial
from .bench import H, L

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 167]
EPOCH_DOW = 4   # 2021-01-01 was a Friday


def base(bench, t_idx, use_spatial=False, nb=None):
    """Horizon-independent features: ([monitors, issues, F], names, profile)."""
    ctx = bench.context(t_idx)
    last = bench.last_obs(t_idx)
    f = {"last": last}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-missing windows stay NaN
        for lag in LAGS:
            f[f"lag{lag}_minus_last"] = ctx[:, :, L - 1 - lag] - last
        for w in (3, 6, 24, 72, 168):
            f[f"mean{w}_minus_last"] = np.nanmean(ctx[:, :, L - w:], axis=2) - last
        f["mean24"] = np.nanmean(ctx[:, :, L - 24:], axis=2)
        f["max24_minus_last"] = np.nanmax(ctx[:, :, L - 24:], axis=2) - last
        f["min24_minus_last"] = np.nanmin(ctx[:, :, L - 24:], axis=2) - last
        f["max168"] = np.nanmax(ctx, axis=2)
        f["std24"] = np.nanstd(ctx[:, :, L - 24:], axis=2)
        f["std168"] = np.nanstd(ctx, axis=2)
        profile = np.nanmean(ctx.reshape(*ctx.shape[:2], L // 24, 24), axis=2)
    f["missing_share"] = 1 - np.isfinite(ctx).mean(axis=2)

    lon = bench.stations["lon"].to_numpy()[:, None]
    solar = (bench.hours.hour.to_numpy()[t_idx][None, :] + lon / 15) % 24
    doy = bench.hours.dayofyear.to_numpy()[t_idx][None, :] / 365.25
    ones = np.ones_like(last)
    f["issue_solar_sin"], f["issue_solar_cos"] = np.sin(2 * np.pi * solar / 24), np.cos(2 * np.pi * solar / 24)
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy) * ones, np.cos(2 * np.pi * doy) * ones
    f["lat"] = bench.stations["lat"].to_numpy()[:, None] * ones
    f["lon"] = lon * ones

    names = list(f)
    X = np.stack([f[k] for k in names], axis=2).astype("float32")
    if use_spatial:
        Xs, s_names = spatial.features(bench, t_idx, nb)
        X, names = np.concatenate([X, Xs], axis=2), names + s_names
    return X, names, profile.astype("float32")


HORIZON_NAMES = ["horizon", "seasonal_minus_last", "profile_minus_last", "target_solar_sin", "target_solar_cos",
                 "target_weekend"]


def rows(bench, t_idx, X, profile, site, day, horizon):
    """Assemble design rows for the chosen (monitor, issue, horizon) triples.

    site, day, horizon are equal-length integer arrays. Returns (features, label, last).
    The label is NaN where the target was not observed.
    """
    t = t_idx[day]
    tt = t + horizon
    last = X[site, day, 0]
    back = 24 * np.ceil(horizon / 24).astype(int)
    seasonal = bench.values[site, tt - back]
    prof = profile[site, day, (L - 1 + horizon) % 24]
    lon = bench.stations["lon"].to_numpy()[site]
    local = tt + lon / 15
    solar = (bench.hours.hour.to_numpy()[tt] + lon / 15) % 24
    dow = (EPOCH_DOW + np.floor(local / 24)) % 7
    extra = np.stack([horizon.astype("float32"), seasonal - last, prof - last,
                      np.sin(2 * np.pi * solar / 24), np.cos(2 * np.pi * solar / 24), (dow >= 5).astype("float32")],
                     axis=1).astype("float32")
    return np.concatenate([X[site, day], extra], axis=1), bench.values[site, tt] - last, last


def sample_training_rows(bench, t_idx, per_origin, rng):
    """Pick `per_origin` random horizons for every eligible (monitor, issue) pair."""
    site, day = np.nonzero(bench.eligible(t_idx))
    site, day = np.repeat(site, per_origin), np.repeat(day, per_origin)
    return site, day, rng.integers(1, H + 1, len(site))


def all_rows(elig, day_lo, day_hi):
    """Every horizon for every eligible pair in a block of issue days."""
    site, day = np.nonzero(elig[:, day_lo:day_hi])
    n = len(site)
    return np.repeat(site, H), np.repeat(day + day_lo, H), np.tile(np.arange(1, H + 1), n)

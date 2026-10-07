"""Reference forecasts that need no learning, or almost none.

A learned model has to beat all three to be worth its complexity.
"""
import warnings

import numpy as np

from .bench import H, L

HOURS = np.arange(1, H + 1)


def persistence(bench, t_idx):
    """Tomorrow looks like right now."""
    return np.repeat(bench.last_obs(t_idx)[:, :, None], H, axis=2)


def seasonal_naive(bench, t_idx):
    """The same clock hour on the most recent day already observed."""
    back = 24 * np.ceil(HOURS / 24).astype(int)          # 24 for 1-24 h, 48 for 25-48 h
    f = bench.values[:, t_idx[:, None] + HOURS[None, :] - back[None, :]]
    return np.where(np.isfinite(f), f, persistence(bench, t_idx))


def _profile_and_anomaly(bench, t_idx):
    """Seven-day mean for each clock hour, and how far 'now' sits from it."""
    ctx = bench.context(t_idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-missing slices give NaN, handled below
        profile = np.nanmean(ctx.reshape(*ctx.shape[:2], L // 24, 24), axis=2)   # [S, D, 24]
        level = np.nanmean(ctx, axis=2, keepdims=True)
    profile = np.where(np.isfinite(profile), profile, level)
    anomaly = bench.last_obs(t_idx) - profile[:, :, (L - 1) % 24]
    # Context position p and target hour h share a clock hour when p % 24 == (L - 1 + h) % 24.
    return profile[:, :, (L - 1 + HOURS) % 24], anomaly


class DampedPersistence:
    """Daily profile plus a decaying share of the current anomaly.

    forecast(h) = profile(clock hour of h) + rho_h * (now - profile(now))

    rho_h is one number per horizon, fitted by least squares on the training
    period. rho = 1 would be persistence of the anomaly, rho = 0 the profile.
    """

    def fit(self, bench, t_idx):
        """Estimate rho for each horizon by least squares over the given issue times."""
        prof, anom = _profile_and_anomaly(bench, t_idx)
        y = bench.targets(t_idx) - prof
        ok = np.isfinite(y) & np.isfinite(anom)[:, :, None] & bench.eligible(t_idx)[:, :, None]
        a = np.where(ok, anom[:, :, None], 0.0)
        self.rho = (a * np.where(ok, y, 0.0)).sum(axis=(0, 1)) / (a * a).sum(axis=(0, 1))
        return self

    def predict(self, bench, t_idx):
        """Forecast every monitor and issue time, falling back to persistence where the profile is missing."""
        prof, anom = _profile_and_anomaly(bench, t_idx)
        f = prof + self.rho[None, None, :] * anom[:, :, None]
        return np.where(np.isfinite(f), f, persistence(bench, t_idx))

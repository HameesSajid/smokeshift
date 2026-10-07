"""The benchmark protocol: who is forecast, when, and from what information.

Every model answers the same question. At an issue time t it may use the
observations up to and including t (at most CONTEXT hours of its own monitor,
and for spatial models the same span at other monitors) and must predict the
next HORIZON hours. Forecasts are stored on a fixed grid

    [monitor, issue day, horizon]

so that any two models are compared on exactly the same cells.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, panel

H = config.HORIZON
L = config.CONTEXT


@dataclass
class Bench:
    """The panel, plus the indexing helpers that define what a forecast may see."""
    values: np.ndarray        # [monitors, hours], NaN where no reading
    hours: pd.DatetimeIndex   # GMT
    stations: pd.DataFrame

    @classmethod
    def load(cls):
        """Read the processed panel from data/processed."""
        return cls(*panel.load())

    @property
    def n_sites(self):
        """Number of monitors."""
        return self.values.shape[0]

    def idx(self, ts):
        """Hour index of a timestamp (hours since the start of the panel)."""
        return int((pd.Timestamp(ts) - self.hours[0]) / pd.Timedelta(hours=1))

    def bounds(self, split):
        """First and one-past-last hour index of the train, valid or test period."""
        return {"train": (0, self.idx(config.TRAIN_END)),
                "valid": (self.idx(config.TRAIN_END), self.idx(config.VALID_END)),
                "test": (self.idx(config.VALID_END), len(self.hours))}[split]

    def issue_times(self, split, issue_hours=(config.ISSUE_HOUR_UTC,)):
        """Hour indices at which forecasts are issued in a split.

        The whole target window must lie inside the split, so no training
        target can overlap a validation or test target.
        """
        lo, hi = self.bounds(split)
        t = np.arange(max(lo, L - 1), hi - H)
        return t[np.isin(self.hours.hour[t], issue_hours)]

    def context(self, t_idx, length=L, values=None):
        """[monitors, issues, length] of observations ending at the issue time."""
        v = self.values if values is None else values
        return v[:, t_idx[:, None] - length + 1 + np.arange(length)[None, :]]

    def targets(self, t_idx):
        """[monitors, issues, HORIZON] of what was later observed."""
        return self.values[:, t_idx[:, None] + 1 + np.arange(H)[None, :]]

    def eligible(self, t_idx):
        """A forecast is required when the monitor was mostly reporting.

        Needs 80% of the context and at least one reading in the last 3 hours.
        """
        ctx = np.isfinite(self.context(t_idx))
        return (ctx.mean(axis=2) >= 0.8) & ctx[:, :, -3:].any(axis=2)

    def last_obs(self, t_idx):
        """Most recent reading within 3 hours of the issue time."""
        ctx = self.context(t_idx, 3)
        out = ctx[:, :, 2].copy()
        for back in (1, 0):
            out = np.where(np.isfinite(out), out, ctx[:, :, back])
        return out


def fill_context(ctx):
    """Forward-fill then back-fill gaps along the last axis.

    Imputing a model's *input* is the model's business. Targets are never filled.
    """
    out = ctx.copy()
    flat = out.reshape(-1, out.shape[-1])
    ok = np.isfinite(flat)
    idx = np.where(ok, np.arange(flat.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    filled = flat[np.arange(flat.shape[0])[:, None], idx]
    # Leading gaps have nothing before them: use the first reading in the row.
    first = np.argmax(ok, axis=1)
    lead = np.arange(flat.shape[1])[None, :] < first[:, None]
    filled = np.where(lead, flat[np.arange(flat.shape[0]), first][:, None], filled)
    return filled.reshape(out.shape)

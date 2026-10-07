"""Turning any forecast into an interval with a coverage guarantee.

Every model gets the same treatment. Its raw 80% interval is the pair of 0.1
and 0.9 quantiles (or a zero-width interval at the point forecast, for models
that give no quantiles). A conformity score measures by how much the truth
fell outside it:

    score = max(lower - y, y - upper)        (Romano et al., 2019)

and the interval is widened by a quantile of recent scores. Two schedules
are compared:

    static     scores from the first WARMUP days only, never refreshed
    adaptive   a sliding window of each monitor's own recent scores, with the
               target level nudged after every miss (Gibbs & Candes, 2021)

A forecast issued on day d for up to 48 hours ahead is only fully verified
on day d + 2, so the adaptive schedule uses scores from day d - 2 and earlier.
"""
import numpy as np

from . import config

ALPHA = round(1 - (config.INTERVAL[1] - config.INTERVAL[0]), 6)   # 0.2 for an 80% interval
WARMUP = 60          # days before calibrated intervals are judged
WINDOW = 60          # days of scores the adaptive schedule remembers
DELAY = 2            # days until a 48-hour forecast is fully verified
GAMMA = 0.02         # step size of the adaptive level
BUCKETS = [(0, 6), (6, 24), (24, 48)]                      # horizon index ranges that share a calibration


def raw_interval(f):
    """A model's own 80% interval: its 0.1 and 0.9 quantiles, or the point forecast twice if it has none."""
    if f["quantiles"] is None:
        return f["point"], f["point"]
    lv = config.QUANTILES
    return f["quantiles"][..., lv.index(config.INTERVAL[0])], f["quantiles"][..., lv.index(config.INTERVAL[1])]


def _row_quantile(pool, level):
    """Per-row empirical quantile of `pool` [rows, n] at per-row `level`, ignoring NaN."""
    srt = np.sort(pool, axis=1)                               # NaN sorts to the end
    n = np.isfinite(pool).sum(axis=1)
    k = np.clip(np.ceil((n + 1) * level).astype(int) - 1, 0, np.maximum(n - 1, 0))
    q = srt[np.arange(len(srt)), k]
    return np.where(n >= 30, q, np.nan)


def calibrate(lo, hi, y, valid):
    """Return {'static': (lo, hi), 'adaptive': (lo, hi)} on the forecast grid."""
    S, D, _ = y.shape
    score = np.where(valid, np.maximum(lo - y, y - hi), np.nan)
    out = {}

    # Static: one widening per monitor and horizon bucket, from the warm-up days.
    add = np.full_like(lo, np.nan)
    for a, b in BUCKETS:
        q = _row_quantile(score[:, :WARMUP, a:b].reshape(S, -1), np.full(S, 1 - ALPHA))
        add[:, :, a:b] = q[:, None, None]
    out["static"] = (lo - add, hi + add)

    # Adaptive: sliding window per monitor, with an adaptive miscoverage level.
    add = np.full_like(lo, np.nan)
    alpha = np.full((S, len(BUCKETS)), ALPHA)
    for d in range(D):
        r = d - DELAY                                          # the latest fully verified day
        for j, (a, b) in enumerate(BUCKETS):
            if r >= 0:
                v = valid[:, r, a:b]
                q_r = add[:, r, a:b]
                miss = (score[:, r, a:b] > np.where(np.isfinite(q_r), q_r, 0.0)) & v
                seen = v.sum(axis=1)
                err = np.where(seen > 0, miss.sum(axis=1) / np.maximum(seen, 1), ALPHA)
                alpha[:, j] = np.clip(alpha[:, j] + GAMMA * (ALPHA - err), 0.01, 0.99)
                pool = score[:, max(0, r - WINDOW + 1):r + 1, a:b].reshape(S, -1)
                add[:, d, a:b] = _row_quantile(pool, 1 - alpha[:, j])[:, None]
    out["adaptive"] = (lo - add, hi + add)
    return out


def interval_score(lo, hi, y):
    """Winkler score: width plus a penalty of 2/alpha per unit of miss. Lower is better."""
    return (hi - lo) + (2 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))

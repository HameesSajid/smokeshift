"""Checks that the forecasting setup cannot see the future.

Run with:  python tests/test_no_leakage.py   (or pytest)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airtrust import config, data, forecasting  # noqa: E402


def test_features_ignore_the_future():
    """Scrambling everything after a cut-off must not change earlier features."""
    df = data.load()
    cut = df.index[5000]
    corrupted = df.copy()
    rng = np.random.default_rng(0)
    after = corrupted.index > cut
    corrupted.loc[after] = rng.normal(1e6, 1e5, size=(after.sum(), corrupted.shape[1]))
    for h in config.HORIZONS:
        clean, _ = forecasting.build_features(df, h)
        dirty, _ = forecasting.build_features(corrupted, h)
        pd.testing.assert_frame_equal(clean.loc[:cut], dirty.loc[:cut])
        b_clean, b_dirty = forecasting.baselines(df, h), forecasting.baselines(corrupted, h)
        pd.testing.assert_frame_equal(b_clean.loc[:cut], b_dirty.loc[:cut])


def test_target_is_h_hours_ahead():
    df = data.load()
    for h in config.HORIZONS:
        _, y = forecasting.build_features(df, h)
        t = df.index[1000]
        expected = df[forecasting.TARGET].loc[t + pd.Timedelta(hours=h)]
        assert (np.isnan(y.loc[t]) and np.isnan(expected)) or y.loc[t] == expected


def test_training_targets_end_before_the_test_month():
    """No training row may have a target inside or after its test month."""
    df = data.load()
    for h in config.HORIZONS:
        tt = df.index + pd.Timedelta(hours=h)
        for start, end in forecasting.folds(df.index):
            train, test = tt < start, (tt >= start) & (tt < end)
            assert not (train & test).any()
            assert tt[train].max() < tt[test].min()
            # The last training origin is at least h hours before the test month.
            assert df.index[train].max() <= start - pd.Timedelta(hours=h)


def test_missing_marker_is_not_a_value():
    df = data.load()
    assert not (df == -200).any().any()
    assert df.index.is_monotonic_increasing and df.index.freqstr.lower() == "h"


def test_conformal_offset_gives_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    y = rng.normal(size=4000)
    lo, hi = np.full(4000, -0.5), np.full(4000, 0.5)  # far too narrow on purpose
    off = forecasting.cqr_offset(lo[:2000], hi[:2000], y[:2000], alpha=0.1)
    covered = ((y[2000:] >= lo[2000:] - off) & (y[2000:] <= hi[2000:] + off)).mean()
    assert 0.87 < covered < 0.93


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")

"""Checks on the benchmark protocol. They run on a small synthetic panel, so
no download is needed:

    python tests/test_smokeshift.py        (or: pytest)

The most important ones corrupt everything after a cut-off hour and assert
that nothing a model is allowed to see before the cut-off changes.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smokeshift import config, conformal, eda, features, metrics, naive, panel, spatial  # noqa: E402
from smokeshift.bench import Bench, H, L, fill_context  # noqa: E402


def synthetic_bench(n_sites=14, n_days=150, seed=0):
    """A small panel with a daily cycle, one travelling 'plume' and missing readings."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2021-01-01", periods=n_days * 24, freq="h")
    lat, lon = rng.uniform(35, 45, n_sites), rng.uniform(-95, -75, n_sites)
    t = np.arange(len(hours))
    base = 8 + 3 * np.sin(2 * np.pi * (t[None, :] + lon[:, None] / 15) / 24) + rng.gamma(2, 1.5, (n_sites, len(t)))
    plume = 120 * np.exp(-0.5 * ((t[None, :] - 1800 - 3 * (lon[:, None] + 95)) / 20) ** 2)
    values = (base + plume).astype("float32")
    values[rng.random(values.shape) < 0.03] = np.nan
    stations = pd.DataFrame({"site": [f"S{i:02d}" for i in range(n_sites)], "lat": lat, "lon": lon,
                             "region": "Midwest"})
    return Bench(values, hours, stations)


def corrupt_after(bench, cut):
    """A copy of the panel with every value after hour `cut` replaced by huge random numbers."""
    v = bench.values.copy()
    v[:, cut + 1:] = np.random.default_rng(1).normal(1e5, 1e4, v[:, cut + 1:].shape)
    return Bench(v, bench.hours, bench.stations)


CUT = 2400
T_IDX = np.arange(L + 30, CUT + 1, 24)          # issue times at or before the cut-off


def same(a, b):
    """Assert two arrays are identical, treating NaN as equal to NaN."""
    np.testing.assert_array_equal(np.nan_to_num(a, nan=-999.0), np.nan_to_num(b, nan=-999.0))


def test_tabular_features_ignore_the_future():
    """Tree features at or before the cut-off must not change when the future is corrupted."""
    clean = synthetic_bench()
    dirty = corrupt_after(clean, CUT)
    for use_spatial in (False, True):
        Xc, names, pc = features.base(clean, T_IDX, use_spatial)
        Xd, _, pd_ = features.base(dirty, T_IDX, use_spatial)
        same(Xc, Xd), same(pc, pd_)
        site, day, hz = features.all_rows(np.ones((clean.n_sites, len(T_IDX)), bool), 0, len(T_IDX))
        ok = T_IDX[day] + hz < clean.values.shape[1]
        rc, _, _ = features.rows(clean, T_IDX, Xc, pc, site[ok], day[ok], hz[ok])
        rd, _, _ = features.rows(dirty, T_IDX, Xd, pd_, site[ok], day[ok], hz[ok])
        same(rc, rd)
        assert rc.shape[1] == len(names) + len(features.HORIZON_NAMES)


def test_spatial_features_ignore_the_future():
    """Neighbour features and channels at or before the cut-off must not change either."""
    clean = synthetic_bench()
    dirty = corrupt_after(clean, CUT)
    same(spatial.features(clean, T_IDX)[0], spatial.features(dirty, T_IDX)[0])
    for a, b in zip(spatial.hourly_channels(clean), spatial.hourly_channels(dirty)):
        same(a[:, :CUT + 1], b[:, :CUT + 1])


def test_naive_forecasts_ignore_the_future():
    """The three reference forecasts must depend only on the past."""
    clean = synthetic_bench()
    dirty = corrupt_after(clean, CUT)
    same(naive.persistence(clean, T_IDX), naive.persistence(dirty, T_IDX))
    same(naive.seasonal_naive(clean, T_IDX), naive.seasonal_naive(dirty, T_IDX))
    model = naive.DampedPersistence().fit(clean, T_IDX[:60])
    same(model.predict(clean, T_IDX), model.predict(dirty, T_IDX))


def test_transformer_inputs_ignore_the_future():
    """The transformer's inputs, covariates and anchor must depend only on the past."""
    try:
        from smokeshift import neural
    except ImportError:
        return                                  # PyTorch not installed: nothing to check
    clean = synthetic_bench()
    dirty = corrupt_after(clean, CUT)
    s = np.repeat(np.arange(clean.n_sites), len(T_IDX)).astype("int32")
    t = np.tile(T_IDX, clean.n_sites).astype("int32")
    for use_spatial in (False, True):
        a = neural.Windows(clean, use_spatial).batch(s, t, "cpu")
        b = neural.Windows(dirty, use_spatial).batch(s, t, "cpu")
        for x, y in zip(a[:3], b[:3]):          # inputs, covariates, anchor (not the targets)
            np.testing.assert_array_equal(x.numpy(), y.numpy())


def test_targets_are_strictly_after_the_issue_time():
    """Targets run from one hour to 48 hours after the issue time; the context ends at it."""
    b = synthetic_bench()
    y = b.targets(T_IDX)
    assert y.shape == (b.n_sites, len(T_IDX), H)
    same(y[:, 3, 0], b.values[:, T_IDX[3] + 1])
    same(y[:, 3, -1], b.values[:, T_IDX[3] + H])
    same(b.context(T_IDX)[:, 3, -1], b.values[:, T_IDX[3]])


def test_real_splits_do_not_overlap():
    """On the real panel, train, validation and test target windows are disjoint and in order."""
    if not (config.PROCESSED / "panel.npz").exists():
        return
    b = Bench.load()
    train, valid, test = (b.issue_times(s, (0, 6, 12, 18)) for s in ("train", "valid", "test"))
    assert train.max() + H < valid.min() and valid.max() + H < test.min()
    assert b.hours[train.max() + H] < pd.Timestamp(config.TRAIN_END)
    assert b.hours[test.min()] >= pd.Timestamp(config.VALID_END)


def test_fill_context_only_looks_back():
    """Gap filling carries earlier values forward; only a leading gap takes the first reading."""
    x = np.array([[np.nan, 1.0, np.nan, np.nan, 4.0, np.nan]])
    np.testing.assert_array_equal(fill_context(x), [[1.0, 1.0, 1.0, 1.0, 4.0, 4.0]])


def test_adaptive_conformal_recovers_coverage_after_a_shift():
    """When noise triples half way, adaptive intervals return to 80% coverage and static ones do not."""
    rng = np.random.default_rng(0)
    S, D = 40, 400
    sd = np.where(np.arange(D) < 200, 1.0, 3.0)[None, :, None]       # noise triples half way
    y = (rng.normal(size=(S, D, H)) * sd).astype("float32")
    zero = np.zeros_like(y)
    cal = conformal.calibrate(zero, zero, y, np.ones_like(y, bool))
    late = slice(260, D)
    cover = {k: float(((y >= lo) & (y <= hi))[:, late].mean()) for k, (lo, hi) in cal.items()}
    assert abs(cover["adaptive"] - 0.8) < 0.03, cover
    assert cover["static"] < 0.5, cover          # calibrated before the shift, never refreshed


def test_pinball_and_interval_score():
    """The quantile loss and the interval score match hand calculations."""
    y = np.array([[[10.0]]])
    q = np.tile(np.array(config.QUANTILES) * 20, (1, 1, 1, 1))        # deciles 2, 4, ..., 18
    manual = np.mean([max(a * (10 - v), (a - 1) * (10 - v)) for a, v in zip(config.QUANTILES, q[0, 0, 0])])
    assert abs(metrics.pinball(y, q).mean() - manual) < 1e-5          # scores are computed in float32
    np.testing.assert_allclose(metrics.pinball_mean(y, q), metrics.pinball(y, q).mean(axis=-1), rtol=1e-5)
    assert conformal.interval_score(np.array(8.0), np.array(12.0), np.array(10.0)) == 4.0
    assert conformal.interval_score(np.array(8.0), np.array(12.0), np.array(14.0)) == 4.0 + 10 * 2.0


def test_benjamini_hochberg_known_values():
    """The false-discovery-rate adjustment matches a worked example."""
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(eda.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_panel_storage_round_trip():
    """Saving and loading the panel changes values by at most the 0.1 rounding."""
    b = synthetic_bench(n_sites=5, n_days=10)
    old = config.PROCESSED
    try:
        config.PROCESSED = Path(tempfile.mkdtemp())
        panel.save(b.values, b.stations, {"note": "test"})
        v, _, st = panel.load()
    finally:
        config.PROCESSED = old
    np.testing.assert_allclose(np.nan_to_num(v, nan=-1), np.nan_to_num(np.round(b.values, 1), nan=-1), atol=0.051)
    assert list(st["site"]) == list(b.stations["site"])


def test_distance_matrix():
    """Distances and bearings match known values for one degree of latitude and longitude at 40N."""
    d, bearing = spatial.distances_and_bearings(np.array([40.0, 40.0, 41.0]), np.array([-90.0, -89.0, -90.0]))
    assert abs(d[0, 1] - 85.2) < 1.0 and abs(d[0, 2] - 111.2) < 1.0      # 1 degree of longitude / latitude at 40N
    assert abs(bearing[0, 2]) < 1.0 and abs(bearing[0, 1] - 90) < 1.0     # north, east


def test_seed_summary_statistics():
    """The seed summary reports mean, sample standard deviation, range and run count per model and metric."""
    try:
        from smokeshift import seeds
    except ImportError:
        return                                  # PyTorch not installed: nothing to check
    runs = pd.DataFrame({"model": ["A"] * 3 + ["B"] * 3, "seed": [1, 2, 3] * 2,
                         "mae | All hours": [1.0, 2.0, 3.0, 5.0, 5.0, 5.0]})
    out = seeds.summarise(runs).set_index("model")
    assert out.loc["A", "mean"] == 2.0 and out.loc["A", "std"] == 1.0
    assert out.loc["A", "min"] == 1.0 and out.loc["A", "max"] == 3.0 and out.loc["A", "runs"] == 3
    assert out.loc["B", "std"] == 0.0


def test_api_reference_is_current():
    """docs/API_REFERENCE.md must match what the generator produces from the code as it is now."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_api_reference
    current = build_api_reference.OUT.read_text(encoding="utf-8") if build_api_reference.OUT.exists() else ""
    assert current == build_api_reference.build(), "run: python scripts/build_api_reference.py"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")

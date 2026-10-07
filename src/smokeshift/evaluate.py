"""Score every stored forecast on the common grid and write the result tables."""
import json

import numpy as np
import pandas as pd

from . import config, metrics, store
from .bench import Bench, H

REFERENCE = "Damped persistence"
AQI_EDGES = [-np.inf, 9.0, 35.4, 55.4, 125.4, np.inf]
AQI_LABELS = ["0-9\nGood", "9-35\nModerate", "35-55\nSensitive groups", "55-125\nUnhealthy", "over 125\nVery unhealthy"]


def common_grid(bench, t_idx):
    """Truth, and the cells on which every stored model produced a number."""
    y = bench.targets(t_idx)
    elig = bench.eligible(t_idx)
    for name in store.available():
        elig &= np.isfinite(store.load(name)["point"]).all(axis=2)
    return y, elig


def run(bench=None, log=print):
    """Score all stored models, one pass each, and write the score, horizon, bias and bootstrap tables."""
    bench = bench or Bench.load()
    t_idx = bench.issue_times("test")
    y, elig = common_grid(bench, t_idx)
    reg = metrics.regimes(bench, t_idx, y, elig)
    valid = reg["All hours"]
    scale = metrics.mase_scale(bench)
    years = bench.hours.year.to_numpy()[t_idx]
    region = bench.stations["region"].to_numpy()

    def extra_slices():
        """Further masks: horizon buckets, test years and regions."""
        for base in ("All hours", "Smoke at issue"):
            for h, hm in metrics.horizon_masks().items():
                yield f"{base} | {h}", reg[base] & hm[None, None, :]
        for yr in config.TEST_YEARS:
            in_year = (years == yr)[None, :, None]
            yield f"Year {yr}", valid & in_year
            yield f"Year {yr} | Smoke at issue", reg["Smoke at issue"] & in_year
        for r in sorted(set(region)):
            yield f"Region {r}", valid & (region == r)[:, None, None]

    by_regime, by_slice, bias_rows = [], [], []
    horizon = {k: {} for k in ("All hours", "Smoke at issue", "Calm at issue", "Onsets")}
    daily = {k: {} for k in reg}
    names = store.available()
    for name in names:
        s = metrics.ModelScores(name, store.load(name), y, scale)
        by_regime += [s.row(k, m) for k, m in reg.items()]
        by_slice += [r for k, m in extra_slices() if (r := s.row(k, m))]
        for k in horizon:
            horizon[k][name] = s.by_horizon(reg[k])
        for k, m in reg.items():
            daily[k][name] = s.daily_sum(m)
        for lo, hi, lab in zip(AQI_EDGES[:-1], AQI_EDGES[1:], AQI_LABELS):
            m = valid & (y > lo) & (y <= hi)
            bias_rows.append({"model": name, "level": lab, "bias": float(s.err[m].mean()),
                              "mae": float(s.abs[m].mean()), "n": int(m.sum())})
        del s

    config.TAB.mkdir(parents=True, exist_ok=True)
    by_regime = pd.DataFrame(by_regime)
    by_regime.to_csv(config.TAB / "scores_by_regime.csv", index=False)
    pd.DataFrame(by_slice).to_csv(config.TAB / "scores_by_slice.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(config.TAB / "bias_by_outcome.csv", index=False)
    index = pd.Index(np.arange(1, H + 1), name="horizon_h")
    for k, d in horizon.items():
        pd.DataFrame(d, index=index).to_csv(config.TAB / f"mae_by_horizon_{k.lower().replace(' ', '_')}.csv")
    boots = []
    for k in reg:
        b = metrics.bootstrap_relative(daily[k], REFERENCE)
        b.insert(0, "slice", k)
        boots.append(b)
    pd.concat(boots, ignore_index=True).to_csv(config.TAB / "relative_mae_bootstrap.csv", index=False)

    counts = {k: int(m.sum()) for k, m in reg.items()}
    summary = {"models": names, "issue_days": int(len(t_idx)), "monitors": int(bench.n_sites),
               "cells_scored": counts, "reference": REFERENCE,
               "forecasts_scored": int(elig.sum()),
               "first_issue": str(bench.hours[t_idx[0]]), "last_issue": str(bench.hours[t_idx[-1]])}
    (config.TAB / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("  cells: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    for metric in ("mae", "wql"):
        wide = by_regime.pivot(index="model", columns="slice", values=metric)[list(reg)].sort_values("All hours")
        log(f"  {metric.upper()} by regime:\n" + wide.round(3).to_string())
    return summary

"""Are the 80% intervals honest, and do they stay honest when smoke arrives?"""
import numpy as np
import pandas as pd

from . import config, conformal, evaluate, metrics, store
from .bench import Bench


def run(bench=None, log=print):
    """Calibrate every model's interval three ways; write coverage, width and interval score by slice and month."""
    bench = bench or Bench.load()
    t_idx = bench.issue_times("test")
    y, elig = evaluate.common_grid(bench, t_idx)
    reg = metrics.regimes(bench, t_idx, y, elig)
    valid = reg["All hours"]
    judged = (np.arange(len(t_idx)) >= conformal.WARMUP)[None, :, None]
    month = bench.hours[t_idx].strftime("%Y-%m").to_numpy()

    rows, monthly = [], []
    for name in store.available():
        f = store.load(name)
        lo, hi = (np.asarray(a) for a in conformal.raw_interval(f))
        variants = {"Raw": (lo, hi)} if f["quantiles"] is not None else {}
        variants |= {k.capitalize() + " conformal": v for k, v in conformal.calibrate(lo, hi, y, valid).items()}
        for vname, (l, h) in variants.items():
            inside = (y >= l) & (y <= h)
            score = conformal.interval_score(l, h, y)
            ok = np.isfinite(l) & judged
            for sname, m in reg.items():
                m = m & ok
                rows.append({"model": name, "method": vname, "slice": sname, "coverage": float(inside[m].mean()),
                             "mean_width": float((h - l)[m].mean()), "interval_score": float(score[m].mean()),
                             "n": int(m.sum())})
            for mo in np.unique(month):
                m = valid & ok & (month == mo)[None, :, None]
                if m.sum():
                    monthly.append({"model": name, "method": vname, "month": mo, "coverage": float(inside[m].mean()),
                                    "mean_width": float((h - l)[m].mean())})
        log(f"  {name}: done")
    table, monthly = pd.DataFrame(rows), pd.DataFrame(monthly)
    config.TAB.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TAB / "intervals.csv", index=False)
    monthly.to_csv(config.TAB / "intervals_monthly.csv", index=False)
    for metric in ("coverage", "interval_score"):
        show = table[table.slice == "All hours"].pivot(index="model", columns="method", values=metric)
        log(f"  {metric} of 80% intervals, all hours:\n" + show.round(3).to_string())
    return table

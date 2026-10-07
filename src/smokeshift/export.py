"""Export a compact JSON bundle for the interactive dashboard.

The full forecast store is several gigabytes. The dashboard needs a small,
honest sample of it: every monitor's location and shift statistics, the
result tables, and for a set of featured monitors the observations and each
model's day-ahead forecasts through the 2023 smoke season.
"""
import json

import numpy as np
import pandas as pd

from . import config, conformal, eda, store
from .bench import Bench

WINDOW = ("2023-05-28", "2023-07-06")
MODELS = ["Damped persistence", "XGBoost", "XGBoost + neighbours", "Transformer + neighbours", "Chronos-2", "TimesFM",
          "Ensemble", "Ensemble + foundation"]
N_FEATURED = 60


def _featured(bench):
    """Case-study cities first, then the monitors with the most smoke hours in the window, one per county."""
    st = bench.stations
    a, b = bench.idx(WINDOW[0]), bench.idx(WINDOW[1])
    hours_over = (bench.values[:, a:b] > config.USG).sum(axis=1)
    rows = [list(st["site"]).index(s) for s in eda.CASE_SITES.values()]
    names = {r: n for r, n in zip(rows, eda.CASE_SITES)}
    seen = {(st.state[r], st.county[r]) for r in rows}
    for r in np.argsort(-hours_over):
        key = (st.state[r], st.county[r])
        if key not in seen:
            rows.append(int(r)), seen.add(key)
        if len(rows) >= N_FEATURED:
            break
    return rows, names


def _round(a, nd=1):
    """Round to `nd` decimals as plain Python floats, with None for missing values (valid JSON)."""
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def build(bench=None):
    """Assemble the dashboard bundle: monitors, featured series and the result tables."""
    bench = bench or Bench.load()
    st = bench.stations
    t_idx = bench.issue_times("test")
    rows, names = _featured(bench)
    d0 = int(np.searchsorted(t_idx, bench.idx(WINDOW[0])))
    d1 = int(np.searchsorted(t_idx, bench.idx(WINDOW[1])))
    days = np.arange(d0, d1)
    a, b = t_idx[d0] + 1, t_idx[d1 - 1] + 25           # hours covered by day-ahead forecasts
    models = [m for m in MODELS if m in store.available()]

    series = {str(r): {"obs": _round(bench.values[r, a:b])} for r in rows}
    for m in models:
        f = store.load(m)
        lo, hi = conformal.raw_interval(f)
        for r in rows:
            s = series[str(r)].setdefault("models", {})
            s[m] = {"p": _round(np.asarray(f["point"][r, days, :24]).ravel())}
            if f["quantiles"] is not None:
                s[m]["lo"] = _round(np.asarray(lo[r, days, :24]).ravel())
                s[m]["hi"] = _round(np.asarray(hi[r, days, :24]).ravel())

    extra = st["hours_over_usg_2023"] - (st["hours_over_usg_2021"] + st["hours_over_usg_2022"]) / 2
    monitors = [{"i": int(i), "lat": round(float(r.lat), 3), "lon": round(float(r.lon), 3),
                 "name": names.get(i, f"{r.county}, {r.state}"), "region": r.region,
                 "h23": int(r.hours_over_usg_2023), "extra": round(float(extra[i]), 0),
                 "featured": i in rows} for i, r in enumerate(st.itertuples())]

    def records(name, nd=4):
        """A result table as a list of row dictionaries."""
        df = pd.read_csv(config.TAB / name).round(nd)
        return df.astype(object).where(df.notna(), None).to_dict("records")       # NaN is not valid JSON

    horizon = {k: pd.read_csv(config.TAB / f"mae_by_horizon_{k}.csv").round(3).to_dict("list")
               for k in ("all_hours", "smoke_at_issue", "calm_at_issue")}
    bundle = {
        "window": {"start": str(bench.hours[a]), "hours": int(b - a), "issue_hour_utc": config.ISSUE_HOUR_UTC},
        "threshold": config.USG,
        "models": models,
        "monitors": monitors,
        "series": series,
        "scores": records("scores_by_regime.csv"),
        "relative": records("relative_mae_bootstrap.csv"),
        "horizon": horizon,
        "alerts": records("alerts.csv"),
        "intervals": records("intervals.csv"),
        "summary": json.loads((config.TAB / "evaluation_summary.json").read_text(encoding="utf-8")),
        "panel": json.loads((config.PROCESSED / "panel_summary.json").read_text(encoding="utf-8")),
        "shift": records("eda_shift_by_region.csv", 3),
    }
    return bundle


SKELETON = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n{page}\n</body>\n</html>\n')


def run(log=print):
    """Write dashboard/data.js and states.js, and a standalone index.html around page.html."""
    out = config.ROOT / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    bundle = build()
    data = out / "data.js"
    data.write_text("window.SMOKESHIFT=" + json.dumps(bundle, separators=(",", ":"), allow_nan=False) + ";",
                    encoding="utf-8")
    states = config.RAW / "us-atlas" / "states-10m.json"
    (out / "states.js").write_text("window.US_STATES=" + states.read_text(encoding="utf-8") + ";", encoding="utf-8")
    (out / "index.html").write_text(SKELETON.format(page=(out / "page.html").read_text(encoding="utf-8")),
                                    encoding="utf-8")
    log(f"  data.js: {data.stat().st_size / 1e6:.1f} MB, {len(bundle['series'])} featured monitors, "
        f"{len(bundle['models'])} models. Open dashboard/index.html in a browser.")
    return out

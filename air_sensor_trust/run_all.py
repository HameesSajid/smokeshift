"""Run the whole analysis: python run_all.py

Writes figures, tables and predictions under outputs/, a machine-readable
outputs/results.json, and the self-contained outputs/report.html.

To rerun only some stages and reuse saved results for the rest, name them:
    python run_all.py eda patterns
Stages: eda, audit, calibration, forecasting, patterns.
"""
import functools
import json
import os
import sys
import time
from pathlib import Path

# Small tables: a few threads per model fit is faster than all cores.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).parent / "src"))
print = functools.partial(print, flush=True)

from airtrust import audit, calibration, config, data, eda, forecasting, patterns, plotting, report  # noqa: E402


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def main():
    t0 = time.time()
    for d in (config.FIG, config.TAB, config.PRED):
        d.mkdir(parents=True, exist_ok=True)
    plotting.setup()

    df = data.load()
    print(f"Loaded {len(df):,} hourly rows, {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")

    only = set(sys.argv[1:])
    saved = config.OUT / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if only and saved.exists() else {}
    stages = [("eda", "Exploratory analysis", lambda: eda.run(df)),
              ("audit", "Audit of the original notebook", lambda: audit.run(df)),
              ("calibration", "Sensor calibration and drift", lambda: calibration.run(df)),
              ("forecasting", "Forecasting, intervals and alerts", lambda: forecasting.run(df)),
              ("patterns", "Anomalies and daily regimes", lambda: patterns.run(df))]
    unknown = only - {key for key, _, _ in stages}
    if unknown:
        sys.exit(f"Unknown stage(s): {', '.join(sorted(unknown))}")
    for key, title, fn in stages:
        if only and key not in only and key in results:
            continue
        t = time.time()
        print(f"[{key}] {title} ...")
        with threadpool_limits(limits=4):
            results[key] = fn()
        print(f"[{key}] done in {time.time() - t:.0f}s")

    text = json.dumps(results, indent=2, default=_json_default)
    saved.write_text(text, encoding="utf-8")
    path = report.build(json.loads(text))
    print(f"Report: {path}")
    print(f"Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

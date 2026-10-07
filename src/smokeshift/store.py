"""Forecasts on disk: one folder per model, all on the same grid.

point.npy      [monitors, issue days, horizon]            the forecast scored with MAE
quantiles.npy  [monitors, issue days, horizon, levels]    optional, deciles 0.1 to 0.9, sorted
info.json      how the forecast was made

Arrays are memory-mapped on load. A quantile array is about 680 MB, so the
scoring code reads one model at a time and never holds two in memory.
"""
import json

import numpy as np

from . import config


def _dir(name, split):
    """Folder holding one model's forecast."""
    return config.FORECASTS / split / name


def save(name, point, quantiles=None, info=None, split="test"):
    """Write a model's point forecast, optional quantiles and metadata."""
    d = _dir(name, split)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "point.npy", point.astype("float32"))
    if quantiles is not None:
        np.save(d / "quantiles.npy", quantiles.astype("float32"))
    elif (d / "quantiles.npy").exists():
        (d / "quantiles.npy").unlink()
    (d / "info.json").write_text(json.dumps(info or {}, indent=2), encoding="utf-8")


def load(name, split="test"):
    """Memory-map a model's forecast; `quantiles` is None for point-only models."""
    d = _dir(name, split)
    q = d / "quantiles.npy"
    return {"point": np.load(d / "point.npy", mmap_mode="r"),
            "quantiles": np.load(q, mmap_mode="r") if q.exists() else None,
            "info": json.loads((d / "info.json").read_text(encoding="utf-8"))}


def available(split="test"):
    """Names of all models with a stored forecast."""
    root = config.FORECASTS / split
    return sorted(p.parent.name for p in root.glob("*/point.npy")) if root.exists() else []

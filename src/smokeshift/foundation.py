"""Zero-shot forecasts from pretrained time-series foundation models.

Nothing is trained here. Each model receives the same input as the other
own-history models (one monitor's recent readings) and is asked for the same
nine quantiles, 48 hours ahead. "Long context" variants give a model three
weeks of history, to check whether the shared 168-hour budget holds it back.

Models (weights are fetched from the Hugging Face Hub on first use, into the
project cache):

    Chronos-Bolt   amazon/chronos-bolt-base          Amazon, 2024
    Chronos-2      amazon/chronos-2                  Amazon, 2025
    TimesFM        google/timesfm-2.5-200m-pytorch   Google, 2025

Needs the optional packages chronos-forecasting and timesfm (see
scripts/setup_env.ps1). A model whose package is missing is skipped.
"""
import os
import time

import numpy as np

from . import config, store
from .bench import H

LONG_CONTEXT = 512

# name -> (family, Hugging Face repository, hours of history given to the model)
MODELS = {
    "Chronos-Bolt": ("bolt", "amazon/chronos-bolt-base", config.CONTEXT),
    "Chronos-Bolt (long context)": ("bolt", "amazon/chronos-bolt-base", LONG_CONTEXT),
    "Chronos-2": ("chronos2", "amazon/chronos-2", config.CONTEXT),
    "Chronos-2 (long context)": ("chronos2", "amazon/chronos-2", LONG_CONTEXT),
    "TimesFM": ("timesfm", "google/timesfm-2.5-200m-pytorch", config.CONTEXT),
    "TimesFM (long context)": ("timesfm", "google/timesfm-2.5-200m-pytorch", LONG_CONTEXT),
}
# What `python run.py foundation` runs unless SMOKESHIFT_FM names others (comma-separated).
DEFAULT = ["Chronos-2", "TimesFM", "Chronos-Bolt", "Chronos-2 (long context)"]


def _batches(bench, t_idx, length, batch):
    """Yield (monitor rows, day indices, context [B, length]) for every eligible forecast."""
    elig = bench.eligible(t_idx)
    s, d = np.nonzero(elig)
    for j in range(0, len(s), batch):
        sj, dj = s[j:j + batch], d[j:j + batch]
        t = t_idx[dj]
        idx = t[:, None] - length + 1 + np.arange(length)[None, :]
        ctx = np.where(idx >= 0, bench.values[sj[:, None], np.maximum(idx, 0)], np.nan)
        yield sj, dj, ctx.astype("float32")


def _empty(bench, t_idx):
    """A quantile array on the forecast grid, filled with NaN."""
    return np.full((bench.n_sites, len(t_idx), H, len(config.QUANTILES)), np.nan, dtype="float32")


def _chronos(bench, t_idx, model_id, length, family, batch=256):
    """Zero-shot quantiles from a Chronos-Bolt or Chronos-2 checkpoint. Missing inputs may stay NaN."""
    import torch
    from chronos import BaseChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=torch.float32)
    q = _empty(bench, t_idx)
    for sj, dj, ctx in _batches(bench, t_idx, length, batch):
        x = torch.from_numpy(ctx)
        # Chronos-Bolt takes a [B, L] tensor; Chronos-2 takes a list of series and returns a list.
        out, _ = pipe.predict_quantiles(inputs=x if family == "bolt" else list(x), prediction_length=H,
                                        quantile_levels=config.QUANTILES)
        if not isinstance(out, torch.Tensor):
            out = torch.stack([o.reshape(H, len(config.QUANTILES)) for o in out])
        q[sj, dj] = out.float().cpu().numpy()
    return q, device


def _timesfm(bench, t_idx, model_id, length, batch=256):
    """Zero-shot quantiles from a TimesFM 2.5 checkpoint. It interpolates missing inputs itself."""
    import timesfm
    import torch

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
    model.compile(timesfm.ForecastConfig(max_context=int(np.ceil(length / 32) * 32), max_horizon=64,
                                         normalize_inputs=True, use_continuous_quantile_head=True,
                                         fix_quantile_crossing=True, per_core_batch_size=64))
    q = _empty(bench, t_idx)
    for sj, dj, ctx in _batches(bench, t_idx, length, batch):
        _, full = model.forecast(horizon=H, inputs=list(ctx))
        q[sj, dj] = np.asarray(full)[:, :, 1:1 + len(config.QUANTILES)]       # column 0 is the mean
    return q, "cuda" if torch.cuda.is_available() else "cpu"


def forecast(bench, name, t_idx):
    """Run one configured model over the whole grid; returns (quantiles, info)."""
    family, model_id, length = MODELS[name]
    t0 = time.time()
    if family == "timesfm":
        q, device = _timesfm(bench, t_idx, model_id, length)
    else:
        q, device = _chronos(bench, t_idx, model_id, length, family)
    return np.sort(q, axis=-1), {"model_id": model_id, "context_hours": length, "device": device,
                                 "seconds": time.time() - t0, "zero_shot": True}


def run(bench, log=print, names=None):
    """Run the selected foundation models that are installed and store their forecasts."""
    chosen = names or [n.strip() for n in os.environ.get("SMOKESHIFT_FM", "").split(",") if n.strip()] or DEFAULT
    t_idx = bench.issue_times("test")
    for name in chosen:
        try:
            q, info = forecast(bench, name, t_idx)
        except ImportError as e:
            log(f"  {name}: skipped, {e.name} is not installed")
            continue
        store.save(name, q[..., len(config.QUANTILES) // 2], q, info)
        log(f"  {name}: {info['seconds'] / 60:.1f} min on {info['device']}")

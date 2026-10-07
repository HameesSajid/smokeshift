"""Gradient-boosted trees: one global model for all monitors and horizons.

A single XGBoost model is trained on rows from every monitor, with the
horizon as a feature, and emits nine quantiles at once. Three variants make
two controlled comparisons:

    own history, change label      the reference tree model
    + neighbour features           does spatial context help?      (same label)
    own history, level label       does the label choice matter?   (same features)
"""
import json

import numpy as np
import xgboost as xgb

from . import config, features, spatial, store
from .bench import H

VARIANTS = {
    "XGBoost": {"use_spatial": False, "label": "change"},
    "XGBoost + neighbours": {"use_spatial": True, "label": "change"},
    "XGBoost (level label)": {"use_spatial": False, "label": "level"},
}
TRAIN_ISSUE_HOURS = (0, 6, 12, 18)
PARAMS = {
    "objective": "reg:quantileerror", "quantile_alpha": config.QUANTILES, "tree_method": "hist",
    "max_depth": 8, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
    "max_bin": 256, "seed": config.SEED,
}


SEARCH_SPACE = {"max_depth": [5, 6, 8, 10], "eta": [0.03, 0.05, 0.08], "min_child_weight": [5, 20, 100, 300],
                "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [1.0, 5.0, 20.0]}
TUNING_FILE = config.TAB / "gbm_tuning.json"


def tuned_params():
    """The search winner if `python run.py tune` has been run, else the defaults."""
    if TUNING_FILE.exists():
        return {**PARAMS, **json.loads(TUNING_FILE.read_text(encoding="utf-8"))["best"]}
    return PARAMS


def tune(bench, log=print, n_trials=10, n_rows=700_000, variant="XGBoost + neighbours"):
    """Random search on the validation period (July to December 2022).

    The test years are never touched. All three variants then share the
    winning settings, so the two controlled comparisons stay controlled.
    """
    spec = VARIANTS[variant]
    rng = np.random.default_rng(config.SEED)
    nb = spatial.Neighbours(bench.stations)
    Xt, yt, names = _design(bench, bench.issue_times("train", TRAIN_ISSUE_HOURS), True, nb, 2, rng, spec["label"])
    Xv, yv, _ = _design(bench, bench.issue_times("valid"), True, nb, 6, rng, spec["label"])
    pick = rng.choice(len(yt), size=min(n_rows, len(yt)), replace=False)
    dev = device()
    dtrain = xgb.QuantileDMatrix(Xt[pick], yt[pick], feature_names=names)
    dvalid = xgb.QuantileDMatrix(Xv, yv, feature_names=names, ref=dtrain)
    default = {k: PARAMS.get(k, 1.0) for k in SEARCH_SPACE}
    trials = [default] + [{k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()} for _ in range(n_trials - 1)]
    rows = []
    for i, trial in enumerate(trials):
        booster = xgb.train({**PARAMS, **trial, "device": dev}, dtrain, 500 if dev == "cuda" else 150,
                            evals=[(dvalid, "valid")], early_stopping_rounds=25, verbose_eval=False)
        rows.append({**trial, "valid_pinball": float(booster.best_score), "best_round": int(booster.best_iteration + 1),
                     "is_default": i == 0})
        log(f"  trial {i + 1:>2}/{len(trials)}: pinball {booster.best_score:.4f} at round {booster.best_iteration + 1}  "
            + ", ".join(f"{k}={v}" for k, v in trial.items()))
    best = min(rows, key=lambda r: r["valid_pinball"])
    config.TAB.mkdir(parents=True, exist_ok=True)
    TUNING_FILE.write_text(json.dumps({"best": {k: best[k] for k in SEARCH_SPACE}, "trials": rows,
                                       "rows_used": int(len(pick))}, indent=2), encoding="utf-8")
    log(f"  best: pinball {best['valid_pinball']:.4f} (default {rows[0]['valid_pinball']:.4f})")
    return best


def device():
    """Use the GPU when XGBoost can; fall back to CPU."""
    try:
        d = xgb.DMatrix(np.zeros((8, 2), dtype="float32"), np.zeros(8, dtype="float32"))
        xgb.train({"tree_method": "hist", "device": "cuda"}, d, 1)
        return "cuda"
    except xgb.core.XGBoostError:
        return "cpu"


def _base_in_blocks(bench, t_idx, use_spatial, nb, block=300):
    """features.base over blocks of issue times, to keep the context tensor small."""
    Xs, profs, names = [], [], None
    for i in range(0, len(t_idx), block):
        X, names, prof = features.base(bench, t_idx[i:i + block], use_spatial, nb)
        Xs.append(X), profs.append(prof)
    return np.concatenate(Xs, axis=1), names, np.concatenate(profs, axis=1)


def _design(bench, t_idx, use_spatial, nb, per_origin, rng, label):
    """Training or validation rows: features, label (change or level) and feature names."""
    X, names, prof = _base_in_blocks(bench, t_idx, use_spatial, nb)
    site, day, hz = features.sample_training_rows(bench, t_idx, per_origin, rng)
    Xr, change, last = features.rows(bench, t_idx, X, prof, site, day, hz)
    ok = np.isfinite(change)
    y = change if label == "change" else change + last
    return Xr[ok], y[ok], names + features.HORIZON_NAMES


def fit(bench, name, log=print, rounds=600):
    """Train one variant with early stopping on the validation period; return the booster and a summary."""
    spec = VARIANTS[name]
    rng = np.random.default_rng(config.SEED)
    nb = spatial.Neighbours(bench.stations) if spec["use_spatial"] else None
    Xt, yt, names = _design(bench, bench.issue_times("train", TRAIN_ISSUE_HOURS), spec["use_spatial"], nb, 2, rng,
                            spec["label"])
    Xv, yv, _ = _design(bench, bench.issue_times("valid"), spec["use_spatial"], nb, 6, rng, spec["label"])
    dev = device()
    log(f"  {name}: {len(yt):,} training rows, {len(yv):,} validation rows, {len(names)} features, device={dev}")
    dtrain = xgb.QuantileDMatrix(Xt, yt, feature_names=names)
    dvalid = xgb.QuantileDMatrix(Xv, yv, feature_names=names, ref=dtrain)
    history = {}
    params = tuned_params()
    booster = xgb.train({**params, "device": dev}, dtrain, rounds if dev == "cuda" else min(rounds, 200),
                        evals=[(dvalid, "valid")], early_stopping_rounds=30, evals_result=history, verbose_eval=False)
    curve = history["valid"]["quantile"]
    log(f"  {name}: best round {booster.best_iteration + 1}, validation pinball {min(curve):.4f}")
    config.MODELS.mkdir(parents=True, exist_ok=True)
    booster.save_model(config.MODELS / f"{name}.json")
    info = {"best_round": int(booster.best_iteration + 1), "valid_pinball": float(min(curve)),
            "train_rows": int(len(yt)), "features": names, "device": dev, **spec,
            "params": {k: params[k] for k in SEARCH_SPACE if k in params}}
    return booster, info


def predict(bench, booster, name, t_idx, block=40):
    """Quantile forecasts on the benchmark grid: ([S, D, H] median, [S, D, H, 9] quantiles)."""
    spec = VARIANTS[name]
    nb = spatial.Neighbours(bench.stations) if spec["use_spatial"] else None
    elig = bench.eligible(t_idx)
    q = np.full((bench.n_sites, len(t_idx), H, len(config.QUANTILES)), np.nan, dtype="float32")
    best = (0, booster.best_iteration + 1)
    for lo in range(0, len(t_idx), block):
        hi = min(lo + block, len(t_idx))
        X, names, prof = features.base(bench, t_idx[lo:hi], spec["use_spatial"], nb)
        site, day, hz = features.all_rows(elig, lo, hi)
        Xr, _, last = features.rows(bench, t_idx[lo:hi], X, prof, site, day - lo, hz)
        pred = booster.predict(xgb.DMatrix(Xr, feature_names=names + features.HORIZON_NAMES), iteration_range=best)
        if spec["label"] == "change":
            pred = pred + last[:, None]
        q[site, day, hz - 1] = pred
    q = np.sort(q, axis=-1)
    return q[..., len(config.QUANTILES) // 2], q


def shap_summary(bench, booster, name, t_idx, n_rows=60_000):
    """Mean |SHAP| of each feature for the median forecast, on clean and elevated targets.

    Uses XGBoost's built-in exact TreeSHAP, on a random sample of test rows.
    """
    spec = VARIANTS[name]
    rng = np.random.default_rng(config.SEED)
    nb = spatial.Neighbours(bench.stations) if spec["use_spatial"] else None
    days = np.sort(rng.choice(len(t_idx), size=min(120, len(t_idx)), replace=False))
    X, names, prof = features.base(bench, t_idx[days], spec["use_spatial"], nb)
    site, day, hz = features.sample_training_rows(bench, t_idx[days], 4, rng)
    Xr, change, last = features.rows(bench, t_idx[days], X, prof, site, day, hz)
    ok = np.isfinite(change)
    # Keep every elevated row (they are rare) plus a random sample of the rest.
    elevated = ok & (change + last > config.USG)
    rest = np.flatnonzero(ok & ~elevated)
    pick = np.concatenate([np.flatnonzero(elevated), rng.choice(rest, size=min(n_rows, len(rest)), replace=False)])
    all_names = names + features.HORIZON_NAMES
    contrib = booster.predict(xgb.DMatrix(Xr[pick], feature_names=all_names), pred_contribs=True,
                              iteration_range=(0, booster.best_iteration + 1))
    median = np.abs(contrib[:, len(config.QUANTILES) // 2, :-1])       # drop the bias column
    is_elev = elevated[pick]
    return {"features": all_names, "clean": median[~is_elev].mean(axis=0).tolist(),
            "elevated": median[is_elev].mean(axis=0).tolist(),
            "n_clean": int((~is_elev).sum()), "n_elevated": int(is_elev.sum())}


def run(bench, log=print):
    """Train, predict and store all three variants, with SHAP summaries for the change-label ones."""
    t_test = bench.issue_times("test")
    out = {}
    for name in VARIANTS:
        booster, info = fit(bench, name, log)
        point, q = predict(bench, booster, name, t_test)
        store.save(name, point, q, info)
        if VARIANTS[name]["label"] == "change":
            info["shap"] = shap_summary(bench, booster, name, t_test)
        out[name] = info
    config.TAB.mkdir(parents=True, exist_ok=True)
    (config.TAB / "gbm_training.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out

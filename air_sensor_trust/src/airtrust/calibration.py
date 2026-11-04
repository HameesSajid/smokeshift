"""Calibrating the low-cost sensor array against the reference analysers.

Scenario: the sensor box sits next to a reference station for a few weeks
(co-location), a calibration model is fitted, and the box is then trusted on
its own. The questions are how good that calibration is, how fast it goes
stale, whether staleness can be spotted without reference data, and what
recalibration schedule fixes it.

Time is cut into 4-week blocks. Blocks 0 and 1 (the first 8 weeks) are the
co-location period; every later block is evaluated strictly out of sample.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, plotting as P

FEATS = config.SENSORS + config.WEATHER
# Reference pollutant -> the sensor channel nominally sold to measure it.
TARGETS = {"co_ref": "s1_co", "nox_ref": "s3_nox", "no2_ref": "s4_no2"}
BLOCK_DAYS = 28
N_REF_BLOCKS = 2

SINGLE, RIDGE, GBM = "Single sensor, linear", "All sensors + weather, ridge", "All sensors + weather, boosted trees"


def _models(nominal):
    return {
        SINGLE: make_pipeline(ColumnTransformer([("s", "passthrough", [nominal])]), LinearRegression()),
        RIDGE: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        GBM: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=20, random_state=config.SEED),
    }


def _frame(df, target):
    d = df[FEATS + [target]].dropna().copy()
    d["block"] = (d.index - df.index.min()).days // BLOCK_DAYS
    return d


def _block_label(df, k):
    start = df.index.min() + pd.Timedelta(days=int(k) * BLOCK_DAYS)
    return start.strftime("%b %d")


def static_calibration(df):
    """Fit once on the co-location period, then never touch it again."""
    rows, per_block = [], []
    for target, nominal in TARGETS.items():
        d = _frame(df, target)
        train, test = d[d.block < N_REF_BLOCKS], d[d.block >= N_REF_BLOCKS]
        for name, model in _models(nominal).items():
            model.fit(train[FEATS], train[target])
            pred = pd.Series(model.predict(test[FEATS]), index=test.index)
            err = pred - test[target]
            rows.append({"target": config.LABELS[target], "model": name,
                         "mae": mean_absolute_error(test[target], pred),
                         "mae_pct_of_mean": mean_absolute_error(test[target], pred) / test[target].mean() * 100,
                         "r2": r2_score(test[target], pred), "bias": err.mean(),
                         "n_train": len(train), "n_test": len(test)})
            g = pd.DataFrame({"abs": err.abs(), "err": err, "block": test.block}).groupby("block")
            for k, grp in g:
                per_block.append({"target": target, "model": name, "block": int(k),
                                  "mae": grp["abs"].mean(), "bias": grp["err"].mean(), "n": len(grp)})
    return pd.DataFrame(rows), pd.DataFrame(per_block)


def recalibration_strategies(df):
    """Static vs rolling (last 8 weeks) vs expanding (all history) refits."""
    rows = []
    for target, nominal in TARGETS.items():
        d = _frame(df, target)
        for model_name in (RIDGE, GBM):
            static = _models(nominal)[model_name].fit(d.loc[d.block < N_REF_BLOCKS, FEATS],
                                                      d.loc[d.block < N_REF_BLOCKS, target])
            for k in sorted(b for b in d.block.unique() if b >= N_REF_BLOCKS):
                test = d[d.block == k]
                if len(test) < 24:
                    continue
                windows = {"Static (first 8 weeks only)": None,
                           "Rolling (refit on last 8 weeks)": d[(d.block >= k - N_REF_BLOCKS) & (d.block < k)],
                           "Expanding (refit on all history)": d[d.block < k]}
                for strategy, train in windows.items():
                    if train is None:
                        m = static
                    elif len(train) < 100:
                        continue
                    else:
                        m = _models(nominal)[model_name].fit(train[FEATS], train[target])
                    rows.append({"target": target, "model": model_name, "strategy": strategy, "block": int(k),
                                 "mae": mean_absolute_error(test[target], m.predict(test[FEATS])), "n": len(test)})
    return pd.DataFrame(rows)


def _psi(ref, cur, bins=10):
    """Population stability index of `cur` against `ref` on ref-quantile bins."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_signals(df):
    """Label-free drift monitors: need only the sensor inputs, no reference.

    PSI and the KS statistic look at one feature at a time. The domain
    classifier asks a single question of all features together: can a model
    tell this month's inputs from the co-location inputs? AUC 0.5 means no.
    """
    d = df[FEATS].dropna().copy()
    d["block"] = (d.index - df.index.min()).days // BLOCK_DAYS
    ref = d[d.block < N_REF_BLOCKS]
    psi_rows, summary = [], []
    for k in sorted(b for b in d.block.unique() if b >= N_REF_BLOCKS):
        cur = d[d.block == k]
        if len(cur) < 100:
            continue
        psi = {f: _psi(ref[f].to_numpy(), cur[f].to_numpy()) for f in FEATS}
        ks = {f: float(stats.ks_2samp(ref[f], cur[f]).statistic) for f in FEATS}
        X = pd.concat([ref[FEATS], cur[FEATS]])
        y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
        clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=8, random_state=config.SEED)
        cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
        auc = roc_auc_score(y, cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1])
        psi_rows.append({"block": int(k), **psi})
        summary.append({"block": int(k), "mean_psi": float(np.mean(list(psi.values()))),
                        "max_ks": max(ks.values()), "domain_auc": float(auc)})
    return pd.DataFrame(psi_rows).set_index("block"), pd.DataFrame(summary)


def colocation_learning_curve(df, weeks=(1, 2, 4, 8, 12, 16)):
    """How long must the box sit next to a reference station?

    Every training length is scored on the same later window (weeks 17-24).
    """
    t0 = df.index.min()
    rows = []
    for target, nominal in TARGETS.items():
        d = df[FEATS + [target]].dropna()
        day = (d.index - t0).days
        test = d[(day >= 112) & (day < 168)]
        for w in weeks:
            train = d[day < 7 * w]
            if len(train) < 50:
                continue
            for name in (RIDGE, GBM):
                m = _models(nominal)[name].fit(train[FEATS], train[target])
                rows.append({"target": config.LABELS[target], "model": name, "weeks": w, "n_train": len(train),
                             "mae": mean_absolute_error(test[target], m.predict(test[FEATS]))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- figures


def fig_strategies(df, strat):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    colors = {"Static (first 8 weeks only)": P.MUTED, "Rolling (refit on last 8 weeks)": P.BLUE,
              "Expanding (refit on all history)": P.ORANGE}
    short = {"Static (first 8 weeks only)": "Static", "Rolling (refit on last 8 weeks)": "Rolling",
             "Expanding (refit on all history)": "Expanding"}
    for ax, target in zip(axes, TARGETS):
        s = strat[(strat.target == target) & (strat.model == RIDGE)]
        blocks = sorted(s.block.unique())
        for strategy, color in colors.items():
            g = s[s.strategy == strategy].set_index("block")["mae"].reindex(blocks)
            ax.plot(range(len(blocks)), g.values, color=color, marker="o", markersize=4, label=strategy)
            P.end_label(ax, len(blocks) - 1, g.values[-1], short[strategy])
        ax.set_xticks(range(len(blocks))[::2], [_block_label(df, b) for b in blocks][::2], rotation=45, ha="right")
        ax.set_xlim(-0.3, len(blocks) + 1.6)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(f"MAE, {config.LABELS[target]}")
        ax.set_title(config.LABELS[target].split(" (")[0])
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 0.93))
    fig.suptitle("Calibration error over the year for three refit schedules (ridge, 4-week blocks)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return P.save(fig, "05_calibration_strategies")


def fig_drift(df, psi, static_blocks, drift):
    """Left: how far the inputs moved. Right: whether that predicted the error."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    im = ax.imshow(psi.T.values, aspect="auto", cmap=P.SEQ, vmin=0, vmax=max(1.0, float(np.nanpercentile(psi.values, 95))))
    labels = [_block_label(df, b) for b in psi.index]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATS)), [config.LABELS[f].split(" (")[0] if f in config.WEATHER else config.LABELS[f]
                                      for f in FEATS])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="PSI vs co-location period", pad=0.02)
    ax.set_title("Input drift per feature (PSI above 0.25 is the usual alarm level)")
    ax = axes[1]
    for (target, _), color, marker in zip(TARGETS.items(), [P.BLUE, P.ORANGE, P.AQUA], ["o", "s", "^"]):
        s = static_blocks[(static_blocks.target == target) & (static_blocks.model == RIDGE)].merge(drift, on="block")
        rel = s["mae"] / s["mae"].iloc[0]
        ax.scatter(s["mean_psi"], rel, color=color, marker=marker, s=34, edgecolor=P.SURFACE, linewidth=1,
                   label=config.LABELS[target].split(" (")[0], zorder=3)
    ax.set_xlabel("Mean PSI of the block's inputs")
    ax.set_ylabel("Stale-model error, relative to first block")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2)
    ax.set_title("Does input drift predict calibration error?")
    fig.tight_layout()
    return P.save(fig, "06_input_drift")


def run(df):
    static, static_blocks = static_calibration(df)
    strat = recalibration_strategies(df)
    psi, drift = drift_signals(df)
    curve = colocation_learning_curve(df)

    overall = (strat.assign(w=strat.mae * strat.n).groupby(["target", "model", "strategy"])
               .apply(lambda g: g.w.sum() / g.n.sum(), include_groups=False).rename("mae").reset_index())
    overall["target"] = overall["target"].map(config.LABELS)

    # Does label-free drift predict the error of the stale model?
    link = []
    for target in TARGETS:
        s = static_blocks[(static_blocks.target == target) & (static_blocks.model == RIDGE)].merge(drift, on="block")
        for signal in ("mean_psi", "domain_auc"):
            rho, p = stats.spearmanr(s[signal], s["mae"])
            link.append({"target": config.LABELS[target], "signal": signal, "spearman_rho": rho, "p_value": p,
                         "n_blocks": len(s)})
    link = pd.DataFrame(link)

    for name, t in [("calibration_static", static), ("calibration_static_by_block", static_blocks),
                    ("calibration_strategies_by_block", strat), ("calibration_strategies_overall", overall),
                    ("drift_psi", psi.reset_index()), ("drift_summary", drift),
                    ("drift_vs_error", link), ("colocation_learning_curve", curve)]:
        t.to_csv(config.TAB / f"{name}.csv", index=False)

    return {
        "static": static.to_dict("records"),
        "strategies_overall": overall.to_dict("records"),
        "drift_summary": drift.to_dict("records"),
        "drift_vs_error": link.to_dict("records"),
        "learning_curve": curve.to_dict("records"),
        "psi_max_by_feature": psi.max().to_dict(),
        "figures": {"strategies": fig_strategies(df, strat), "drift": fig_drift(df, psi, static_blocks, drift)},
    }

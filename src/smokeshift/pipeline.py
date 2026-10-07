"""The stages of the benchmark, in the order they run."""
import numpy as np

from . import config, download, evaluate, gbm, naive, panel, store
from .bench import Bench


def naive_stage(log=print):
    """Fit damped persistence on the training period and store the three reference forecasts."""
    bench = Bench.load()
    t_test = bench.issue_times("test")
    not_eligible = ~bench.eligible(t_test)[:, :, None]
    damped = naive.DampedPersistence().fit(bench, bench.issue_times("train", gbm.TRAIN_ISSUE_HOURS))
    log("  damped persistence, share of the anomaly kept at 1, 6, 24, 48 h: "
        + ", ".join(f"{damped.rho[h - 1]:.2f}" for h in (1, 6, 24, 48)))
    for name, f in [("Persistence", naive.persistence(bench, t_test)),
                    ("Seasonal naive", naive.seasonal_naive(bench, t_test)),
                    ("Damped persistence", damped.predict(bench, t_test))]:
        store.save(name, np.where(not_eligible, np.nan, f), info={"rho": damped.rho.tolist()} if "Damped" in name else {})


def tune_stage(log=print):
    """Random search for the tree models on the validation period."""
    gbm.tune(Bench.load(), log)


def gbm_stage(log=print):
    """Train and store the three XGBoost variants."""
    gbm.run(Bench.load(), log)


def neural_stage(log=print):
    """Train and store the two transformer variants."""
    from . import neural          # imported here so the other stages run without PyTorch
    neural.run(Bench.load(), log)


def foundation_stage(log=print):
    """Run the zero-shot foundation models that are installed."""
    from . import foundation      # optional dependencies; models that are not installed are skipped
    foundation.run(Bench.load(), log)


# Members were fixed before any foundation model had been scored on the test
# period: the two neighbour-aware trained models, and for the second ensemble
# the newest zero-shot model from each laboratory as well.
ENSEMBLES = {
    "Ensemble": ["XGBoost + neighbours", "Transformer + neighbours"],
    "Ensemble + foundation": ["XGBoost + neighbours", "Transformer + neighbours", "Chronos-2", "TimesFM"],
}


def ensemble_stage(log=print):
    """Average the quantiles of each ensemble's members (equal weights, nothing fitted)."""
    have = store.available()
    for name, members in ENSEMBLES.items():
        if not all(m in have for m in members):
            log(f"  {name}: skipped, missing " + ", ".join(m for m in members if m not in have))
            continue
        q = np.zeros(store.load(members[0])["quantiles"].shape, dtype="float32")
        for m in members:                       # one member in memory at a time
            q += np.asarray(store.load(m)["quantiles"])
        q /= len(members)
        store.save(name, q[..., q.shape[-1] // 2], q, {"members": members})
        log(f"  {name}: " + ", ".join(members))


def evaluate_stage(log=print):
    """Score every stored forecast."""
    evaluate.run(log=log)


def eda_stage(log=print):
    """Shift statistics and exploratory figures."""
    from . import eda
    eda.run(log=log)


def alerts_stage(log=print):
    """Next-day unhealthy-air alerts."""
    from . import alerts
    alerts.run(log=log)


def uncertainty_stage(log=print):
    """Raw, static-conformal and adaptive-conformal intervals for every model."""
    from . import uncertainty
    uncertainty.run(log=log)


def figures_stage(log=print):
    """Draw the result charts."""
    from . import figures
    figures.run(log=log)


def report_stage(log=print):
    """Write outputs/report.html."""
    from . import report
    report.build(log=log)


def dashboard_stage(log=print):
    """Write the dashboard data bundle and the standalone page."""
    from . import export
    export.run(log=log)


STAGES = {
    "download": ("Fetch raw EPA hourly PM2.5 files", download.run),
    "panel": ("Build and audit the monitor-by-hour panel", panel.run),
    "eda": ("Statistics of the 2023 shift and its spatial structure", eda_stage),
    "naive": ("Reference forecasts", naive_stage),
    "tune": ("Random search for the tree models, on the validation period", tune_stage),
    "gbm": ("Gradient-boosted trees, with and without neighbours", gbm_stage),
    "neural": ("Patch transformer, with and without neighbours", neural_stage),
    "foundation": ("Zero-shot foundation models (optional packages)", foundation_stage),
    "ensemble": ("Equal-weight quantile average of the strongest models", ensemble_stage),
    "evaluate": ("Score every model on the common grid", evaluate_stage),
    "uncertainty": ("Prediction intervals: raw, static conformal, adaptive conformal", uncertainty_stage),
    "alerts": ("Next-day unhealthy-air alerts", alerts_stage),
    "figures": ("Result charts", figures_stage),
    "report": ("Write outputs/report.html", report_stage),
    "dashboard": ("Export the data bundle for the interactive dashboard", dashboard_stage),
}

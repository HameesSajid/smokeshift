# Concepts: what was used, what was not, and why this is enough

This document answers three questions for every part of the project:

1. Which data science concept is used, and where in the code?
2. What else could have been used?
3. Why is the chosen approach sufficient for the question this project asks?

The question is narrow on purpose: **when wildfire smoke pushes air quality outside anything in the training
data, which kinds of forecast hold up, and how should that be measured?** A concept earns its place only if it
helps answer that. Section 12 lists what is honestly missing.

---

## 1. Data engineering

| Used | Where |
|---|---|
| Chunked reading of 35.6 million rows from zipped CSV | `panel._read_year` |
| Reshaping long records into a dense monitor-by-hour matrix | `panel.build` |
| Completeness rule (at least 85% of hours in every year) | `config.MIN_COMPLETENESS`, `panel.build` |
| Range validation of readings | `config.VALID_RANGE` |
| Compact storage (tenths of a µg/m³ as int16, with a sentinel for missing) | `panel.save` |
| Checksums of the raw files | `download.run`, `data/processed/raw_manifest.json` |
| Memory-mapped arrays for results larger than memory | `store.load` |

**Could also be used:** a database or data warehouse (DuckDB, BigQuery), a columnar format (Parquet), a dataflow
framework (Spark, Dask), a data version control tool (DVC), a feature store.

**Why this is enough:** the whole cleaned panel is 538 by 35,064 numbers and fits in 75 MB of memory. A distributed
engine would add moving parts and no speed. The one place where size really matters, 5 GB of stored forecasts, is
handled with memory-mapped NumPy files, which need no extra dependency. Checksums plus a download script give
reproducibility without a versioning tool. If the project grew to all low-cost sensors in the country (tens of
thousands of series), Parquet and DuckDB would be the first things to add.

## 2. Missing data

| Used | Where |
|---|---|
| Never imputing a target; unobserved hours are simply not scored | `bench.targets`, `evaluate.common_grid` |
| Forward-fill of model inputs with an explicit "was observed" mask channel | `bench.fill_context`, `neural.Windows` |
| Native missing-value handling in trees | XGBoost, `features.base` leaves NaN in place |
| Eligibility rule (80% of the context present, a reading in the last 3 hours) | `bench.eligible` |

**Could also be used:** model-based imputation (Kalman smoothing, matrix factorisation, multiple imputation by
chained equations), or deep imputers.

**Why this is enough:** only 3.1% of cells are missing and gaps are short. Imputing targets would invent the
numbers the models are scored against, so it is ruled out on principle. For inputs, a mask lets the model learn
what a gap means instead of trusting a guess. Model-based imputation becomes worth its complexity when gaps are
long or frequent, which is not the case for monitors selected for completeness.

## 3. Exploratory statistics and inference

| Used | Where |
|---|---|
| Distribution summaries by region and year, tail percentiles | `eda.shift_by_region` |
| Paired non-parametric test (Wilcoxon signed-rank) | `eda.paired_shift_test` |
| Family-wise error control (Holm) | `eda.holm` |
| False discovery rate control (Benjamini-Hochberg) across 538 tests | `eda.benjamini_hochberg`, `eda.weekday_effect_fdr` |
| Two-sample distances (Kolmogorov-Smirnov, Wasserstein) | `eda.distribution_distance` |
| Block bootstrap for dependent data | `metrics.bootstrap_relative` |

**Could also be used:** parametric tests (t-tests, ANOVA), Bayesian hierarchical models of the regional effect,
permutation tests, extreme value theory (generalised Pareto tails) for the upper tail, formal change-point
detection.

**Why this is enough:** PM2.5 is heavily skewed, so rank-based tests are the safe default and need no
distributional assumption. The monitors are the same in both years, which makes a paired test the correct design
and removes between-monitor differences. With 538 simultaneous tests, a multiple-testing correction is required,
and showing both Bonferroni and FDR makes the trade-off visible. The effects are so large (98% of Northeast
monitors moved the same way) that a hierarchical Bayesian model would refine the estimate without changing any
conclusion. Extreme value theory is the most defensible addition: it would give a principled statement of how
unprecedented 2023 was. It is listed as future work.

## 4. Spatial and spatio-temporal analysis

| Used | Where |
|---|---|
| Great-circle distances and bearings | `spatial.distances_and_bearings` |
| Correlogram: correlation as a function of distance | `eda.correlogram` |
| Principal components of the space-time matrix (EOF analysis) | `eda.eof` |
| Lagged cross-correlation to time a plume's arrival | `eda.plume_lags` |
| Inverse-distance weighting, distance bands and compass quadrants as features | `spatial.features` |

**Could also be used:** kriging or Gaussian process regression, graph neural networks over the monitor network,
convolutional or attention models over gridded fields, satellite aerosol data, physical transport models
(HYSPLIT trajectories, the NOAA HRRR-Smoke forecast).

**Why this is enough:** the project asks whether spatial context helps and by how much, not which spatial model is
best. Sixteen hand-built neighbour features answer that with a controlled comparison: the same model with and
without them. They are also directly interpretable in the SHAP figure. A graph neural network is the natural next
step and would probably do better, but it would add a third thing to explain when the first two (trees and a
transformer, each with and without neighbours) already give the same answer. Physical transport models and
satellite data are a different kind of information and are the real route to predicting onsets; see section 12.

## 5. Forecasting baselines

| Used | Where |
|---|---|
| Persistence | `naive.persistence` |
| Seasonal naive | `naive.seasonal_naive` |
| Damped persistence: daily profile plus a fitted decay of the current anomaly | `naive.DampedPersistence` |

**Could also be used:** ARIMA and seasonal ARIMA, exponential smoothing (ETS), Theta, TBATS, Prophet.

**Why this is enough:** a baseline's job is to be hard to beat for its cost. Damped persistence is a one-parameter
version of what an AR(1) model with a seasonal mean does, fitted globally, and it beats plain persistence by 12%
here. Fitting 538 separate ARIMA models would take far longer, would need a library that is not otherwise used,
and classical per-series models are known to lose to global models on large panels of related series (the M5
competition result). Leaving them out is a deliberate scope decision and the main gap a reviewer from classical
statistics would point to.

## 6. Supervised learning

| Used | Where |
|---|---|
| Gradient-boosted decision trees | `gbm.py` (XGBoost) |
| One global model across all series, with the horizon as a feature | `features.rows`, `gbm.fit` |
| Quantile regression with the pinball loss, nine quantiles at once | `gbm.PARAMS` |
| Feature engineering: lags, rolling statistics, calendar, location | `features.base` |
| Differenced target (predict the change, not the level), with an ablation | `gbm.VARIANTS` |
| Random search with early stopping on a held-out period | `gbm.tune` |
| GPU training | `gbm.device` |

**Could also be used:** linear and regularised regression, random forests, LightGBM or CatBoost, Bayesian
hyperparameter optimisation (Optuna), automated machine learning, target encoding of monitor identity,
monotonic constraints.

**Why this is enough:** gradient-boosted trees are the strongest general-purpose method for tabular data and the
standard strong baseline in forecasting competitions. LightGBM and CatBoost are the same family and would differ
by a fraction of a percent. Random search over ten settings improved validation loss by only 0.3%, which shows the
model is not sensitive to its settings, so a more elaborate optimiser has nothing to find. Monitor identity is
left out on purpose: a model that memorises monitors cannot be moved to a new one.

## 7. Deep learning

| Used | Where |
|---|---|
| Transformer encoder over patches of the series (PatchTST style) | `neural.PatchTransformer` |
| Variance-stabilising transform (asinh), anchoring on the latest reading | `neural.to_z`, `PatchTransformer.forward` |
| Masked loss for missing targets; mask channel for missing inputs | `neural.pinball_loss`, `neural.Windows` |
| AdamW, one-cycle learning rate, gradient clipping, early stopping | `neural.fit` |
| Multi-channel input for spatial context | `neural.Windows` |

**Could also be used:** recurrent networks (LSTM, GRU, DeepAR), temporal convolutional networks, N-BEATS and
N-HiTS, the Temporal Fusion Transformer, linear models such as DLinear, graph neural networks, diffusion models
for probabilistic forecasting, mixed-precision or distributed training.

**Why this is enough:** one well-built neural model is needed to represent the family, and the patch transformer
is the current reference architecture for this task. The original project used an LSTM; a transformer is both
more current and trains faster. The purpose is a fair comparison with trees under identical information, not a
search for the best architecture. A linear baseline of the DLinear kind would be a cheap and informative addition,
because published work has shown such models can match transformers; it is listed in section 12.

## 8. Foundation models and transfer

| Used | Where |
|---|---|
| Zero-shot evaluation of pretrained time-series models (Amazon Chronos-Bolt and Chronos-2, Google TimesFM 2.5) | `foundation.py` |
| Controlling the information budget (same 168-hour context, plus a long-context variant) | `foundation.MODELS` |
| A pre-specified ensemble that includes them | `pipeline.ENSEMBLES` |

**Could also be used:** fine-tuning the foundation models on the training period, parameter-efficient tuning
(LoRA), other pretrained models (Moirai, Lag-Llama, TimeGPT), feeding covariates, large language models used as
forecasters.

**Why this is enough:** zero-shot is the claim these models make, so it is the claim worth testing first.
Fine-tuning answers a different question (how good can they become with my data) and would blur the comparison
with models trained from scratch. Three models from two laboratories are enough to see whether the result is
about the approach or about one model. TimesFM and Chronos-2 land between the strongest naive rule and the
models trained on this data on every fair slice, the older Chronos-Bolt only matches the naive rule, and the
order of the three is the same on every fair slice. One limit cannot be removed by any method: the pretraining data of
these models are not fully documented and may include the test years.

## 9. Evaluation design

| Used | Where |
|---|---|
| Split by time, with a validation period between train and test | `config.TRAIN_END`, `config.VALID_END` |
| A fixed forecast grid so every model is scored on identical cells | `bench.py`, `evaluate.common_grid` |
| Scale-free and probabilistic scores (MASE, weighted quantile loss) | `metrics.py` |
| Strata defined by information available at forecast time | `metrics.regimes` |
| Explicit demonstration of the forecaster's dilemma | `evaluate.run`, report section 6 |
| Controlled ablations (neighbours on or off; level or change label) | `gbm.VARIANTS`, `neural.VARIANTS` |
| Automated leakage tests | `tests/test_smokeshift.py` |

**Could also be used:** rolling-origin evaluation with retraining at each origin, k-fold or blocked
cross-validation, the continuous ranked probability score in closed form, threshold-weighted scores for extremes,
Diebold-Mariano tests, statistical tests on per-monitor win rates.

**Why this is enough:** the scientific point of the project is that models are frozen before the shift and then
tested through it. Retraining at each origin would let the models see smoke before being tested on smoke and would
destroy the experiment. Random k-fold cross-validation is invalid for time series. The weighted quantile loss
over nine quantiles is a discrete approximation of the continuous ranked probability score, so adding the latter
would repeat the same information. The day-block bootstrap plays the role of a Diebold-Mariano test while also
respecting dependence between monitors, which the classical test does not. Threshold-weighted scores are the
proper way to emphasise extremes and would be the most valuable addition to this section.

## 10. Uncertainty quantification

| Used | Where |
|---|---|
| Quantile regression | trees and transformer |
| Conformalized quantile regression (static split conformal) | `conformal.calibrate` |
| Adaptive conformal inference with delayed feedback | `conformal.calibrate` |
| Coverage, width and the Winkler interval score | `uncertainty.py`, `conformal.interval_score` |
| Reliability of predicted probabilities | `alerts.reliability` |

**Could also be used:** Bayesian neural networks, Monte Carlo dropout, deep ensembles, Gaussian processes,
parametric predictive distributions, isotonic or Platt recalibration of the alert probabilities.

**Why this is enough:** conformal prediction gives a coverage guarantee that holds without assumptions about the
model or the data distribution, and its adaptive form is designed for exactly the situation studied here, a
distribution that moves. Bayesian methods give uncertainty that is only as good as the model's assumptions, which
are the thing a distribution shift breaks. The honest result, that no method covers onsets, would not change
with a different interval method, because the limit is missing information, not calibration. Recalibrating the
classifier's probabilities is a small, clearly useful step that was left undone.

## 11. Classification, explainability, ensembling, software

| Used | Where |
|---|---|
| Rare-event classification scored with PR-AUC, recall at fixed precision and Brier skill | `alerts.py` |
| Exact TreeSHAP feature attributions, split by regime | `gbm.shap_summary` |
| Equal-weight quantile averaging across model families | `pipeline.ensemble_stage` |
| Package layout, staged command-line interface, tests, continuous integration | `run.py`, `pipeline.py`, `tests/`, `.github/` |
| Interactive visualisation | `dashboard/` |
| Generated API reference with an up-to-date check in the test suite | `scripts/build_api_reference.py` |

**Could also be used:** cost-sensitive learning and resampling for the rare class, attention maps or integrated
gradients for the transformer, stacking with a learned combiner, experiment tracking (MLflow, Weights and
Biases), containers, a model-serving endpoint, workflow orchestration.

**Why this is enough:** ROC-AUC is deliberately not the headline, because with 0.6% positives it stays high for
weak models; precision-recall is the correct lens. Equal weights were chosen for the ensemble because learned
weights need a validation period that resembles the test period, and here it does not. Experiment tracking and
serving infrastructure matter when many people run many experiments or when a model is deployed; this is a
single-author benchmark with fixed seeds whose outputs are tables under version control.

---

## 12. What is missing, stated plainly

Ordered by how much each would add.

1. **Meteorology and smoke transport.** Wind, boundary-layer height, and forecasts from physical smoke models are
   what would let a model predict an onset before the plume reaches any monitor. This is the largest limit of
   the project and no statistical method removes it.
2. **Threshold-weighted proper scores** for extremes, to evaluate the tail without falling into the outcome
   selection trap.
3. **A linear deep baseline** (DLinear style) and **per-series classical models** (seasonal ARIMA, ETS) for
   completeness of the comparison.
4. **A graph neural network** over the monitor network, as a learned replacement for the hand-built neighbour
   features.
5. **Fine-tuned foundation models**, to separate what pretraining gives from what local data gives.
6. **Extreme value analysis** of how unusual 2023 was.
7. **Recalibration of alert probabilities** and a cost-based choice of alert threshold.
8. **More than one shift.** Western fire seasons (2020, 2021) could serve as further test cases with earlier data.

None of these would change the three central findings: learned models beat naive ones but not at onsets, the way
the test set is sliced can reverse a leaderboard, and spatial context is what helps where a monitor's own history
is silent.

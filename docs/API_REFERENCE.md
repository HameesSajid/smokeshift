# API reference

Generated from the source by `python scripts/build_api_reference.py`. Do not edit by hand.
For the narrative explanation of how the pieces fit together, read `CODE_GUIDE.md` first.

## `run.py`

43 lines.

Command-line entry point.

python run.py                 list the stages
python run.py panel naive     run the named stages in order
python run.py all             run every stage

**Constants**

| Name | Value |
|---|---|
| `print` | `functools.partial(print, flush=True)` |

#### `main()`

Parse the stage names from the command line and run them in order.

## `src/smokeshift/__init__.py`

9 lines.

smokeshift: which forecasting paradigm holds up when wildfire smoke arrives?

A benchmark of naive rules, gradient-boosted trees with spatial context, a
transformer trained from scratch and zero-shot time-series foundation models
on hourly PM2.5 from US EPA monitors, with the 2023 smoke season as a natural
distribution shift.


## `src/smokeshift/alerts.py`

144 lines.

Health alerts: will the next day be unhealthy for sensitive groups?

The event is a 24-hour mean PM2.5 above 35.4 µg/m³ (the US AQI category
"Unhealthy for Sensitive Groups" or worse), for the day starting at the issue
time (hours 1-24) and for the day after (hours 25-48).

Two questions are asked of every method:

* ranking: does a higher score mean the event is more likely? (PR-AUC)
* onset: can it warn when the air is still clean at issue time? This is the
  case a warning system exists for. Persistence cannot do it by construction.

A classifier trained directly on the event gives probabilities, fitted with
and without the neighbour features. Whether those probabilities can be taken
at face value is checked with a reliability table rather than assumed.

**Constants**

| Name | Value |
|---|---|
| `DAYS` | `{"Day 1 (hours 1-24)": (0, 24), "Day 2 (hours 25-48)": (24, 48)}` |
| `MIN_HOURS` | `18` |
| `PARAMS` | `{"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist", "max_de...` |

#### `events(bench, t_idx)`

{day: (label [S, D] float with NaN where unknown)}, plus the onset mask [S, D].

#### `_design(bench, t_idx, use_spatial, nb)`

Flatten the horizon-independent feature tensor to one row per (monitor, issue time).

#### `fit_classifiers(bench, log=print)`

Probability of each event, from features known at the issue time.

#### `_recall_at_precision(y, score, precision=0.5)`

Largest share of events caught while at least `precision` of the alerts raised are correct.

#### `_row(method, day, slice_name, y, score, is_prob)`

One line of the results table: ranking skill of one method on one event day and slice.

#### `reliability(y, p, edges=(0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001))`

Predicted probability against observed frequency, in fixed probability bins holding at least 30 cases.

#### `run(bench=None, log=print)`

Fit the classifiers, score every method on both event days and both slices, and write the alert tables.

## `src/smokeshift/bench.py`

101 lines.

The benchmark protocol: who is forecast, when, and from what information.

Every model answers the same question. At an issue time t it may use the
observations up to and including t (at most CONTEXT hours of its own monitor,
and for spatial models the same span at other monitors) and must predict the
next HORIZON hours. Forecasts are stored on a fixed grid

    [monitor, issue day, horizon]

so that any two models are compared on exactly the same cells.

**Constants**

| Name | Value |
|---|---|
| `H` | `config.HORIZON` |
| `L` | `config.CONTEXT` |

#### class `Bench`

The panel, plus the indexing helpers that define what a forecast may see.

#### `Bench.load(cls)`

Read the processed panel from data/processed.

#### `Bench.n_sites(self)`

Number of monitors.

#### `Bench.idx(self, ts)`

Hour index of a timestamp (hours since the start of the panel).

#### `Bench.bounds(self, split)`

First and one-past-last hour index of the train, valid or test period.

#### `Bench.issue_times(self, split, issue_hours=(config.ISSUE_HOUR_UTC,))`

Hour indices at which forecasts are issued in a split.

The whole target window must lie inside the split, so no training
target can overlap a validation or test target.

#### `Bench.context(self, t_idx, length=L, values=None)`

[monitors, issues, length] of observations ending at the issue time.

#### `Bench.targets(self, t_idx)`

[monitors, issues, HORIZON] of what was later observed.

#### `Bench.eligible(self, t_idx)`

A forecast is required when the monitor was mostly reporting.

Needs 80% of the context and at least one reading in the last 3 hours.

#### `Bench.last_obs(self, t_idx)`

Most recent reading within 3 hours of the issue time.

#### `fill_context(ctx)`

Forward-fill then back-fill gaps along the last axis.

Imputing a model's *input* is the model's business. Targets are never filled.

## `src/smokeshift/config.py`

49 lines.

Paths and the experimental design, in one place.

**Constants**

| Name | Value |
|---|---|
| `ROOT` | `Path(__file__).resolve().parents[2]` |
| `PROCESSED` | `ROOT / "data" / "processed"` |
| `OUT` | `ROOT / "outputs"` |
| `FIG` | `OUT / "figures"` |
| `TAB` | `OUT / "tables"` |
| `CACHE` | `Path(os.environ.get("SMOKESHIFT_CACHE", ROOT / ".cache"))` |
| `RAW` | `CACHE / "raw"` |
| `FORECASTS` | `CACHE / "forecasts"` |
| `MODELS` | `CACHE / "models"` |
| `SEED` | `7` |
| `YEARS` | `[2021, 2022, 2023, 2024]` |
| `EPA_URL` | `"https://aqs.epa.gov/aqsweb/airdata/hourly_88101_{year}.zip"` |
| `MIN_COMPLETENESS` | `0.85` |
| `VALID_RANGE` | `(-10.0, 1000.0)` |
| `TRAIN_END` | `"2022-07-01"` |
| `VALID_END` | `"2023-01-01"` |
| `TEST_YEARS` | `[2023, 2024]` |
| `CONTEXT` | `168` |
| `HORIZON` | `48` |
| `ISSUE_HOUR_UTC` | `12` |
| `QUANTILES` | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]` |
| `INTERVAL` | `(0.1, 0.9)` |
| `HORIZON_BUCKETS` | `{"1-6 h": (1, 6), "7-24 h": (7, 24), "25-48 h": (25, 48)}` |
| `USG` | `35.4` |


## `src/smokeshift/conformal.py`

83 lines.

Turning any forecast into an interval with a coverage guarantee.

Every model gets the same treatment. Its raw 80% interval is the pair of 0.1
and 0.9 quantiles (or a zero-width interval at the point forecast, for models
that give no quantiles). A conformity score measures by how much the truth
fell outside it:

    score = max(lower - y, y - upper)        (Romano et al., 2019)

and the interval is widened by a quantile of recent scores. Two schedules
are compared:

    static     scores from the first WARMUP days only, never refreshed
    adaptive   a sliding window of each monitor's own recent scores, with the
               target level nudged after every miss (Gibbs & Candes, 2021)

A forecast issued on day d for up to 48 hours ahead is only fully verified
on day d + 2, so the adaptive schedule uses scores from day d - 2 and earlier.

**Constants**

| Name | Value |
|---|---|
| `ALPHA` | `round(1 - (config.INTERVAL[1] - config.INTERVAL[0]), 6)` |
| `WARMUP` | `60` |
| `WINDOW` | `60` |
| `DELAY` | `2` |
| `GAMMA` | `0.02` |
| `BUCKETS` | `[(0, 6), (6, 24), (24, 48)]` |

#### `raw_interval(f)`

A model's own 80% interval: its 0.1 and 0.9 quantiles, or the point forecast twice if it has none.

#### `_row_quantile(pool, level)`

Per-row empirical quantile of `pool` [rows, n] at per-row `level`, ignoring NaN.

#### `calibrate(lo, hi, y, valid)`

Return {'static': (lo, hi), 'adaptive': (lo, hi)} on the forecast grid.

#### `interval_score(lo, hi, y)`

Winkler score: width plus a penalty of 2/alpha per unit of miss. Lower is better.

## `src/smokeshift/download.py`

73 lines.

Fetch the raw EPA files into the cache directory.

Source: US EPA Air Quality System pre-generated data files (public domain),
https://aqs.epa.gov/aqsweb/airdata/download_files.html
Parameter 88101 is PM2.5 measured by federal reference or equivalent methods.

**Constants**

| Name | Value |
|---|---|
| `STATES_URL` | `"https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-10m.json"` |

#### `_sha256(path)`

Checksum of a file, read in 1 MB pieces.

#### `fetch(year, log=print)`

Download one year's EPA file into the cache, skipping it if a complete copy is already there.

#### `fetch_states(log=print)`

State outlines for the dashboard map (US Census Bureau cartographic boundaries, via us-atlas).

#### `run(log=print)`

Fetch every raw file and write their sizes and checksums to data/processed/raw_manifest.json.

## `src/smokeshift/eda.py`

323 lines.

What the data say before any model is fitted.

Each function answers one question with a statistic and, where it helps, a
figure:

    shift_by_region      how different was 2023, and where?
    paired_shift_test    is the change real at the level of individual monitors?
    distribution_distance   how far did the whole distribution move?
    weekday_effect_fdr   a multiple-testing example across 538 monitors
    correlogram          how far does air quality stay correlated?
    eof                  what are the dominant spatial patterns?
    plume_lags           does smoke arrive at upwind monitors first?

**Constants**

| Name | Value |
|---|---|
| `REGION_ORDER` | `["Northeast", "Midwest", "South", "West"]` |
| `CASE_SITES` | `{"Minneapolis": "27-053-0962", "Chicago": "17-031-4007", "Detroit": "26-163-0001", "Cle...` |

#### `daily_means(bench, min_hours=18)`

Daily mean per monitor ([monitors, days]); missing where fewer than `min_hours` readings exist.

#### `holm(p)`

Holm step-down adjustment of a family of p-values (controls the chance of any false positive).

#### `benjamini_hochberg(p)`

Adjusted p-values controlling the false discovery rate.

#### `shift_by_region(bench)`

Mean, median, upper percentiles and share of hours over the threshold, by region and year.

#### `paired_shift_test(bench)`

Per monitor: hours above the threshold in 2023 against each other year.

The same monitors are measured in both years, so the test is paired
(Wilcoxon signed-rank), which removes between-monitor differences.

#### `distribution_distance(bench, n=400_000)`

How far the hourly distribution moved from the training period, by region.

#### `weekday_effect_fdr(bench)`

Weekday vs weekend at every monitor: 538 tests, so control false discoveries.

#### `_anomalies(daily)`

Log daily means, standardised per monitor; missing days become 0 (the mean).

#### `correlogram(bench, bins=np.arange(0, 2001, 100))`

Correlation of daily anomalies between monitor pairs, by distance.

#### `eof(bench, n_modes=10)`

Principal components of the monitor-by-day anomaly matrix (EOF analysis).

#### `plume_lags(bench, start="2023-06-25", end="2023-07-02", max_lag=96)`

Hours by which each downwind city trails the first one, by cross-correlation.

#### `fig_map(bench)`

Map of monitors coloured by extra unhealthy hours in 2023.

#### `fig_shift(shift)`

Bars of the share of hours over the threshold, by region and year.

#### `fig_case(bench)`

Hourly series in eight cities during the late-June 2023 plume.

#### `fig_correlogram(cg)`

Correlation against distance, for the training period and for the smoke season.

#### `fig_eof(bench, e)`

Loadings of the leading principal component on a map, and its strength over time.

#### `run(bench=None, log=print)`

Compute every statistic, draw the five figures and write the eda tables.

## `src/smokeshift/evaluate.py`

91 lines.

Score every stored forecast on the common grid and write the result tables.

**Constants**

| Name | Value |
|---|---|
| `REFERENCE` | `"Damped persistence"` |
| `AQI_EDGES` | `[-np.inf, 9.0, 35.4, 55.4, 125.4, np.inf]` |
| `AQI_LABELS` | `["0-9\nGood", "9-35\nModerate", "35-55\nSensitive groups", "55-125\nUnhealthy", "over 1...` |

#### `common_grid(bench, t_idx)`

Truth, and the cells on which every stored model produced a number.

#### `run(bench=None, log=print)`

Score all stored models, one pass each, and write the score, horizon, bias and bootstrap tables.

- nested helper `extra_slices()`: Further masks: horizon buckets, test years and regions.

## `src/smokeshift/export.py`

116 lines.

Export a compact JSON bundle for the interactive dashboard.

The full forecast store is several gigabytes. The dashboard needs a small,
honest sample of it: every monitor's location and shift statistics, the
result tables, and for a set of featured monitors the observations and each
model's day-ahead forecasts through the 2023 smoke season.

**Constants**

| Name | Value |
|---|---|
| `WINDOW` | `("2023-05-28", "2023-07-06")` |
| `MODELS` | `["Damped persistence", "XGBoost", "XGBoost + neighbours", "Transformer + neighbours", "...` |
| `N_FEATURED` | `60` |
| `SKELETON` | `'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n' '<meta name="view...` |

#### `_featured(bench)`

Case-study cities first, then the monitors with the most smoke hours in the window, one per county.

#### `_round(a, nd=1)`

Round to `nd` decimals as plain Python floats, with None for missing values (valid JSON).

#### `build(bench=None)`

Assemble the dashboard bundle: monitors, featured series and the result tables.

- nested helper `records(name, nd=4)`: A result table as a list of row dictionaries.

#### `run(log=print)`

Write dashboard/data.js and states.js, and a standalone index.html around page.html.

## `src/smokeshift/features.py`

99 lines.

Tabular features for the tree models.

One row is (monitor, issue time, horizon). Everything in a row is known at
the issue time; `tests/test_no_leakage.py` corrupts the future and checks
that no feature moves.

The label is the *change* from the latest reading, not the level. The
reasoning: trees cannot predict a value outside the range of their training
labels, so a tree trained on levels is capped near the highest level it has
seen, while a tree predicting the change can ride on top of whatever the
current reading is. The "XGBoost (level label)" variant in gbm.py tests that
reasoning rather than assuming it; the report gives the result.

**Constants**

| Name | Value |
|---|---|
| `LAGS` | `[1, 2, 3, 6, 12, 24, 48, 72, 167]` |
| `EPOCH_DOW` | `4` |
| `HORIZON_NAMES` | `["horizon", "seasonal_minus_last", "profile_minus_last", "target_solar_sin", "target_so...` |

#### `base(bench, t_idx, use_spatial=False, nb=None)`

Horizon-independent features: ([monitors, issues, F], names, profile).

#### `rows(bench, t_idx, X, profile, site, day, horizon)`

Assemble design rows for the chosen (monitor, issue, horizon) triples.

site, day, horizon are equal-length integer arrays. Returns (features, label, last).
The label is NaN where the target was not observed.

#### `sample_training_rows(bench, t_idx, per_origin, rng)`

Pick `per_origin` random horizons for every eligible (monitor, issue) pair.

#### `all_rows(elig, day_lo, day_hi)`

Every horizon for every eligible pair in a block of issue days.

## `src/smokeshift/figures.py`

235 lines.

Result figures, drawn from the tables that the evaluation stages wrote.

**Constants**

| Name | Value |
|---|---|
| `NAIVE` | `["Persistence", "Seasonal naive", "Damped persistence"]` |
| `HEADLINE` | `["Damped persistence", "XGBoost + neighbours", "Transformer + neighbours", "Chronos-2",...` |

#### `_order(models)`

Sort model names into the fixed display order used in every chart.

#### `fig_leaderboard()`

Relative MAE with day-block bootstrap intervals, in three forecast-time strata.

#### `fig_horizon()`

MAE against hours ahead, for all hours and for forecasts issued in smoke.

#### `fig_dilemma()`

Bias by what was later observed: every model under-predicts the worst hours.

#### `fig_spatial_ablation()`

Own history against own history plus neighbours, for each model family.

#### `fig_coverage(model)`

Monthly coverage of raw, static-conformal and adaptive-conformal 80% intervals for one model.

#### `fig_alerts()`

PR-AUC of every alert method, for all days and for onsets.

#### `fig_shap(model="XGBoost + neighbours", top=12)`

Top features by mean absolute SHAP value, for clean and for elevated target hours.

#### `fig_case_forecast(bench, site="42-101-0057", city="Philadelphia", start="2023-06-04", days=7)`

Day-ahead forecasts through the 6-8 June 2023 episode, for the headline models.

#### `run(bench=None, log=print)`

Draw every result chart and record the file names in outputs/tables/figures.json.

## `src/smokeshift/foundation.py`

116 lines.

Zero-shot forecasts from pretrained time-series foundation models.

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

**Constants**

| Name | Value |
|---|---|
| `LONG_CONTEXT` | `512` |
| `MODELS` | `{ "Chronos-Bolt": ("bolt", "amazon/chronos-bolt-base", config.CONTEXT), "Chronos-Bolt (...` |
| `DEFAULT` | `list(MODELS)` |

#### `_batches(bench, t_idx, length, batch)`

Yield (monitor rows, day indices, context [B, length]) for every eligible forecast.

#### `_empty(bench, t_idx)`

A quantile array on the forecast grid, filled with NaN.

#### `_chronos(bench, t_idx, model_id, length, family, batch=256)`

Zero-shot quantiles from a Chronos-Bolt or Chronos-2 checkpoint. Missing inputs may stay NaN.

#### `_timesfm(bench, t_idx, model_id, length, batch=256)`

Zero-shot quantiles from a TimesFM 2.5 checkpoint. It interpolates missing inputs itself.

#### `forecast(bench, name, t_idx)`

Run one configured model over the whole grid; returns (quantiles, info).

#### `run(bench, log=print, names=None)`

Run the selected foundation models that are installed and store their forecasts.

## `src/smokeshift/gbm.py`

192 lines.

Gradient-boosted trees: one global model for all monitors and horizons.

A single XGBoost model is trained on rows from every monitor, with the
horizon as a feature, and emits nine quantiles at once. Three variants make
two controlled comparisons:

    own history, change label      the reference tree model
    + neighbour features           does spatial context help?      (same label)
    own history, level label       does the label choice matter?   (same features)

**Constants**

| Name | Value |
|---|---|
| `VARIANTS` | `{ "XGBoost": {"use_spatial": False, "label": "change"}, "XGBoost + neighbours": {"use_s...` |
| `TRAIN_ISSUE_HOURS` | `(0, 6, 12, 18)` |
| `PARAMS` | `{ "objective": "reg:quantileerror", "quantile_alpha": config.QUANTILES, "tree_method": ...` |
| `SEARCH_SPACE` | `{"max_depth": [5, 6, 8, 10], "eta": [0.03, 0.05, 0.08], "min_child_weight": [5, 20, 100...` |
| `TUNING_FILE` | `config.TAB / "gbm_tuning.json"` |

#### `tuned_params()`

The search winner if `python run.py tune` has been run, else the defaults.

#### `tune(bench, log=print, n_trials=10, n_rows=700_000, variant="XGBoost + neighbours")`

Random search on the validation period (July to December 2022).

The test years are never touched. All three variants then share the
winning settings, so the two controlled comparisons stay controlled.

#### `device()`

Use the GPU when XGBoost can; fall back to CPU.

#### `_base_in_blocks(bench, t_idx, use_spatial, nb, block=300)`

features.base over blocks of issue times, to keep the context tensor small.

#### `_design(bench, t_idx, use_spatial, nb, per_origin, rng, label)`

Training or validation rows: features, label (change or level) and feature names.

#### `fit(bench, name, log=print, rounds=600)`

Train one variant with early stopping on the validation period; return the booster and a summary.

#### `predict(bench, booster, name, t_idx, block=40)`

Quantile forecasts on the benchmark grid: ([S, D, H] median, [S, D, H, 9] quantiles).

#### `shap_summary(bench, booster, name, t_idx, n_rows=60_000)`

Mean |SHAP| of each feature for the median forecast, on clean and elevated targets.

Uses XGBoost's built-in exact TreeSHAP, on a random sample of test rows.

#### `run(bench, log=print)`

Train, predict and store all three variants, with SHAP summaries for the change-label ones.

## `src/smokeshift/metrics.py`

123 lines.

Scoring. Every function takes forecasts on the benchmark grid.

Three accuracy measures are reported because each answers a different question:

* MAE in µg/m³: how far off, in units a person can picture.
* MASE: MAE divided by the in-sample error of the seasonal naive forecast at
  the same monitor, so that clean and dirty sites weigh equally.
* WQL (weighted quantile loss): accuracy of the whole predictive
  distribution, the standard score for probabilistic forecasts.

Uncertainty for differences between models comes from a block bootstrap over
issue days: all monitors on a day are resampled together, because a smoke
plume makes errors on the same day strongly dependent across monitors.

**Constants**

| Name | Value |
|---|---|
| `Q` | `np.array(config.QUANTILES, dtype="float32")` |

#### `regimes(bench, t_idx, y, elig)`

Boolean masks [monitors, days, horizon] for conditional verification.

Two kinds of slice, and the difference matters:

* By what was known at issue time ("Calm at issue", "Smoke at issue").
  These are fair strata: a forecaster could have been told which one
  applies before forecasting.
* By what then happened ("Elevated hours", "Onsets"). These select on the
  outcome. They show where the damage is, but a score computed only on
  extreme outcomes rewards any forecast that leans high (the "forecaster's
  dilemma", Lerch et al. 2017), so they are not used to rank models.

#### `horizon_masks()`

Boolean masks over the 48 horizons for the buckets 1-6 h, 7-24 h and 25-48 h.

#### `mase_scale(bench)`

Per-monitor MAE of 'same hour yesterday' over the training period.

#### `pinball(y, q)`

Pinball loss per level; y [...], q [..., levels]. For small arrays.

#### `pinball_mean(y, q)`

Pinball loss averaged over levels, one level at a time to bound memory.

#### class `ModelScores`

Everything needed from one model, computed in a single pass over its forecast.

#### `ModelScores.__init__(self, name, forecast, y, scale)`

Compute error, absolute error and pinball loss for one model, once.

#### `ModelScores.row(self, slice_name, m)`

MAE, bias, MASE and WQL on the cells selected by mask `m`.

#### `ModelScores.by_horizon(self, m)`

MAE at each of the 48 horizons on the cells selected by `m`.

#### `ModelScores.daily_sum(self, m)`

Sum of absolute errors per issue day on the cells selected by `m` (input to the bootstrap).

#### `bootstrap_relative(daily, reference, n_boot=1000, seed=config.SEED)`

MAE of each model relative to `reference`, with a day-block bootstrap CI.

daily maps model -> per-day sum of absolute errors on a fixed set of cells.

## `src/smokeshift/naive.py`

61 lines.

Reference forecasts that need no learning, or almost none.

A learned model has to beat all three to be worth its complexity.

**Constants**

| Name | Value |
|---|---|
| `HOURS` | `np.arange(1, H + 1)` |

#### `persistence(bench, t_idx)`

Tomorrow looks like right now.

#### `seasonal_naive(bench, t_idx)`

The same clock hour on the most recent day already observed.

#### `_profile_and_anomaly(bench, t_idx)`

Seven-day mean for each clock hour, and how far 'now' sits from it.

#### class `DampedPersistence`

Daily profile plus a decaying share of the current anomaly.

forecast(h) = profile(clock hour of h) + rho_h * (now - profile(now))

rho_h is one number per horizon, fitted by least squares on the training
period. rho = 1 would be persistence of the anomaly, rho = 0 the profile.

#### `DampedPersistence.fit(self, bench, t_idx)`

Estimate rho for each horizon by least squares over the given issue times.

#### `DampedPersistence.predict(self, bench, t_idx)`

Forecast every monitor and issue time, falling back to persistence where the profile is missing.

## `src/smokeshift/neural.py`

242 lines.

A patch transformer for multi-horizon quantile forecasting, written in PyTorch.

Architecture (in the spirit of PatchTST, Nie et al. 2023):

    168 h of history  ->  21 patches of 8 h  ->  linear embedding
    + one token for calendar and location
    -> 3 transformer encoder layers
    -> flatten -> 48 horizons x 9 quantiles

Three choices matter more than the layer count:

* Inputs and outputs live in asinh space (asinh(x / 5)), which is linear near
  zero and logarithmic for large values. Quantiles survive a monotone
  transform, so predicting them there and mapping back is exact.
* The network predicts the change from the latest reading, like the tree
  models, so it can follow an episode above anything seen in training.
* Missing hours are forward-filled in the input and flagged in a mask
  channel. Missing targets are left out of the loss.

One global model serves all monitors. A variant adds two input channels: the
mean of neighbours within 300 km, and of those 300 to 800 km away.

**Constants**

| Name | Value |
|---|---|
| `PATCH` | `8` |
| `SCALE` | `5.0` |
| `Q` | `torch.tensor(config.QUANTILES)` |
| `VARIANTS` | `{"Transformer": False, "Transformer + neighbours": True}` |
| `PRESETS` | `{ "cpu": {"d_model": 96, "n_layers": 2, "batch": 256, "steps": 300, "epochs": 25, "pati...` |

#### `to_z(x)`

Map concentrations to the asinh scale the network works in.

#### `from_z(z)`

Inverse of to_z.

#### class `PatchTransformer`

Encoder-only transformer over 8-hour patches, with a flatten head that emits 48 x 9 quantiles.

#### `PatchTransformer.__init__(self, n_channels, n_cov, d_model=128, n_heads=4, n_layers=3, dropout=0.1)`

Build the patch embedding, covariate token, encoder stack and output head.

#### `PatchTransformer.forward(self, x, cov, anchor)`

x [B, C, L], cov [B, n_cov], anchor [B] (latest reading, asinh space) -> [B, H, 9].

#### `pinball_loss(pred, y, mask)`

Quantile (pinball) loss averaged over the nine levels and over observed targets only.

#### class `Windows`

Cuts training and inference windows straight out of the panel matrix.

#### `Windows.__init__(self, bench, use_spatial)`

Precompute the transformed, forward-filled input channels and which (monitor, hour) pairs are eligible.

#### `Windows.pairs(self, split)`

All eligible (monitor, hour) issue pairs whose targets fall inside a split.

#### `Windows.batch(self, s, t, device)`

Cut inputs, covariates, anchors, targets and target masks for the given pairs.

#### `fit(bench, name, log=print, seed=config.SEED, save=True)`

Train one variant with AdamW, a one-cycle schedule and early stopping; return the best model.

`seed` sets the weight initialisation and the order of training windows.
The validation sample is the same for every seed, so that runs differ
only in what the seed is meant to vary. `save=False` leaves the stored
weights of the main run untouched.

#### `predict(bench, model, w, t_idx, batch=4096)`

Quantile forecasts on the benchmark grid, mapped back to concentrations.

#### `run(bench, log=print, names=None)`

Train, predict and store each transformer variant and record its training history.

## `src/smokeshift/panel.py`

136 lines.

Turn the raw EPA files into one monitor-by-hour panel.

Raw: about 27 million rows, one per monitor-instrument-hour, in local and
GMT time. Output: a dense matrix (monitors x hours, GMT) holding only monitors
that reported consistently in every year, plus a table describing them.

Decisions made here, and why:
* GMT timestamps, so that "the same hour" means the same instant everywhere.
  Spatial features depend on that.
* Collocated instruments at one site are averaged.
* Readings outside a physical range are treated as missing. Small negative
  values are kept: they are instrument noise around zero, and deleting them
  would bias clean-air averages upwards.
* Nothing is imputed. A missing hour stays missing.

**Constants**

| Name | Value |
|---|---|
| `COLS` | `["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude", "Date GMT", "...` |
| `EPOCH` | `pd.Timestamp(f"{config.YEARS[0]}-01-01")` |
| `N_HOURS` | `int((pd.Timestamp(f"{config.YEARS[-1] + 1}-01-01") - EPOCH) / pd.Timedelta(hours=1))` |
| `NOT_CONUS` | `{"02", "15", "72", "78", "80", "CC", "66"}` |
| `REGIONS` | `{ "Northeast": ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island...` |
| `STATE_REGION` | `{s: r for r, states in REGIONS.items() for s in states}` |

#### `_read_year(year, log)`

One year of raw rows -> (site, hour index, value) with instruments averaged.

#### `build(log=print)`

Read all years, pivot to a dense matrix and keep the monitors that pass the completeness rule.

#### `save(values, stations, summary)`

Write panel.npz, stations.csv and panel_summary.json to data/processed.

#### `load()`

Return (values [monitors x hours, float32 with NaN], hours, stations).

#### `run(log=print)`

Build and save the panel.

## `src/smokeshift/pipeline.py`

132 lines.

The stages of the benchmark, in the order they run.

**Constants**

| Name | Value |
|---|---|
| `ENSEMBLES` | `{ "Ensemble": ["XGBoost + neighbours", "Transformer + neighbours"], "Ensemble + foundat...` |
| `STAGES` | `{ "download": ("Fetch raw EPA hourly PM2.5 files", download.run), "panel": ("Build and ...` |

#### `naive_stage(log=print)`

Fit damped persistence on the training period and store the three reference forecasts.

#### `tune_stage(log=print)`

Random search for the tree models on the validation period.

#### `gbm_stage(log=print)`

Train and store the three XGBoost variants.

#### `neural_stage(log=print)`

Train and store the two transformer variants.

#### `foundation_stage(log=print)`

Run the zero-shot foundation models that are installed.

#### `ensemble_stage(log=print)`

Average the quantiles of each ensemble's members (equal weights, nothing fitted).

#### `evaluate_stage(log=print)`

Score every stored forecast.

#### `eda_stage(log=print)`

Shift statistics and exploratory figures.

#### `alerts_stage(log=print)`

Next-day unhealthy-air alerts.

#### `uncertainty_stage(log=print)`

Raw, static-conformal and adaptive-conformal intervals for every model.

#### `seeds_stage(log=print)`

Retrain the transformers with several random seeds and measure the spread of their scores.

#### `figures_stage(log=print)`

Draw the result charts.

#### `report_stage(log=print)`

Write outputs/report.html.

#### `dashboard_stage(log=print)`

Write the dashboard data bundle and the standalone page.

## `src/smokeshift/plotting.py`

70 lines.

One chart style for every figure.

Colours follow the entity, not the chart: a model family keeps its colour in
every figure, and a variant with neighbour features is the same colour drawn
solid where the own-history variant is dashed.

**Constants**

| Name | Value |
|---|---|
| `SEQ` | `LinearSegmentedColormap.from_list("seq_blue", ["#eef4fd", "#9ec5f4", "#2a78d6", "#0d366...` |
| `SEQ_WARM` | `LinearSegmentedColormap.from_list("seq_warm", ["#fdf1ea", "#f5b08f", "#eb6834", "#8a2f0...` |
| `DIV` | `LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#e...` |
| `MODEL_STYLE` | `{ "Persistence": (MUTED, ":"), "Seasonal naive": (MUTED, "-."), "Damped persistence": (...` |

#### `style(model)`

Colour and line style for a model name.

#### `setup()`

Apply the shared matplotlib settings.

#### `save(fig, name)`

Save a figure to outputs/figures and close it; returns the file name.

## `src/smokeshift/report.py`

616 lines.

Builds outputs/report.html: one self-contained page with every result.

Every number in the text is read from the tables the pipeline wrote, so the
write-up cannot drift away from what was computed.

**Constants**

| Name | Value |
|---|---|
| `CSS` | `""" :root{--bg:#f7f8f8;--surface:#fdfdfc;--ink:#101314;--ink2:#4a5256;--muted:#7f888c;-...` |
| `NAIVE` | `["Persistence", "Seasonal naive", "Damped persistence"]` |
| `FOUNDATION` | `["Chronos-Bolt", "Chronos-Bolt (long context)", "Chronos-2", "Chronos-2 (long context)"...` |
| `ENSEMBLES` | `["Ensemble", "Ensemble + foundation"]` |
| `DESCRIPTIONS` | `{ "Persistence": "The latest reading, repeated for 48 hours", "Seasonal naive": "The sa...` |

#### `_img(name)`

An <img> tag with the PNG embedded as base64.

#### `_fig(name, caption)`

A figure block: embedded image plus caption.

#### `_table(df, formats=None, left=(), highlight=None)`

An HTML table from a DataFrame, with per-column number formats and an optional row highlight.

#### `_json(name)`

Read a JSON table from outputs/tables.

#### `_csv(name)`

Read a CSV table from outputs/tables.

#### `build(log=print)`

Read every result table, compute the numbers quoted in the text and write the page.

## `src/smokeshift/seeds.py`

81 lines.

How much does the transformer's score move from one training run to the next?

A neural network trained twice with different random seeds gives two slightly
different models. Until that spread is measured, a small gap between two
models on a leaderboard cannot be called a difference. This stage trains each
transformer variant with several seeds and scores every run on the same cells
as the main benchmark, so the leaderboard gaps can be read against it.

The forecasts of these extra runs are scored and then discarded (each would
take 0.75 GB); only the score tables are kept. The leaderboard entry for each
variant remains the single run stored by the `neural` stage.

**Constants**

| Name | Value |
|---|---|
| `SEEDS` | `[7, 11, 23, 37, 53]` |
| `SLICES` | `["All hours", "Calm at issue", "Smoke at issue"]` |

#### `score_run(point, quantiles, y, reg, scale, day1, base, calm, name="run")`

Benchmark scores of one set of forecasts: MAE and WQL by slice, and day-1 alert PR-AUC.

#### `summarise(runs)`

Mean, standard deviation, minimum and maximum of every score, per model.

#### `run(bench=None, log=print, seeds=SEEDS)`

Train every transformer variant once per seed, score each run and write the two seed tables.

## `src/smokeshift/spatial.py`

97 lines.

What nearby and upwind monitors are reading right now.

Smoke travels. A plume over Minnesota today is over Michigan tomorrow, and a
model that only sees one monitor's history cannot know it is coming. These
features summarise the current state of the monitors around each site, at
several distances and in four compass quadrants, using only readings taken at
or before the issue time.

**Constants**

| Name | Value |
|---|---|
| `EARTH_KM` | `6371.0` |
| `QUADRANTS` | `{"ne": (0, 90), "se": (90, 180), "sw": (180, 270), "nw": (270, 360)}` |

#### `distances_and_bearings(lat, lon)`

Great-circle distance (km) and initial bearing (degrees from north) between all pairs.

#### `masked_mean(W, V)`

Weighted mean over neighbours that have a reading; NaN if none do.

#### `masked_max(M, V)`

Largest reading among each site's neighbours (boolean matrix M).

#### class `Neighbours`

Pairwise distances and bearings between monitors, with helpers that select neighbours.

#### `Neighbours.__init__(self, stations)`

Compute the distance and bearing matrices once.

#### `Neighbours.band(self, lo_km, hi_km, sector=None)`

Boolean matrix: j is within [lo, hi) km of i, optionally in a compass quadrant.

#### `Neighbours.idw(self, lo_km, hi_km)`

Inverse-distance weights within a band (not normalised; masked_mean does that).

#### `features(bench, t_idx, nb=None)`

Spatial features at each issue time: ([monitors, issues, F], names).

#### `hourly_channels(bench, nb=None)`

Two neighbour series at every hour, for sequence models: within 300 km, and 300-800 km.

## `src/smokeshift/store.py`

46 lines.

Forecasts on disk: one folder per model, all on the same grid.

point.npy      [monitors, issue days, horizon]            the forecast scored with MAE
quantiles.npy  [monitors, issue days, horizon, levels]    optional, deciles 0.1 to 0.9, sorted
info.json      how the forecast was made

Arrays are memory-mapped on load. A quantile array is about 680 MB, so the
scoring code reads one model at a time and never holds two in memory.

#### `_dir(name, split)`

Folder holding one model's forecast.

#### `save(name, point, quantiles=None, info=None, split="test")`

Write a model's point forecast, optional quantiles and metadata.

#### `load(name, split="test")`

Memory-map a model's forecast; `quantiles` is None for point-only models.

#### `available(split="test")`

Names of all models with a stored forecast.

## `src/smokeshift/uncertainty.py`

47 lines.

Are the 80% intervals honest, and do they stay honest when smoke arrives?

#### `run(bench=None, log=print)`

Calibrate every model's interval three ways; write coverage, width and interval score by slice and month.

## `scripts/build_api_reference.py`

113 lines.

Regenerate docs/API_REFERENCE.md from the source code.

    python scripts/build_api_reference.py          write the file
    python scripts/build_api_reference.py --check  exit 1 if the file is out of date

The reference lists every module, constant, class and function with its
signature and docstring, read straight from the code with `ast`, so it cannot
describe something that no longer exists. A test runs the --check mode, which
means a function added without updating the reference fails the test suite.

**Constants**

| Name | Value |
|---|---|
| `ROOT` | `Path(__file__).resolve().parents[1]` |
| `OUT` | `ROOT / "docs" / "API_REFERENCE.md"` |
| `SOURCES` | `[ROOT / "run.py"] + sorted((ROOT / "src" / "smokeshift").glob("*.py")) + [ ROOT / "scri...` |

#### `_signature(node, source)`

`name(arguments)` as written in the source.

#### `_doc(node, indent="")`

The docstring as indented markdown text, or a marker if there is none.

#### `_constants(tree, source)`

Top-level names assigned in a module, with their source text when it is short.

The value is copied from the source file, not re-rendered from the syntax
tree, so the output is the same on every Python version.

#### `_arguments(node, source)`

The argument list of a function exactly as written, on one line.

#### `_functions(body, source, prefix="")`

Markdown for every function and class in a body, in source order, recursing into classes.

#### `build()`

Return the full text of the reference.

#### `main()`

Write the reference, or with --check compare it with what is on disk.

## `tests/test_smokeshift.py`

211 lines.

Checks on the benchmark protocol. They run on a small synthetic panel, so
no download is needed:

    python tests/test_smokeshift.py        (or: pytest)

The most important ones corrupt everything after a cut-off hour and assert
that nothing a model is allowed to see before the cut-off changes.

**Constants**

| Name | Value |
|---|---|
| `CUT` | `2400` |
| `T_IDX` | `np.arange(L + 30, CUT + 1, 24)` |

#### `synthetic_bench(n_sites=14, n_days=150, seed=0)`

A small panel with a daily cycle, one travelling 'plume' and missing readings.

#### `corrupt_after(bench, cut)`

A copy of the panel with every value after hour `cut` replaced by huge random numbers.

#### `same(a, b)`

Assert two arrays are identical, treating NaN as equal to NaN.

#### `test_tabular_features_ignore_the_future()`

Tree features at or before the cut-off must not change when the future is corrupted.

#### `test_spatial_features_ignore_the_future()`

Neighbour features and channels at or before the cut-off must not change either.

#### `test_naive_forecasts_ignore_the_future()`

The three reference forecasts must depend only on the past.

#### `test_transformer_inputs_ignore_the_future()`

The transformer's inputs, covariates and anchor must depend only on the past.

#### `test_targets_are_strictly_after_the_issue_time()`

Targets run from one hour to 48 hours after the issue time; the context ends at it.

#### `test_real_splits_do_not_overlap()`

On the real panel, train, validation and test target windows are disjoint and in order.

#### `test_fill_context_only_looks_back()`

Gap filling carries earlier values forward; only a leading gap takes the first reading.

#### `test_adaptive_conformal_recovers_coverage_after_a_shift()`

When noise triples half way, adaptive intervals return to 80% coverage and static ones do not.

#### `test_pinball_and_interval_score()`

The quantile loss and the interval score match hand calculations.

#### `test_benjamini_hochberg_known_values()`

The false-discovery-rate adjustment matches a worked example.

#### `test_panel_storage_round_trip()`

Saving and loading the panel changes values by at most the 0.1 rounding.

#### `test_distance_matrix()`

Distances and bearings match known values for one degree of latitude and longitude at 40N.

#### `test_seed_summary_statistics()`

The seed summary reports mean, sample standard deviation, range and run count per model and metric.

#### `test_api_reference_is_current()`

docs/API_REFERENCE.md must match what the generator produces from the code as it is now.

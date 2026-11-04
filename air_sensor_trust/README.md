# Can we trust cheap air quality sensors?

Calibration, drift, forecasting and honest uncertainty for a low-cost air quality sensor box, built on a year of
hourly roadside measurements taken next to reference-grade analysers (UCI Air Quality dataset, De Vito et al., 2008).

This is a rebuild of an earlier course project ("Air Quality Prediction Using Machine Learning Approaches"). It keeps
the same data and domain and changes the question, the evaluation and the methods. The earlier files in the parent
folder are untouched.

**Start here:** open `outputs/report.html` in a browser. It is a single self-contained page with every figure,
table and finding.

## Why the project was reframed

The original notebook reported R² of 0.99. Re-running its preprocessing line by line shows that the score did not
measure prediction:

| Check on the original pipeline | Result |
|---|---|
| R² of plain linear regression using only the three columns the target was built from | 1.000 |
| Hours above the 200 µg/m³ NO2 limit deleted by the IQR "outlier" filter | 282 of 386 (73%) |
| Top 5% of CO hours deleted by the same filter | 92% |
| Missing-value markers (-200) that stayed in the CO column as real numbers | 1,683 |
| Missing NO2 readings silently replaced by the previous value | 1,642 |

The "hazard score" target was a weighted sum of CO, NO2 and benzene, and those three columns were among the
features. The outlier filter then removed most of the pollution episodes. `src/airtrust/audit.py` reproduces all
of this.

The new framing asks questions that low-cost sensor networks face today, and uses the part of the dataset the
original ignored: the cheap sensor channels sitting beside certified instruments.

## Findings

All numbers come from `python run_all.py` and are saved in `outputs/results.json`.

1. **Missing data is scheduled, not random.** The NO2 analyser has no reading at 03:00 on 93% of days (a nightly
   self-calibration; chi-square p < 0.001, Cramér's V 0.42). The target is therefore never imputed.
2. **The "NO2 sensor" does not measure NO2.** Its rank correlation with reference NO2 is 0.15. Used alone it is
   worse than predicting the mean (R² -0.66). Combining all five channels with temperature and humidity gives
   an error of 20.7 µg/m³, 18% of the mean.
3. **Calibrations go stale, unevenly.** A NOx calibration fitted on the first 8 weeks has an error of 160 ppb over
   the rest of the year. Refitting every 4 weeks on the latest 8 weeks cuts that by 65%, to 55 ppb.
   Refitting barely helps NO2 (20.7 to 19.9).
4. **Input drift alarms are not performance alarms.** PSI exceeded the usual 0.25 alarm level in 11 of 12 four-week
   blocks and a domain classifier separated every block from the co-location period (AUC at least 0.93), yet
   neither ranked the blocks by actual calibration error. Seasons move the inputs whether or not the model still
   works.
5. **Forecasts beat naive rules, modestly and measurably.** Six hours ahead, ridge regression has an MAE of 25.4
   µg/m³ against 32.3 for "same hour yesterday", an improvement of 6.9 (95% day-block bootstrap interval 5.4 to
   8.6). Ridge matches or beats gradient-boosted trees beyond one hour, because the test months are more polluted
   than anything the trees saw in training.
6. **Models are weakest where it matters.** On hours above the limit the 6-hour forecast is 47 µg/m³ too low on
   average.
7. **"90%" intervals covered 78% until made adaptive.** Raw quantile regression covered 78%. Standard conformal
   calibration reached 86% overall but 62% in its worst month. Adaptive conformal prediction reached 89.8% and
   never fell below 88% in any month, at the cost of wider intervals.
8. **Probabilistic alerts catch three times more exceedances than a threshold on the current reading.** Six hours
   ahead: 70% recall at 37% precision, against 22% recall for alerting when the current value is over the limit.
9. **Anomalies depend on context.** 62% of the hours flagged as unusual for their time of day and season sit
   inside the fixed IQR range the original project used, and 79% of that range's "outliers" are ordinary in
   context.
10. **Days cluster into two shapes** without being told the weekday: a commuter pattern (95% weekdays) and a
    late-evening pattern (67% weekend days).

## Run it

```
pip install -r requirements.txt
python run_all.py                  # everything, about 5 minutes
python run_all.py eda patterns     # rerun chosen stages, reuse saved results for the rest
python build_report.py             # rebuild the report from saved results
python tests/test_no_leakage.py    # 5 checks, including "features cannot see the future"
```

Python 3.10 or later. Only numpy, pandas, scipy, scikit-learn and matplotlib are needed.

## Layout

```
air_sensor_trust/
  run_all.py              one command for the whole analysis
  build_report.py         report only
  data/AirQuality.csv     copy of the raw UCI file
  src/airtrust/
    config.py             paths, limit value, horizons, seeds
    data.py               loading and missing-data audit (no imputation)
    audit.py              line-by-line reproduction of the original preprocessing
    eda.py                seasonality, hypothesis tests, autocorrelation, PCA
    calibration.py        sensor calibration, refit schedules, drift monitors
    forecasting.py        features, baselines, walk-forward evaluation, conformal intervals, alerts
    patterns.py           contextual anomalies, daily regime clustering
    plotting.py           shared chart style
    report.py             builds outputs/report.html from the results
  tests/test_no_leakage.py
  outputs/
    report.html           the write-up
    results.json          every reported number
    figures/              12 charts (PNG)
    tables/               result tables (CSV)
    predictions/          row-level forecasts, intervals and alert probabilities per horizon
```

## Methods, and where to find them

| Concept | Where |
|---|---|
| Missing-data mechanisms, sentinel values | `data.py`, `audit.py` |
| Hypothesis tests, multiple-comparison correction, bootstrap intervals, effect sizes | `eda.py`, `forecasting._block_bootstrap_diff` |
| PCA, rank correlation, autocorrelation | `eda.py` |
| Linear, ridge and gradient-boosted regression | `calibration.py`, `forecasting.py` |
| Walk-forward evaluation, gap-aware time-series CV, naive baselines | `forecasting.py` |
| Target leakage, and tests that guard against it | `audit.py`, `tests/test_no_leakage.py` |
| Distribution shift: PSI, KS statistic, domain classifier | `calibration.drift_signals` |
| Quantile regression, conformalized quantile regression, adaptive conformal inference | `forecasting.py` |
| Rare-event classification: PR-AUC, Brier skill, Platt scaling, cost-based threshold | `forecasting.alert_metrics` |
| Robust anomaly detection, k-means with silhouette selection | `patterns.py` |

## Limits

- One station and one year (March 2004 to April 2005). There is a single winter, so seasonal conclusions rest on
  one cycle and may not transfer to other sites.
- No wind, rain or mixing-height data, which are the main drivers of dispersion.
- The forecast uses reference NO2 as an input. A sensor-only deployment would use calibrated sensor values and
  inherit their larger error.
- Prediction intervals are honest but wide (about 118 µg/m³ at 6 hours).

## References

- S. De Vito et al., "On field calibration of an electronic nose for benzene estimation in an urban pollution
  monitoring scenario," Sensors and Actuators B: Chemical, 129(2), 2008. Dataset source.
- S. De Vito et al., "CO, NO2 and NOx urban pollution monitoring with on-field calibrated electronic nose by
  automatic Bayesian regularization," Sensors and Actuators B: Chemical, 143(1), 2009.
- Y. Romano, E. Patterson, E. Candès, "Conformalized quantile regression," NeurIPS 2019.
- I. Gibbs, E. Candès, "Adaptive conformal inference under distribution shift," NeurIPS 2021.
- S. Rabanser, S. Günnemann, Z. Lipton, "Failing loudly: an empirical study of methods for detecting dataset
  shift," NeurIPS 2019.
- S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," Patterns,
  4(9), 2023.
- Directive 2008/50/EC on ambient air quality and cleaner air for Europe (hourly NO2 limit value, 200 µg/m³).

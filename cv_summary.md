# Summary lines for a CV or LinkedIn

This folder is ignored by git and is not published.

- Built a reproducible benchmark of 16 forecasting methods (GPU gradient-boosted trees, a PyTorch patch
  transformer, zero-shot foundation models from Amazon and Google, ensembles) on 35.6 million EPA air quality
  readings, using the 2023 wildfire-smoke season as a natural distribution shift; best model cut error 14%
  against the strongest baseline.
- Showed that outcome-selected evaluation reverses the leaderboard (plain persistence ranks 1st or 15th of 16
  depending only on how the test set is sliced), and designed forecast-time strata, day-block bootstrap intervals,
  automated leakage tests and a five-seed variance study to prevent it.
- Found that zero-shot foundation models beat naive baselines by up to 6% but trail locally trained models by
  4.5% on the same input, and that spatial context makes early warning of unhealthy-air onsets 1.7 times better.

Shorter, one line:

- Benchmarked trees, a transformer and Amazon and Google foundation models on 35.6M EPA air quality readings
  under wildfire-smoke distribution shift; exposed an evaluation trap that reverses the leaderboard.

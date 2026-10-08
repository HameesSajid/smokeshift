# Should this run on Google Colab?

**Short answer: not needed for the benchmark as it stands, worth it for the extensions.**

Everything in the published results ran on a laptop with a GTX 1660 Ti (6 GB). A bigger GPU would not make any of
those results more accurate, because none of them is limited by compute. It would make them faster, and it would
make practical some additional experiments that are too slow on the laptop.

## What the laptop GPU already handles

Measured on the GTX 1660 Ti:

| Stage | Time on the laptop GPU | Limited by |
|---|---|---|
| Tree tuning (10 settings) | 33 minutes | number of settings tried |
| Three tree models | 18 minutes | data size |
| Two transformers (1.6 million parameters) | 6 minutes | stops early: validation loss stops improving after 3 to 5 epochs |
| Chronos-Bolt, 168-hour context | 10 minutes | 375,426 forecasts |
| Chronos-2, 168-hour context | 14 minutes | same |
| TimesFM 2.5, 168-hour context | 33 minutes | same |
| Chronos-2, 512-hour context | 34 minutes | same |
| Chronos-Bolt and TimesFM, 512-hour context | 67 and 73 minutes, measured while two jobs shared the GPU | same |
| Ten transformer runs for the seed experiment | 86 minutes while sharing the GPU; about 30 alone | number of runs |

Two results here are themselves reasons not to expect much from more compute.

- The transformer was trained twice: a 1.1-million-parameter model on the CPU and a 1.6-million-parameter model on
  the GPU. The two reached the same validation loss (0.1315 and 0.1314 with neighbours). The model is not short
  of capacity.
- Adding two pretrained foundation models, each about a hundred times larger than the transformer, to the
  ensemble made it slightly worse.

What every method lacks is information about smoke that has not arrived yet, and no GPU supplies that.

## Where a Colab GPU would help

Two things that were on this list have since been done on the laptop: five random seeds per transformer, and all
six foundation-model variants. Together they took about three hours of GPU time overnight. What remains, in order
of how much each would add:

1. **A longer context for the trained models.** Three weeks of history improved every foundation model by about
   3%. The trees and the transformer still see one week. Giving them three would show whether the same gain is
   available to them, and needs retraining with larger inputs.
2. **A larger transformer and a wider search.** More layers, a sweep over learning rates. Useful mainly to confirm
   that size is not the limit.
3. **Fine-tuning a foundation model** on the training period. This needs more memory than 6 GB for comfortable
   batch sizes and is the first thing here that the laptop cannot do well.
4. **A graph neural network** over the monitor network.
5. **More seeds, and seeds for the tree models.** Five runs give a usable spread; twenty would give a tighter one.

All of these are new work except the last, which is `python run.py seeds` with a longer list in `seeds.SEEDS`.

## What you need to do

1. Open [colab.research.google.com](https://colab.research.google.com), choose **File > Open notebook > GitHub**,
   enter `HameesSajid/smokeshift` and open `notebooks/smokeshift_colab.ipynb`. For a private repository, tick
   "Include private repos" and sign in to GitHub when asked.
2. **Runtime > Change runtime type**, pick a GPU. T4 is enough. L4 or A100 is faster.
3. While the repository is private, the notebook needs permission to clone it. Create a GitHub personal access
   token with read access to this one repository, and add it in Colab's **Secrets** panel (the key icon) under the
   name `GITHUB_TOKEN`, with notebook access switched on. If the repository is public, skip this step.
4. Run the cells from top to bottom. In section 2 run only one of the two storage cells.
5. Forecasts are not stored in the repository, so for a full comparison leave the model list in the
   foundation-model cell as it is and run every cell. Remove names from it to save time.
6. The last cell downloads `smokeshift_results.zip`. Unzip it over the project folder on your computer, then
   commit and push.

Everything the notebook does is also plain command-line usage, so the same steps work on any Linux machine with a
GPU:

```
git clone https://github.com/HameesSajid/smokeshift && cd smokeshift
pip install chronos-forecasting "timesfm[torch]" xgboost
python run.py naive gbm neural foundation ensemble evaluate uncertainty alerts figures report dashboard
```

The cleaned panel (17 MB) is in the repository, so the EPA download and the panel build are not needed.

## A caution about comparing runs

Results from different machines are not bit-for-bit identical. GPU arithmetic, library versions and the order of
floating-point operations differ. The two transformer runs in this project, on different hardware and with
different model sizes, differed by 0.01 µg/m³ in overall error. Differences of that size between a laptop run and
a Colab run are not findings. The seed experiment puts a number on this: across five seeds the transformer's
overall error has a standard deviation of 0.02 µg/m³.

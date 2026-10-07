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

Two results here are themselves reasons not to expect much from more compute.

- The transformer was trained twice: a 1.1-million-parameter model on the CPU and a 1.6-million-parameter model on
  the GPU. The two reached the same validation loss (0.1315 and 0.1314 with neighbours). The model is not short
  of capacity.
- Adding two pretrained foundation models, each about a hundred times larger than the transformer, to the
  ensemble made it slightly worse.

What every method lacks is information about smoke that has not arrived yet, and no GPU supplies that.

## Where a Colab GPU would help

In order of how much each would add to the project:

1. **Several random seeds.** Training the transformer five times with different seeds would put an interval on
   its score, so that "trees and transformer tie" becomes a statement with error bars. At six minutes each this is
   feasible on the laptop too; Colab just makes it painless.
2. **All six foundation-model variants.** Three models with two context lengths each is about two and a half hours
   on the laptop. The long-context run already done shows it matters: Chronos-2 improved by 3.6% with three weeks
   of history instead of one. TimesFM and Chronos-Bolt have not been run with the long context.
3. **A larger transformer and a wider search.** More layers, longer context, a sweep over learning rates. Useful
   mainly to confirm that size is not the limit.
4. **Fine-tuning a foundation model** on the training period. This needs more memory than 6 GB for comfortable
   batch sizes and is the first thing here that the laptop cannot do well.
5. **A graph neural network** over the monitor network.

Items 1 and 2 need no new code. Items 3 to 5 are new work.

## What you need to do

1. Open [colab.research.google.com](https://colab.research.google.com), choose **File > Open notebook > GitHub**,
   enter `HameesSajid/smokeshift` and open `notebooks/smokeshift_colab.ipynb`. For a private repository, tick
   "Include private repos" and sign in to GitHub when asked.
2. **Runtime > Change runtime type**, pick a GPU. T4 is enough. L4 or A100 is faster.
3. While the repository is private, the notebook needs permission to clone it. Create a GitHub personal access
   token with read access to this one repository, and add it in Colab's **Secrets** panel (the key icon) under the
   name `GITHUB_TOKEN`, with notebook access switched on. If the repository is public, skip this step.
4. Run the cells from top to bottom. In section 2 run only one of the two storage cells.
5. To run the two variants that are still missing, set the list in the foundation-model cell to
   `"TimesFM (long context)", "Chronos-Bolt (long context)"`. The other models' forecasts are not in the
   repository, so for a full comparison leave the list as it is and run every cell.
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
a Colab run are not findings. This is another reason item 1 above, several seeds, is the most valuable use of
extra compute.

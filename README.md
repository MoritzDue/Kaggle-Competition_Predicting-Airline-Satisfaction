# Predicting Airline Satisfaction | Kaggle S6E10

**Final rank: 1,249 / 3,735 (top 34%)** · **Private ROC AUC: 0.96177** · **Public ROC AUC: 0.96080**

[Competition](https://www.kaggle.com/competitions/playground-series-s6e10) · [Leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard)

## Project summary

Predicted airline passenger satisfaction from tabular data (699,635 training rows; 299,844 test rows). I compared **CatBoost, XGBoost, LightGBM, RealMLP, feature engineering, and weighted ensembles**, using five-fold stratified cross-validation and ROC AUC. My final entry combined **my feature-engineered LightGBM model** with **publicly shared TabPFN-3.5 predictions** through a logistic stacking model.

## What worked

**Leakage-aware feature engineering was the clearest improvement in my own models.** I investigated the repeated values of `Flight Distance`, creating count encoding and smoothed target encoding. Training target encodings were cross-fitted *within each outer training fold* so validation labels were not used to build features.

| LightGBM features | 5-fold OOF ROC AUC |
|---|---:|
| Original features | 0.958810 |
| + Flight Distance count encoding | 0.959526 |
| + Smoothed target encoding | 0.959974 |
| + Both features | **0.960196** |

The combined features added **+0.001386 AUC** over the original LightGBM. Substituting that model into my tree/RealMLP blend increased OOF AUC from **0.959059 to 0.960059** (positive differences on all five folds).

## Model progression

| Approach | Private leaderboard ROC AUC |
|---|---:|
| CatBoost baseline | 0.95821 |
| CatBoost + XGBoost + LightGBM ensemble | 0.95931 |
| Flight Distance engineered blend | 0.96039 |
| Optimised weighted ensemble | 0.96085 |
| Public TabPFN predictions alone | 0.96161 |
| **LightGBM + public TabPFN logistic stack (final)** | **0.96177** |

**Takeaway:** More extensive blending of my tree and neural models did not outperform the public TabPFN predictions. The small extra gain from combining TabPFN with my improved LightGBM shows that the locally engineered features still contributed useful complementary signal. The final gain over standalone TabPFN was modest and should not be overstated.

## Files

- [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) — exploratory analysis.
- [`notebooks/04_cross_validation_ensemble.ipynb`](notebooks/04_cross_validation_ensemble.ipynb) — tree models and OOF ensembles.
- `experiments/07_realmlp_cv_submission.py` — RealMLP five-fold experiment (copy in the original uploaded script; see `experiments/README.md`).
- [`notebooks/08_flight_distance_experiments.ipynb`](notebooks/08_flight_distance_experiments.ipynb) — feature-engineering ablation and test predictions.
- [`notebooks/13_final_submission_lgb_tabpfn.ipynb`](notebooks/13_final_submission_lgb_tabpfn.ipynb) — stacking using saved prediction arrays.

## Reproduce / limitations

1. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the [Kaggle data page](https://www.kaggle.com/competitions/playground-series-s6e10/data) into `data/raw/` (not committed).
2. Install the packages listed in `requirements.txt` (versions are not locked to the original environment).
3. Copy your original `07_realmlp_cv_submission.py` to `experiments/`, then run notebook **04**, script **07**, and notebook **08** to regenerate the owned-model OOF predictions and the feature-engineering study. These are computationally expensive, and exact scores can vary by environment.
4. Notebook **13** consumes the improved LightGBM OOF/test CSVs from 08 plus externally produced `oof_tabpfn35.npy` and `test_tabpfn35.npy` arrays in `submissions/public_tfm/`. It **does not train TabPFN**. Its historical hyperparameter-selection notebook (Experiment 12) is no longer available; the retained notebook reproduces the fixed stacking configuration *given the inputs*, not the complete original selection process.

**Provenance and test ordering:** The TabPFN arrays were sourced from publicly shared competition materials, probably associated with [this contemporaneous Kaggle TFM discussion](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363); the exact generating notebook/dataset version is no longer known. The `.npy` files do not carry test-row IDs, so test-order alignment cannot be independently verified from the surviving files. The notebook therefore retains submission-writing safety checks. A [first-place retrospective on TFMs and LLMs](https://www.kaggle.com/competitions/playground-series-s6e10/writeups/1st-place-tfms-and-llms-with-tabular-data) is linked for later context; it was **not** used as a competition-time source. Credit for public TabPFN predictions belongs to their original creator, not to this repository.

*Scores and rank above are from the participant's final Kaggle submission history; OOF experiments and private leaderboard scores evaluate different samples.*
